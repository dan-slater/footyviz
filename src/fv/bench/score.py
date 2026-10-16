"""`fv.bench score`: score runs against keyframe labels (BENCHMARK.md 2).

    python -m fv.bench score --run b0-cv-job@2026-10-16-base [--write-results]

Scored keyframes = keyframes marked done in the label page (all of them once verified).
Per clip and repeat:
  T1 detection   greedy IoU >= 0.5 (highest conf first) against player/referee GT; predictions
                 matching an `ignore` box (IoU >= 0.5) are neither TP nor FP. Recall, precision at
                 the method's operating point, recall by GT height (<80 / 80-150 / >150 px),
                 AP50, AP75.
  T3 teams       pred A/B -> GT T1/T2 by Hungarian on matched outfield boxes (one mapping per
                 clip, reused below). Team accuracy (outfield), keeper accuracy (O or own team
                 = correct), false-participant rate (ignore-matched preds given team A/B).
  T4 identity    on matched GT boxes with a P-id: all-pairs same-person P/R/F1 over pairs from
                 different keyframes; keyframe IDF1 (Hungarian GT-id <-> pred-id; unmatched GT
                 = IDFN, unmatched non-ignore preds = IDFP); identity count error per team
                 (pred identities with a matched box on that team + phantoms that persist >= 2 s).
Aggregates: mean over clips (equal weight), grouped by image tier and by kit tier; repeats ->
mean +- sd.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import manifest as M
from .run import RUNS

LABELS = M.DATA / "labels"
RESULTS = M.REPO / "bench" / "RESULTS.md"
H_BUCKETS = [(0, 80, "lt80"), (80, 150, "80_150"), (150, 1e9, "gt150")]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_labels(cid: str, labels: Path = LABELS, only_done: bool = True) -> tuple[dict, dict] | None:
    d = labels / cid
    if not (d / "keyframes.json").exists():
        return None
    kf = json.loads((d / "keyframes.json").read_text())
    clip = json.loads((d / "clip.json").read_text())
    if only_done:
        kf["frames"] = [f for f in kf["frames"] if f.get("done", True)]
    return clip, kf


def label_version(labels: Path, cids: list[str]) -> str:
    h = hashlib.sha1()
    for c in sorted(cids):
        for f in ("clip.json", "keyframes.json"):
            p = labels / c / f
            if p.exists():
                h.update(p.read_bytes())
    return h.hexdigest()[:8]


def match_frame(gt: list[dict], pred: pd.DataFrame):
    """Greedy IoU>=0.5 matching. Returns (pairs [(gi, pi)], ignored pred idx set, unmatched pred idx)."""
    pos = [i for i, b in enumerate(gt) if b["role"] in ("player", "referee")]
    ign = [i for i, b in enumerate(gt) if b["role"] == "ignore"]
    pb = pred[["x1", "y1", "x2", "y2"]].to_numpy(float)
    order = np.argsort(-pred["conf"].to_numpy())
    gtb = np.array([gt[i]["box"] for i in pos], float).reshape(-1, 4)
    igb = np.array([gt[i]["box"] for i in ign], float).reshape(-1, 4)
    I = iou_matrix(pb, gtb)
    Ig = iou_matrix(pb, igb)
    used = set()
    pairs, ignored, fp = [], set(), []
    for p in order:
        best, bi = 0.5, None
        for j in range(len(pos)):
            if j not in used and I[p, j] >= best:
                best, bi = I[p, j], j
        if bi is not None:
            used.add(bi)
            pairs.append((pos[bi], p))
        elif len(ign) and Ig[p].max() >= 0.5:
            ignored.add(p)
        else:
            fp.append(p)
    return pairs, ignored, fp


def ap(scores: list[float], tp: list[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-np.asarray(scores))
    t = np.asarray(tp)[o]
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # VOC-style all-point interpolation
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pair_counts(frames: list, gid: list, pid: list) -> dict:
    """All pairs of boxes from DIFFERENT keyframes: counts of same-pred, same-gt, both."""
    def same_pairs(keys):
        c = Counter(keys)
        return sum(n * (n - 1) // 2 for n in c.values())
    def cross(group_keys):
        return same_pairs(group_keys) - same_pairs([(k, f) for k, f in zip(group_keys, frames)])
    return {"same_pred": cross(pid), "same_gt": cross(gid), "both": cross(list(zip(gid, pid)))}


def score_clip(clip: dict, kf: dict, dets: pd.DataFrame, fps: float) -> dict:
    people = {p["pid"]: p for p in clip["people"]}
    by_frame = {f: g.reset_index(drop=True) for f, g in dets.groupby("frame")}
    det_scores, det_tp, n_gt = [], [], 0
    det_tp75 = []
    hb = {k: [0, 0] for *_, k in H_BUCKETS}
    n_tp = n_fp = 0
    matched = []           # (frame, gt box dict, pred row)
    ign_preds = []         # pred rows matched to ignore
    gt_boxes_id = []       # (frame, pid) for IDF1 GT side
    pred_nonign = []       # (frame, identity_id, matched_gt_pid or None)
    empty = pd.DataFrame(columns=dets.columns)
    for fr in kf["frames"]:
        gt = fr["boxes"]
        pred = by_frame.get(fr["frame"], empty)
        pairs, ignored, fps_ = match_frame(gt, pred)
        n_pos = sum(b["role"] in ("player", "referee") for b in gt)
        n_gt += n_pos
        n_tp += len(pairs); n_fp += len(fps_)
        mg = {p: g for g, p in pairs}
        for p in range(len(pred)):
            if p in ignored:
                continue
            det_scores.append(float(pred.at[p, "conf"]))
            det_tp.append(int(p in mg))
            det_tp75.append(int(p in mg and iou_matrix(pred.loc[[p], ["x1", "y1", "x2", "y2"]].to_numpy(float),
                                                         np.array([gt[mg[p]]["box"]], float))[0, 0] >= 0.75))
        gm = {g for g, _ in pairs}
        for i, b in enumerate(gt):
            if b["role"] not in ("player", "referee"):
                continue
            h = b["box"][3] - b["box"][1]
            for lo, hi, k in H_BUCKETS:
                if lo <= h < hi:
                    hb[k][0] += 1; hb[k][1] += int(i in gm)
            if b.get("pid") not in (None, "unsure"):
                gt_boxes_id.append((fr["frame"], b["pid"]))
        for g, p in pairs:
            matched.append((fr["frame"], gt[g], pred.iloc[p]))
        for p in ignored:
            ign_preds.append(pred.iloc[p])
        for p in range(len(pred)):
            if p in ignored:
                continue
            g = mg.get(p)
            gpid = gt[g].get("pid") if g is not None else None
            pred_nonign.append((fr["frame"], pred.at[p, "identity_id"], gpid if gpid not in (None, "unsure") else None,
                                g is not None))
    out = {"n_kf": len(kf["frames"]), "n_gt": n_gt}
    out["det_recall"] = n_tp / n_gt if n_gt else float("nan")
    out["det_precision"] = n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan")
    out["AP50"] = ap(det_scores, det_tp, n_gt)
    out["AP75"] = ap(det_scores, det_tp75, n_gt)
    for k, (n, t) in hb.items():
        out[f"recall_{k}"] = t / n if n else float("nan")

    # --- T3 teams: Hungarian A/B -> T1/T2 on matched outfield boxes
    def gt_team(b):
        p = people.get(b.get("pid"))
        return (p or {}).get("team") or b.get("team")
    def gt_role(b):
        p = people.get(b.get("pid"))
        return "referee" if b["role"] == "referee" else (p or {}).get("role", "outfield")
    C = np.zeros((2, 2))
    for _, b, r in matched:
        t = gt_team(b)
        if gt_role(b) == "outfield" and t in ("T1", "T2") and r["team"] in ("A", "B"):
            C["AB".index(r["team"]), int(t[1]) - 1] += 1
    ri, ci = linear_sum_assignment(-C)
    tmap = {"AB"[i]: f"T{j + 1}" for i, j in zip(ri, ci)}
    ok = n = 0
    kok = kn = 0
    for _, b, r in matched:
        t, role = gt_team(b), gt_role(b)
        if t not in ("T1", "T2"):
            continue
        pt = tmap.get(r["team"])
        if role == "outfield":
            n += 1; ok += int(pt == t)
        elif role == "keeper":
            kn += 1; kok += int(pt in (None, t))
    out["team_acc"] = ok / n if n else float("nan")
    out["keeper_acc"] = kok / kn if kn else float("nan")
    out["n_keeper_boxes"] = kn
    fpart = sum(r["team"] in ("A", "B") for r in ign_preds)
    out["false_participant_rate"] = fpart / len(ign_preds) if ign_preds else float("nan")
    out["n_ignore_matched"] = len(ign_preds)
    out["team_map"] = tmap

    # --- T4 identity: all pairs over matched GT boxes with a P-id
    ids = [(f, b["pid"], r["identity_id"]) for f, b, r in matched if b.get("pid") not in (None, "unsure")]
    if ids:
        pc = pair_counts([i[0] for i in ids], [i[1] for i in ids], [i[2] for i in ids])
        P = pc["both"] / pc["same_pred"] if pc["same_pred"] else float("nan")
        R = pc["both"] / pc["same_gt"] if pc["same_gt"] else float("nan")
        out.update(pair_precision=P, pair_recall=R,
                   pair_f1=2 * P * R / (P + R) if P + R and not np.isnan(P + R) else float("nan"),
                   n_pairs_same_gt=pc["same_gt"])
    # keyframe IDF1
    gid_list = sorted({p for _, p in gt_boxes_id})
    pid_list = sorted({str(i) for _, i, _, _ in pred_nonign})
    if gid_list and pid_list:
        co = np.zeros((len(gid_list), len(pid_list)))
        gi = {g: k for k, g in enumerate(gid_list)}; pi = {p: k for k, p in enumerate(pid_list)}
        for f, iid, gpid, _ in pred_nonign:
            if gpid is not None:
                co[gi[gpid], pi[str(iid)]] += 1
        r_, c_ = linear_sum_assignment(-co)
        idtp = co[r_, c_].sum()
        out["IDF1"] = 2 * idtp / (len(gt_boxes_id) + len(pred_nonign))
        out["IDP"] = idtp / len(pred_nonign)
        out["IDR"] = idtp / len(gt_boxes_id)
    # identity count error per team
    true = clip.get("true_count", {})
    pred_team_ids = defaultdict(set)
    for _, b, r in matched:
        pt = tmap.get(r["team"])
        if pt and gt_role(b) != "referee":
            pred_team_ids[pt].add(r["identity_id"])
    matched_ids = {r["identity_id"] for _, _, r in matched} | {r["identity_id"] for r in ign_preds}
    span = dets.groupby("identity_id")["frame"].agg(["min", "max"])
    team_of_id = dets.groupby("identity_id")["team"].agg(lambda s: s.mode().iat[0])
    phantoms = defaultdict(int)
    for iid, row in span.iterrows():
        if iid not in matched_ids and (row["max"] - row["min"]) >= 2 * fps:
            pt = tmap.get(team_of_id[iid])
            if pt:
                phantoms[pt] += 1
    for t in ("T1", "T2"):
        if t in true:
            pc_ = len(pred_team_ids[t]) + phantoms[t]
            out[f"count_{t}_true"] = true[t]
            out[f"count_{t}_pred"] = pc_
            out[f"count_err_{t}"] = pc_ - true[t]
    errs = [abs(out[f"count_err_{t}"]) for t in ("T1", "T2") if f"count_err_{t}" in out]
    out["count_abs_err"] = float(np.mean(errs)) if errs else float("nan")
    return out


HEAD = [("det_recall", "det R", "{:.3f}"), ("det_precision", "det P", "{:.3f}"), ("AP50", "AP50", "{:.3f}"),
        ("AP75", "AP75", "{:.3f}"), ("recall_lt80", "R<80", "{:.2f}"), ("team_acc", "team acc", "{:.3f}"),
        ("keeper_acc", "GK acc", "{:.2f}"), ("false_participant_rate", "false part.", "{:.2f}"),
        ("pair_precision", "pair P", "{:.3f}"), ("pair_recall", "pair R", "{:.3f}"),
        ("pair_f1", "pair F1", "{:.3f}"), ("IDF1", "kf IDF1", "{:.3f}"), ("count_abs_err", "|count err|", "{:.1f}")]


def fmt(vals: list[float], f: str) -> str:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "–"
    s = f.format(np.mean(v))
    return s + (f" ± {f.format(np.std(v, ddof=1))}" if len(v) > 1 else "")


def score_run(name: str, labels: Path = LABELS) -> dict:
    root = RUNS / name
    rj = json.loads((root / "run.json").read_text())
    m = M.load()
    per = {}
    for cid in rj["clips"]:
        lab = load_labels(cid, labels)
        if lab is None:
            continue
        clip, kf = lab
        if not kf["frames"]:
            continue
        fps = float(eval(clip["fps"])) if isinstance(clip["fps"], str) else float(clip["fps"])
        reps = sorted((root / cid).glob("r*/dets.parquet"))
        per[cid] = [score_clip(clip, kf, pd.read_parquet(p), fps) for p in reps]
    lv = label_version(labels, list(per))
    res = {"run": name, "commit": rj["commit"], "date": rj["date"], "config_hash": rj["config_hash"],
           "method": rj["method"], "batch": rj["batch"], "label_version": lv,
           "scored_at": time.strftime("%Y-%m-%d %H:%M"), "clips": per,
           "tiers": {cid: {"image": M.match(m, cid).get("image_tier"), "kit": M.match(m, cid).get("kit_tier"),
                           "eye": M.match(m, cid)["eyeball_tier"]} for cid in per}}
    (root / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def table(results: list[dict]) -> str:
    """Markdown: one row per run x (clip | image tier | kit tier | overall)."""
    lines = ["| run | group | n kf | " + " | ".join(h for _, h, _ in HEAD) + " |",
             "|---|---|---|" + "---|" * len(HEAD)]
    for r in results:
        per, tiers = r["clips"], r["tiers"]
        groups = [(c, [c]) for c in sorted(per)]
        for axis in ("image", "kit"):
            vals = sorted({tiers[c][axis] for c in per})
            groups += [(f"{axis}:{v}", [c for c in per if tiers[c][axis] == v]) for v in vals]
        groups.append(("**overall**", list(per)))
        for g, cs in groups:
            if not cs:
                continue
            nkf = sum(per[c][0]["n_kf"] for c in cs)
            cells = []
            for key, _, f in HEAD:
                # clip mean per repeat, then mean +- sd over repeats
                nrep = min(len(per[c]) for c in cs)
                vals = [np.nanmean([per[c][k].get(key, np.nan) for c in cs]) for k in range(nrep)]
                cells.append(fmt(vals, f))
            lines.append(f"| {r['run']} | {g} | {nkf} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(runs: list[str], write: bool = False, labels: Path = LABELS) -> int:
    results = [score_run(n, labels) for n in runs]
    results = [r for r in results if r["clips"]]
    if not results:
        print("nothing scored: no labelled (done) keyframes for these runs' clips yet")
        return 1
    md = table(results)
    print(md)
    if write:
        hdr = []
        for r in results:
            rj = json.loads((RUNS / r["run"] / "run.json").read_text())
            hdr.append(f"- **{r['run']}**: date {r['date']}, commit `{r['commit']}`, config `{r['config_hash']}` "
                       f"({json.dumps(rj['config'], separators=(',', ':'))}), labels `{r['label_version']}`, "
                       f"repeats {rj['repeats']}, scored {r['scored_at']}")
        block = f"\n## {time.strftime('%Y-%m-%d %H:%M')} — {', '.join(r['run'] for r in results)}\n\n" + \
            "\n".join(hdr) + "\n\n" + md + "\n"
        if not RESULTS.exists():
            RESULTS.write_text("# footyviz-bench results\n\nWritten by `python -m fv.bench score --write-results`. "
                               "Metric definitions: BENCHMARK.md §2. Values are mean over clips; ± is sd over repeats.\n")
        with RESULTS.open("a") as f:
            f.write(block)
        print(f"appended to {RESULTS}")
    return 0
