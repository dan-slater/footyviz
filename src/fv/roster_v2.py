"""M4 roster collapse — multi-view grids + roster-prior grouping + pairwise verify.

Prior fusions over-split (24 ids for a ~7-8-per-team match). Three techniques,
all with a hard time-overlap veto (a player can't be in two places at once):

  1. MULTI-VIEW GRID per tracklet: 4-6 crops spread across the tracklet's
     lifespan (sharpest per temporal bin), tiled + captioned with the track_id.
     This grid represents the tracklet to the VLM.
  2. ROSTER-PRIOR GROUPING: per team, hand gemini-3.1-pro ALL the team's grids at
     once with the prior "a team has ~7-8 players" and jersey numbers as hard
     anchors; it returns groups.
  3. PAIRWISE VERIFY: while a team has > ~8 groups, show the two most-similar
     groups' grids side by side and ask same/different; merge on confident same.

Goalkeepers (from the participant filter) are kept as their own identity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CELL = (110, 172)


def build_grids(video: Path, df: pd.DataFrame, lo: int, hi: int, scale: float = 1.5,
                per_track: int = 6, samples: int = 18, min_h: int = 60):
    """Per track: sharpest crop per temporal bin -> one captioned multi-view grid."""
    from .identity import _sample_frames

    needed: dict[int, list] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        for f in _sample_frames(g["frame"].to_numpy(), samples):
            r = g[g["frame"] == f].iloc[0]
            needed.setdefault(int(f), []).append(
                (int(tid), (r.x1 * scale, r.y1 * scale, r.x2 * scale, r.y2 * scale)))
    cand: dict[int, list] = {}
    cap = cv2.VideoCapture(str(video))
    idx = -1
    while idx < hi:
        if not cap.grab():
            break
        idx += 1
        items = needed.get(idx)
        if items is None:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        Hf, Wf = frame.shape[:2]
        for tid, (x1, y1, x2, y2) in items:
            x1i, y1i, x2i, y2i = max(0, int(x1)), max(0, int(y1)), min(Wf, int(x2)), min(Hf, int(y2))
            if y2i - y1i < min_h or x2i - x1i < int(min_h * 0.3):
                continue
            crop = frame[y1i:y2i, x1i:x2i]
            sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            cand.setdefault(tid, []).append((idx, sharp, crop.copy()))
    cap.release()

    grids, chosen = {}, {}
    for tid, lst in cand.items():
        lst.sort(key=lambda t: t[0])                       # by frame (time)
        chunks = np.array_split(np.arange(len(lst)), min(per_track, len(lst)))
        picks = [max((lst[i] for i in ch), key=lambda t: t[1])[2]
                 for ch in chunks if len(ch)]              # sharpest per temporal bin
        chosen[tid] = picks
        cells = [cv2.resize(c, CELL) for c in picks]
        cols = 3
        while len(cells) % cols:
            cells.append(np.full((CELL[1], CELL[0], 3), 40, np.uint8))
        rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
        grid = np.vstack(rows)
        band = np.full((30, grid.shape[1], 3), 15, np.uint8)
        cv2.putText(band, f"track {tid}", (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        grids[tid] = np.vstack([band, grid])
    return grids, chosen


_P_ROSTER = (
    "These images are multi-view grids of tracked people from ONE team in an "
    "amateur NIGHT 5-a-side match; each grid is captioned with its track_id and "
    "shows the SAME tracklet from several moments (different poses/angles). "
    "A team has about {target} players total. Some track_ids are the SAME person "
    "in different clips. Group ALL track_ids into approximately {target} players.\n"
    "Rules: jersey NUMBERS are HARD anchors — same visible number => same player; "
    "clearly different numbers => different players. Otherwise use hair, socks, "
    "shoes, build, skin. Dark/plain kits are normal (night footage).\n"
    "Think step by step, then return STRICT JSON: "
    '{{"groups": [[id,id,...], ...], "reasoning": "<short>"}}. '
    "Every listed track_id must be in exactly one group. Aim for ~{target} groups "
    "unless the evidence truly forbids it.\nTrack ids and known jersey numbers: {hints}"
)


def _parse_groups(txt: str, ids: list[int]):
    from .llm import _parse_json
    d = _parse_json(txt)
    groups = d.get("groups", []) if isinstance(d, dict) else []
    seen, clean = set(), []
    for g in groups:
        gg = [int(x) for x in g if isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit()
              and int(x) in ids and int(x) not in seen]
        for x in gg:
            seen.add(x)
        if gg:
            clean.append(gg)
    for t in ids:
        if t not in seen:
            clean.append([t])
    return clean, (d.get("reasoning", "") if isinstance(d, dict) else "")[:300]


def _overlap(a_ids, b_ids, frames):
    fa = set().union(*[frames[t] for t in a_ids])
    fb = set().union(*[frames[t] for t in b_ids])
    return bool(fa & fb)


def _enforce_overlap(groups, frames, vetoes):
    """Split any group whose members coexist; record which pairs forced a split."""
    out = []
    for g in groups:
        comps = [[g[0]]]
        for t in g[1:]:
            placed = False
            for c in comps:
                if not _overlap(c, [t], frames):
                    c.append(t); placed = True; break
            if not placed:
                comps.append([t])
                for c in comps[:-1]:
                    if _overlap(c, [t], frames):
                        vetoes.append((t, list(c)))
                        break
        out += comps
    return out


def _num_split(group, num_of):
    by, free = {}, []
    for t in group:
        n = num_of.get(t)
        (by.setdefault(n, []) if n is not None else free).append(t)
    if not by:
        return [group]
    parts = list(by.values())
    parts[0] += free
    return parts


_P_PAIR = (
    "Two multi-view grids of tracked people from a night 5-a-side match, each "
    "captioned with its track_id. Are these the SAME individual person (same hair, "
    "build, socks, shoes, skin, and any jersey number)? Dark/plain kits are normal. "
    'Return STRICT JSON {"same": true|false, "confidence": 0.0-1.0, "reason": "<one line>"}.'
)


def _vlm_pair(llm, grid_a, grid_b):
    from .llm import _parse_json
    try:
        d = _parse_json(llm._vision(_P_PAIR, [grid_a, grid_b], want_json=True))
        return bool(d.get("same")), float(d.get("confidence", 0)), str(d.get("reason", ""))[:100]
    except Exception as e:
        return False, 0.0, repr(e)[:60]


def _vlm_roster(llm, ids, grids, num_of, target, retries: int = 3):
    import time
    hints = ", ".join(f"{t}#{num_of[t]}" if t in num_of else str(t) for t in ids)
    prompt = _P_ROSTER.format(target=target, hints=hints)
    have = [t for t in ids if t in grids]
    imgs = [grids[t] for t in have]
    best = None
    for attempt in range(retries):
        try:
            txt = llm._vision(prompt, imgs, want_json=True)
            groups, reasoning = _parse_groups(txt, have)
            # a successful grouping merges something; all-singletons == failed/empty parse
            if best is None or len(groups) < len(best[0]):
                best = (groups, reasoning)
            if len(groups) < len(have):
                return groups, reasoning
        except Exception as e:
            print(f"   [roster] call failed (attempt {attempt+1}): {repr(e)[:80]}")
        time.sleep(2 * (attempt + 1))
    return best if best else ([[t] for t in have], "")


def refine_pairwise(groups, grids, embs, num_of, frames, llm, target=8,
                    max_calls=8, min_conf=0.6):
    """Merge the most-similar group pairs via VLM same/different, guarded by overlap."""
    merges, vetoes, calls = [], [], 0

    def cent(g):
        v = np.mean([embs[t] for t in g], axis=0); return v / (np.linalg.norm(v) + 1e-9)

    def rep(g):  # representative track = longest-lived
        return max(g, key=lambda t: len(frames[t]))

    tried = set()
    while len(groups) > target and calls < max_calls:
        # candidate pairs by embedding closeness (same-number pairs float to top)
        cand = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _overlap(groups[i], groups[j], frames):
                    continue
                ni = {num_of[t] for t in groups[i] if t in num_of}
                nj = {num_of[t] for t in groups[j] if t in num_of}
                if ni and nj and ni.isdisjoint(nj):
                    continue                                    # different confident numbers
                key = frozenset((rep(groups[i]), rep(groups[j])))
                if key in tried:
                    continue
                d = 1 - float(cent(groups[i]) @ cent(groups[j]))
                cand.append((0.0 if (ni & nj) else d, i, j, key))
        if not cand:
            break
        cand.sort(key=lambda c: c[0])
        _, i, j, key = cand[0]
        tried.add(key)
        same, conf, reason = _vlm_pair(llm, grids[rep(groups[i])], grids[rep(groups[j])])
        calls += 1
        if same and conf >= min_conf and not _overlap(groups[i], groups[j], frames):
            merges.append((list(groups[i]), list(groups[j]), round(conf, 2), reason))
            groups[i] += groups[j]; groups.pop(j)
            tried.clear()                                       # indices shifted; re-evaluate
        # else: try the next-closest untried pair rather than stopping
    return groups, merges, vetoes, calls


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
TEAM_TARGET = {"A": 8, "B": 8, "O": 6}


def run(video1080, reid_tracks, cache, teams, jersey, participant_csv, out, montage_dir,
        llm, lo=0, hi=1800, scale=1.5):
    from .roster import load_context, _emit, _name_clusters

    ctx = load_context(Path(cache), Path(teams), Path(jersey),
                       Path("data/tracks/identities_vlm_night_vlmfp.parquet"), 15)
    pc = pd.read_csv(participant_csv)
    plabel = dict(zip(pc["track_id"].astype(int), pc["label"]))
    gk_ids = [t for t in ctx["keep"] if plabel.get(t) == "GK"]
    players = [t for t in ctx["keep"] if plabel.get(t) in ("PLAYER", "GK")]
    print(f"participants: {len(players)} (of {len(ctx['keep'])} kept); goalkeepers: {gk_ids}")

    df = pd.read_parquet(reid_tracks)
    df = df[(df["frame"] >= lo) & (df["frame"] <= hi) & (df["track_id"].isin(players))]
    grids, chosen = build_grids(Path(video1080), df, lo, hi, scale)
    print(f"built {len(grids)} multi-view grids")

    all_groups, all_veto, merge_log, failures = [], [], {}, []
    for team in ("A", "B", "O"):
        ids = [t for t in players if ctx["team_of"].get(t, "O") == team and t not in gk_ids and t in grids]
        if not ids:
            continue
        target = TEAM_TARGET[team]
        # T2 in CHUNKS of <=8 grids so every grouping call is small enough to parse.
        chunks = [ids[k:k + 8] for k in range(0, len(ids), 8)]
        groups = []
        for ci, chunk in enumerate(chunks):
            g, reasoning = _vlm_roster(llm, chunk, grids, ctx["num_of"], min(target, len(chunk)))
            if len(g) == len(chunk) and len(chunk) > 3:
                failures.append(f"team {team} chunk {ci}: grouping returned all-singletons")
                print(f"   !! team {team} chunk {ci}: all-singletons (grouping likely failed)")
            groups += g
        groups = _enforce_overlap(groups, ctx["frames"], all_veto)
        groups = [p for g in groups for p in _num_split(g, ctx["num_of"])]
        n_after_t2 = len(groups)
        # T3 + sanity floor: pairwise-merge across chunks toward the target.
        groups, merges, vetoes, calls = refine_pairwise(
            groups, grids, ctx["embs"], ctx["num_of"], ctx["frames"], llm,
            target=target, max_calls=14)
        all_veto += vetoes
        merge_log[team] = {"chunks": len(chunks), "t2_groups": n_after_t2,
                           "t3_merges": len(merges), "t3_calls": calls, "final": len(groups)}
        print(f"[team {team}] {len(ids)} tracks, {len(chunks)} chunks -> T2 {n_after_t2} -> T3 {len(groups)} (calls {calls})")
        all_groups += groups

    by_track = _name_clusters(all_groups, ctx)
    for g in ([gk] for gk in gk_ids):                    # GK: own identity
        team = ctx["team_of"].get(g[0], "O")
        by_track[g[0]] = f"{team}-GK"
    _emit(by_track, ctx, Path(out))

    # marks + roster montages
    _write_marks(by_track, gk_ids, ctx, reid_tracks, lo, hi, Path("data/tracks/marks_roster_night.parquet"))
    _roster_montage(all_groups, gk_ids, by_track, chosen, ctx, montage_dir)
    print("MERGE LOG:", json.dumps(merge_log, default=str)[:1500])
    print("OVERLAP VETOES:", [(v[0], list(v[1])) for v in all_veto][:20])
    print("GROUPING FAILURES:", failures if failures else "none")
    return by_track, all_groups, gk_ids, merge_log, all_veto, failures


def _write_marks(by_track, gk_ids, ctx, reid_tracks, lo, hi, mk):
    tm = pd.read_parquet("data/tracks/teams_reid_night.parquet")
    team_rgb = {}
    for _, r in tm.iterrows():
        team_rgb.setdefault(r.team, (int(r.color_r), int(r.color_g), int(r.color_b)))
    pc = pd.read_csv("data/out/participant_classification.csv")
    plabel = dict(zip(pc["track_id"].astype(int), pc["label"]))
    allt = sorted(pd.read_parquet(reid_tracks).query("@lo<=frame<=@hi").track_id.unique())
    rows = []
    for t in allt:
        t = int(t); sid = by_track.get(t); lab = plabel.get(t, "PLAYER")
        if t in gk_ids:
            color, label = (0, 255, 255), (sid or f"GK{t}")
        elif lab == "SPECTATOR":
            color, label = (150, 150, 150), "SPECTATOR"
        elif sid:
            color, label = team_rgb.get(ctx["team_of"].get(t, "O"), (170, 170, 170)), sid
        else:
            color, label = (150, 150, 150), (lab if lab != "PLAYER" else f"?{t}")
        rows.append({"track_id": t, "label": label, "color_r": color[0], "color_g": color[1], "color_b": color[2]})
    pd.DataFrame(rows).to_parquet(mk, index=False)
    print(f"wrote marks {mk}")


def _roster_montage(all_groups, gk_ids, by_track, chosen, ctx, out_dir):
    out_dir = Path(out_dir)
    groups = list(all_groups) + [[g] for g in gk_ids]
    by_team = {}
    for members in groups:
        team = ctx["team_of"].get(members[0], "O")
        by_team.setdefault(team, []).append(members)
    for team, glist in by_team.items():
        rows = []
        for members in sorted(glist, key=lambda m: by_track.get(m[0], "")):
            sid = by_track.get(members[0], "?")
            lab = np.full((120, 150, 3), 25, np.uint8)
            cv2.putText(lab, sid, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(lab, f"{len(members)}trk", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cells = [lab]
            for t in members:
                for c in chosen.get(t, [])[:3]:
                    im = cv2.resize(c, (80, 120))
                    cv2.putText(im, str(t), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cells.append(im)
            rows.append(np.hstack(cells[:16]) if len(cells) > 1 else lab)
        if not rows:
            continue
        W = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 20, np.uint8)]) if r.shape[1] < W else r for r in rows]
        cv2.imwrite(str(out_dir / f"roster_montage_{team}.jpg"),
                    np.vstack([np.vstack([r, np.full((3, W, 3), 80, np.uint8)]) for r in rows]))
        print(f"[roster] montage {team}: {len(rows)} players")


# --------------------------------------------------------------------------- #
# Holistic roster-grid dedup — review the WHOLE per-team grid and merge duplicates
# --------------------------------------------------------------------------- #
_P_DEDUP = (
    "Each labeled ROW below is a candidate player from ONE team in a night 5-a-side "
    "match, shown with example crops (super-resolved) and any known jersey number. "
    "Some rows are the SAME physical player duplicated across rows. A real team has "
    "~7-8 outfield players. Tell me which ROW-LABELS to merge into one player.\n"
    "Hard evidence: jersey numbers/names — same number => same player; clearly "
    "different numbers => different players. Otherwise use hair, facial hair, skin, "
    "build, socks, shoes, kit shade. Dark/plain kits are normal.\n"
    "Return STRICT JSON {\"merge_groups\": [[label,label,...], ...], \"reasoning\": \"<short>\"}. "
    "Only include labels that should merge; omit singletons. Merge aggressively toward "
    "~7-8 players when the crops clearly show the same person."
)


def _roster_grid_image(id_to_tracks, chosen, num_of, sr, per_row=5, rowh=150):
    """One image: a labeled row per identity (label + number + SR sample crops)."""
    import cv2 as _cv2
    rows, labels = [], []
    for sid in sorted(id_to_tracks):
        tracks = id_to_tracks[sid]
        nums = sorted({num_of[t] for t in tracks if t in num_of})
        cap = sid + (f"  #{','.join(map(str, nums))}" if nums else "")
        lab = np.full((rowh, 190, 3), 25, np.uint8)
        _cv2.putText(lab, sid, (6, 40), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if nums:
            _cv2.putText(lab, f"#{','.join(map(str, nums))}", (6, 78),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cells = [lab]
        picks = [c for t in tracks for c in chosen.get(t, [])][:per_row]
        for c in picks:
            im = _cv2.resize(sr(c, 4), (int(rowh * 0.66), rowh))
            cells.append(im)
        row = np.hstack(cells)
        rows.append(row); labels.append(sid)
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((rowh, W - r.shape[1], 3), 20, np.uint8)]) if r.shape[1] < W else r for r in rows]
    img = np.vstack([np.vstack([r, np.full((3, W, 3), 90, np.uint8)]) for r in rows])
    return img, labels


def holistic_dedup(identities_parquet, ctx, chosen, llm, out, montage_dir,
                   max_iters=4, target=8):
    """Iteratively review each team's whole roster grid and merge duplicate rows.

    Mutates one shared groups dict {label -> [track_ids]}; GK identities (label
    ends '-GK') are never merged. Emits the deduped identities parquet + table.
    """
    from .sr import upscale
    from .llm import _parse_json
    from .roster import _majority

    idf = pd.read_parquet(identities_parquet)
    groups: dict[str, list[int]] = {}
    for t, s in zip(idf["raw_track_id"].astype(int), idf["stable_id"]):
        groups.setdefault(s, []).append(int(t))

    log, vetoes = {}, []
    for team in ("A", "B", "O"):
        team_labels = lambda: [s for s, ts in groups.items()
                               if ctx["team_of"].get(ts[0], "O") == team and not s.endswith("-GK")]
        history = [len(team_labels())]
        for it in range(max_iters):
            labs_now = team_labels()
            if len(labs_now) <= target:
                break
            sub = {s: groups[s] for s in labs_now}
            img, labels = _roster_grid_image(sub, chosen, ctx["num_of"], upscale)
            cv2.imwrite(str(Path(montage_dir) / f"dedup_grid_{team}_iter{it}.jpg"), img)
            try:
                d = _parse_json(llm._vision(_P_DEDUP + f"\nRow labels present: {labels}.",
                                            [img], want_json=True))
                merge_groups = d.get("merge_groups", []) if isinstance(d, dict) else []
                reasoning = (d.get("reasoning", "") if isinstance(d, dict) else "")[:280]
            except Exception as e:
                merge_groups, reasoning = [], f"ERR {e!r}"[:120]
            applied = 0
            for grp in merge_groups:
                labs = [l for l in dict.fromkeys(grp) if l in groups]
                if len(labs) < 2:
                    continue
                members = [t for l in labs for t in groups[l]]
                overlap = any(ctx["frames"][members[x]] & ctx["frames"][members[y]]
                              for x in range(len(members)) for y in range(x + 1, len(members)))
                distinct_nums = {ctx["num_of"][t] for t in members if t in ctx["num_of"]}
                if overlap:
                    vetoes.append((team, labs, "time-overlap"))
                    continue
                if len(distinct_nums) > 1:                      # different confident numbers
                    vetoes.append((team, labs, f"number-conflict {sorted(distinct_nums)}"))
                    continue
                keep = _pick_label(labs, groups, ctx["num_of"])
                for l in labs:
                    if l != keep:
                        groups[keep] += groups[l]; del groups[l]
                applied += 1
            history.append(len(team_labels()))
            print(f"[dedup {team}] iter {it}: {history[-2]} -> {history[-1]} rows (merged {applied}); {reasoning}")
            if applied == 0:
                break
        log[team] = history

    # Emit deduped identities (+ table). Re-canonicalise names: numbered kept, else team-index.
    rows, id_table = [], []
    counter = {}
    for sid, tracks in groups.items():
        team = _majority([ctx["team_of"].get(t, "O") for t in tracks])
        nums = sorted({ctx["num_of"][t] for t in tracks if t in ctx["num_of"]})
        if sid.endswith("-GK"):
            label = sid
        elif nums:
            label = f"{team}#{nums[0]}"
        else:
            counter[team] = counter.get(team, 0) + 1
            label = f"{team}-{counter[team]}"
        n_frames = len(set().union(*[ctx["frames"][t] for t in tracks]))
        id_table.append({"stable_id": label, "team": team, "number": nums[0] if nums else -1,
                         "n_raw_tracks": len(tracks), "n_frames": n_frames,
                         "raw_track_ids": ",".join(map(str, sorted(tracks)))})
        for t in tracks:
            rows.append({"raw_track_id": t, "stable_id": label, "team": team,
                         "number": nums[0] if nums else -1})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("raw_track_id").to_parquet(out, index=False)
    pd.DataFrame(id_table).sort_values(["team", "stable_id"]).to_parquet(
        Path(out).parent / f"{Path(out).stem}_table.parquet", index=False)
    tot = len(id_table)
    per = pd.Series([r["team"] for r in id_table]).value_counts().to_dict()
    print(f"[dedup] final {tot} identities, per-team {per}")
    return log, vetoes, {s: groups[s] for s in groups}


def _pick_label(labs, groups, num_of):
    numbered = [l for l in labs if any(t in num_of for t in groups[l])]
    pool = numbered or labs
    return max(pool, key=lambda l: len(groups[l]))


def main():
    import argparse, yaml
    from .llm import get_llm
    ap = argparse.ArgumentParser()
    ap.add_argument("--video1080", default="data/raw/hhaWMwLlugE_1080.mp4")
    ap.add_argument("--reid", default="data/tracks/reid_night_0_1800.parquet")
    ap.add_argument("--cache", default="data/tracks/identities_night_desc.npz")
    ap.add_argument("--teams", default="data/tracks/teams_reid_night.parquet")
    ap.add_argument("--jersey", default="data/tracks/jersey_reid_night.parquet")
    ap.add_argument("--participant-csv", default="data/out/participant_classification.csv")
    ap.add_argument("--out", default="data/tracks/identities_roster_night.parquet")
    ap.add_argument("--montage-dir", default="data/out")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    args = ap.parse_args()
    cfg = {"llm": {"enabled": True, "provider": "gemini", "model": args.model, "uses": {"fingerprint": True}}}
    llm = get_llm(cfg)
    print(f"[roster_v2] model = {llm.model}")
    run(args.video1080, args.reid, args.cache, args.teams, args.jersey, args.participant_csv,
        args.out, args.montage_dir, llm)


if __name__ == "__main__":
    main()
