"""M4 Layers 2+4 — collapse raw tracklets into stable player identities.

The re-ID tracker (fv.reid_track) still leaves ~5-6x more tracklets than there
are players: whenever a player is off-screen long enough, even BoT-SORT starts a
new track. This stage fuses those tracklets **offline** into a small set of
stable identities using, in priority order:

  1. PRIMARY — a part-based APPEARANCE FINGERPRINT per track: the crop is split
     into horizontal bands (hair, skin, shirt, shorts, socks, shoes) and a robust
     dominant colour (median Lab) is taken per band, plus build (bbox aspect).
     Fused with the deep OSNet ReID embedding into one descriptor.
  2. TEAM (fv.teams) — a HARD gate: tracklets on different teams never merge.
  3. TEMPORAL — tracklets that coexist in any frame are different people (hard
     cannot-link).
  4. JERSEY NUMBER (fv.jersey) — a weak signal: confident matching numbers pull
     two tracklets together; confident *different* numbers block a merge.

Fusion is constrained agglomerative clustering (average linkage) with the hard
cannot-links above. Output: data/tracks/identities.parquet mapping every raw
track_id -> a stable_id like "A#7" (number known) or "A-3" (appearance cluster).

Two phases with a cache in between (extract is the expensive GPU/decode pass;
fuse is cheap so thresholds can be retuned without re-decoding).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Horizontal bands as (name, y_lo, y_hi, side_inset) fractions of the player box.
BANDS = [
    ("hair",   0.00, 0.12, 0.28),
    ("shirt",  0.16, 0.40, 0.16),
    ("shorts", 0.45, 0.63, 0.16),
    ("socks",  0.70, 0.90, 0.20),
    ("shoes",  0.90, 1.00, 0.20),
]
# Skin sampled from face + lower legs and merged into one "skin" descriptor.
SKIN_BANDS = [("face", 0.02, 0.12, 0.30), ("legs", 0.62, 0.71, 0.22)]
BAND_NAMES = [b[0] for b in BANDS] + ["skin"]

# Per-band weight in the fingerprint distance. Shirt is downweighted (it is
# essentially the team, already a hard gate); socks/shorts/shoes/hair are what
# separate team-mates, so they carry the most weight.
BAND_W = {"hair": 1.0, "shirt": 0.3, "shorts": 1.1, "socks": 1.2, "shoes": 0.8, "skin": 0.6}

GREEN_H = (30, 95)      # pitch green (OpenCV HSV hue)
GREEN_MIN_S = 40
GREEN_MIN_V = 35
DARK_MAX_V = 20


def _band_lab(region: np.ndarray) -> np.ndarray | None:
    """Median Lab colour of a band region after masking pitch-green + shadow."""
    if region.size == 0 or region.shape[0] < 2 or region.shape[1] < 2:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (H >= GREEN_H[0]) & (H <= GREEN_H[1]) & (S >= GREEN_MIN_S) & (V >= GREEN_MIN_V)
    dark = V < DARK_MAX_V
    keep = ~(green | dark)
    if keep.sum() < 8:
        keep = np.ones_like(keep)  # fall back to whole band
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(float)
    return np.median(lab[keep], axis=0)  # (L, a, b)


def _fingerprint(crop: np.ndarray) -> dict[str, np.ndarray]:
    """Per-band median Lab for one player crop (missing bands omitted)."""
    h, w = crop.shape[:2]
    fp: dict[str, np.ndarray] = {}
    for name, y0, y1, inset in BANDS:
        r = crop[int(h * y0):int(h * y1), int(w * inset):int(w * (1 - inset))]
        lab = _band_lab(r)
        if lab is not None:
            fp[name] = lab
    skin = []
    for _, y0, y1, inset in SKIN_BANDS:
        r = crop[int(h * y0):int(h * y1), int(w * inset):int(w * (1 - inset))]
        lab = _band_lab(r)
        if lab is not None:
            skin.append(lab)
    if skin:
        fp["skin"] = np.median(np.stack(skin), axis=0)
    return fp


def _sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[np.unique(idx)]


def extract(video: Path, tracks: Path, cache_out: Path,
            start_frame: int = 0, end_frame: int | None = None,
            samples_per_track: int = 40, device: str = "cuda:0",
            reid_weights: str = "osnet_x1_0_msmt17.pt") -> Path:
    """Decode once; per track store OSNet embedding, per-band Lab, aspect, frames."""
    from boxmot.reid.core.reid_handler import ReID  # boxmot 16.0.x (pinned)

    df = pd.read_parquet(tracks)
    lo = int(start_frame)
    hi = int(df["frame"].max()) if end_frame is None else int(end_frame)
    df = df[(df["frame"] >= lo) & (df["frame"] <= hi)]

    # Which boxes to sample at each frame.
    needed: dict[int, list[tuple[int, tuple]]] = {}
    frames_of: dict[int, set] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        frames_of[int(tid)] = set(int(f) for f in g["frame"])
        for f in _sample_frames(g["frame"].to_numpy(), samples_per_track):
            row = g[g["frame"] == f].iloc[0]
            needed.setdefault(int(f), []).append(
                (int(tid), (float(row.x1), float(row.y1), float(row.x2), float(row.y2))))

    reid = ReID(weights=reid_weights, device=device, half=False)
    emb_of: dict[int, list[np.ndarray]] = {}
    fp_of: dict[int, dict[str, list[np.ndarray]]] = {}
    aspect_of: dict[int, list[float]] = {}
    best_crop: dict[int, tuple[float, np.ndarray]] = {}  # track -> (area, crop)

    cap = cv2.VideoCapture(str(video))
    idx = -1
    while idx < hi:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        items = needed.get(idx)
        if items is None:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        Hf, Wf = frame.shape[:2]
        boxes, tids = [], []
        for tid, (x1, y1, x2, y2) in items:
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(Wf, int(x2)), min(Hf, int(y2))
            if x2i - x1i < 6 or y2i - y1i < 12:
                continue
            crop = frame[y1i:y2i, x1i:x2i]
            fp = _fingerprint(crop)
            if fp:
                d = fp_of.setdefault(tid, {})
                for k, v in fp.items():
                    d.setdefault(k, []).append(v)
            aspect_of.setdefault(tid, []).append((y2i - y1i) / max(1, (x2i - x1i)))
            area = (x2i - x1i) * (y2i - y1i)
            if tid not in best_crop or area > best_crop[tid][0]:
                best_crop[tid] = (area, crop.copy())
            boxes.append([x1i, y1i, x2i, y2i]); tids.append(tid)
        if boxes:
            feats = reid.model.get_features(np.array(boxes, dtype=float), frame)
            feats = np.asarray(feats, dtype=float)
            for tid, f in zip(tids, feats):
                n = np.linalg.norm(f) + 1e-9
                emb_of.setdefault(tid, []).append(f / n)
    cap.release()

    # Aggregate per track.
    track_ids = sorted(emb_of.keys())
    embs = np.stack([_l2(np.median(np.stack(emb_of[t]), axis=0)) for t in track_ids])
    bands = np.full((len(track_ids), len(BAND_NAMES), 3), np.nan)
    aspect = np.array([float(np.median(aspect_of.get(t, [2.5]))) for t in track_ids])
    for i, t in enumerate(track_ids):
        for j, name in enumerate(BAND_NAMES):
            vals = fp_of.get(t, {}).get(name)
            if vals:
                bands[i, j] = np.median(np.stack(vals), axis=0)

    cache_out.parent.mkdir(parents=True, exist_ok=True)
    # Save representative crops alongside for the identity table.
    crop_dir = cache_out.parent / f"{cache_out.stem}_crops"
    crop_dir.mkdir(exist_ok=True)
    for t in track_ids:
        cv2.imwrite(str(crop_dir / f"t{t}.jpg"), best_crop[t][1])
    np.savez(cache_out, track_ids=np.array(track_ids), embs=embs, bands=bands,
             band_names=np.array(BAND_NAMES), aspect=aspect,
             frames=np.array([sorted(frames_of[t]) for t in track_ids], dtype=object))
    print(f"[extract] {len(track_ids)} tracks -> {cache_out} (crops in {crop_dir})")
    return cache_out


def _l2(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _fp_dist(bi: np.ndarray, bj: np.ndarray) -> float:
    """Weighted mean band deltaE (Lab euclidean) over bands present in both."""
    num = den = 0.0
    for k, name in enumerate(BAND_NAMES):
        if np.isnan(bi[k]).any() or np.isnan(bj[k]).any():
            continue
        de = float(np.linalg.norm(bi[k] - bj[k]))
        w = BAND_W[name]
        num += w * min(de / 35.0, 1.0)
        den += w
    return num / den if den > 0 else 0.5


def fuse(cache: Path, teams: Path, out: Path, jersey: Path | None = None,
         w_emb: float = 0.5, w_fp: float = 0.45, w_asp: float = 0.05,
         threshold: float = 0.42) -> Path:
    z = np.load(cache, allow_pickle=True)
    track_ids = list(z["track_ids"])
    embs, bands, aspect = z["embs"], z["bands"], z["aspect"]
    frames = [set(f) for f in z["frames"]]
    n = len(track_ids)

    tdf = pd.read_parquet(teams)
    team_of = dict(zip(tdf["track_id"].astype(int), tdf["team"]))
    teams_arr = [team_of.get(int(t), "O") for t in track_ids]

    num_of = {}
    if jersey is not None and Path(jersey).exists():
        jdf = pd.read_parquet(jersey)
        for _, r in jdf.iterrows():
            if bool(r.get("confident", False)) and int(r["number"]) >= 0:
                num_of[int(r["track_id"])] = int(r["number"])

    INF = 1e9
    D = np.full((n, n), INF)
    for i in range(n):
        for j in range(i + 1, n):
            if teams_arr[i] != teams_arr[j]:
                continue                              # team hard gate
            if frames[i] & frames[j]:
                continue                              # coexist -> different players
            ni, nj = num_of.get(track_ids[i]), num_of.get(track_ids[j])
            if ni is not None and nj is not None and ni != nj:
                continue                              # confident different numbers
            d_emb = 1.0 - float(embs[i] @ embs[j])
            d_fp = _fp_dist(bands[i], bands[j])
            d_asp = min(abs(aspect[i] - aspect[j]) / 1.0, 1.0)
            d = w_emb * d_emb + w_fp * d_fp + w_asp * d_asp
            if ni is not None and nj is not None and ni == nj:
                d *= 0.3                              # same confident number -> pull together
            D[i, j] = D[j, i] = d

    clusters = _agglomerate(D, threshold, INF)

    # Assign stable ids. Number-majority within a cluster names it "A#7", else "A-k".
    rows = []
    id_table = []
    per_team_counter: dict[str, int] = {}
    for members in clusters:
        team = _majority([teams_arr[m] for m in members])
        nums = [num_of[track_ids[m]] for m in members if track_ids[m] in num_of]
        if nums:
            number = _majority(nums)
            sid = f"{team}#{number}"
        else:
            per_team_counter[team] = per_team_counter.get(team, 0) + 1
            number = -1
            sid = f"{team}-{per_team_counter[team]}"
        n_frames = len(set().union(*[frames[m] for m in members]))
        id_table.append({"stable_id": sid, "team": team, "number": number,
                         "n_raw_tracks": len(members), "n_frames": n_frames,
                         "raw_track_ids": ",".join(str(track_ids[m]) for m in members)})
        for m in members:
            rows.append({"raw_track_id": int(track_ids[m]), "stable_id": sid,
                         "team": team, "number": number})

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("raw_track_id").to_parquet(out, index=False)
    tbl = out.parent / f"{out.stem}_table.parquet"
    pd.DataFrame(id_table).sort_values(["team", "stable_id"]).to_parquet(tbl, index=False)

    print(f"[fuse] {n} raw tracks -> {len(clusters)} stable identities (threshold {threshold})")
    per_team = pd.Series([r["team"] for r in id_table]).value_counts().to_dict()
    print(f"  identities per team: {per_team}")
    print(f"  with jersey number: {sum(1 for r in id_table if r['number'] >= 0)}")
    print(f"wrote {out}  and  {tbl}")
    return out


def _agglomerate(D: np.ndarray, threshold: float, INF: float = 1e9) -> list[list[int]]:
    """Constrained average-linkage agglomerative clustering.

    Merge the closest pair of clusters while their linkage is below `threshold`
    and NO cross-member pair is a cannot-link (distance >= INF). Cannot-links
    (team/temporal/number conflicts) are baked into D by the caller.
    """
    n = D.shape[0]
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                pairs = [(i, j) for i in clusters[a] for j in clusters[b]]
                if any(D[i, j] >= INF for i, j in pairs):
                    continue
                avg = float(np.mean([D[i, j] for i, j in pairs]))
                if best is None or avg < best[0]:
                    best = (avg, a, b)
        if best is None or best[0] > threshold:
            break
        _, a, b = best
        clusters[a] += clusters[b]
        clusters.pop(b)
    return clusters


def _majority(xs):
    return pd.Series(xs).value_counts().index[0]


# --------------------------------------------------------------------------- #
# Optional VLM appearance-fingerprint fusion (gated by config: llm.uses.fingerprint)
# --------------------------------------------------------------------------- #
# Attributes the VLM returns (fv.llm.describe_player). Weights favour the ones
# that separate same-kit team-mates; shirt ~= team so it is nearly ignored.
VLM_ATTR_W = {"hair": 1.2, "socks": 1.2, "shoes": 1.0, "build": 0.8, "shorts": 1.0,
              "skin": 0.5, "facial_hair": 0.6, "shirt": 0.2}
_STOP = {"a", "an", "the", "with", "and", "or", "of", "in", "on", "medium", "short",
         "long", "number", "no", "none", "unknown", "colour", "color", "coloured"}


def _sharpness(crop: np.ndarray) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _best_crops_per_track(video: Path, df: pd.DataFrame, lo: int, hi: int,
                          per_track: int = 3, samples: int = 24,
                          min_h: int = 55, min_w: int = 22) -> dict[int, list[np.ndarray]]:
    """Decode once; per track return its 2-3 clearest crops (large + sharp)."""
    needed: dict[int, list[tuple]] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        for f in _sample_frames(g["frame"].to_numpy(), samples):
            r = g[g["frame"] == f].iloc[0]
            needed.setdefault(int(f), []).append(
                (int(tid), (float(r.x1), float(r.y1), float(r.x2), float(r.y2))))
    cand: dict[int, list[tuple]] = {}
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
            if y2i - y1i < min_h or x2i - x1i < min_w:
                continue
            crop = frame[y1i:y2i, x1i:x2i]
            cand.setdefault(tid, []).append(
                ((x2i - x1i) * (y2i - y1i), _sharpness(crop), crop.copy()))
    cap.release()
    best: dict[int, list[np.ndarray]] = {}
    for tid, lst in cand.items():
        # Prefer the largest crops, then the sharpest among them.
        lst.sort(key=lambda t: t[0], reverse=True)
        top_area = lst[: max(per_track * 2, per_track)]
        top_area.sort(key=lambda t: t[1], reverse=True)
        best[tid] = [c for _, _, c in top_area[:per_track]]
    return best


def vlm_fingerprints(video: Path, tracks: Path, out: Path, llm,
                     start_frame: int = 0, end_frame: int | None = None,
                     per_track: int = 3, cache: bool = True) -> dict[int, dict]:
    """One describe_player call per track (cached by raw_track_id)."""
    df = pd.read_parquet(tracks)
    lo = int(start_frame)
    hi = int(df["frame"].max()) if end_frame is None else int(end_frame)
    df = df[(df["frame"] >= lo) & (df["frame"] <= hi)]

    # Cache one flat row per track (one string column per attribute) — avoids any
    # JSON-blob round-trip ambiguity in parquet. Only cache non-empty fingerprints
    # so a track that yielded no usable crop is retried on a later run.
    attr_keys = [k for k in VLM_ATTR_W]  # hair, socks, shoes, build, shorts, skin, facial_hair, shirt
    have: dict[int, dict] = {}
    if cache and out.exists():
        old = pd.read_parquet(out)
        for _, r in old.iterrows():
            fp = {k: r[k] for k in attr_keys if k in old.columns and isinstance(r[k], str) and r[k]}
            if fp:
                have[int(r["track_id"])] = fp
        print(f"[vlm] loaded {len(have)} cached fingerprints from {out}")

    todo = [int(t) for t in df["track_id"].unique() if int(t) not in have]
    if todo:
        crops = _best_crops_per_track(video, df[df["track_id"].isin(todo)], lo, hi, per_track)
        got = 0
        for i, tid in enumerate(todo, 1):
            cs = crops.get(tid, [])
            if not cs:
                continue
            try:
                fp = llm.describe_player(cs)
                if isinstance(fp, dict) and fp:
                    have[tid] = {k: str(v) for k, v in fp.items()}
                    got += 1
            except Exception as e:  # keep going; one bad call shouldn't sink the run
                print(f"  [vlm] track {tid} describe failed: {e!r}")
            if i % 10 == 0:
                print(f"  [vlm] described {i}/{len(todo)} tracks ({got} non-empty)")
        rows = [{"track_id": t, **{k: a.get(k, "") for k in attr_keys}} for t, a in have.items()]
        if rows:
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("track_id").to_parquet(out, index=False)
            print(f"[vlm] wrote {len(rows)} fingerprints ({got} freshly described) -> {out}")
    return have


def _norm_tokens(val) -> set:
    if not isinstance(val, str):
        return set()
    toks = re.findall(r"[a-z]+", val.lower())
    return {t for t in toks if t not in _STOP and len(t) > 1}


def _attr_dist(a: dict, b: dict) -> float:
    """Weighted attribute distance in [0,1] over shared, known attributes."""
    num = den = 0.0
    for k, w in VLM_ATTR_W.items():
        ta, tb = _norm_tokens(a.get(k)), _norm_tokens(b.get(k))
        if not ta or not tb:
            continue                         # unknown/missing -> skip (neutral)
        inter = len(ta & tb)
        union = len(ta | tb)
        sim = inter / union if union else 0.0
        num += w * (1.0 - sim)
        den += w
    return num / den if den > 0 else 0.5


def _vlm_same_player(llm, crop_a, crop_b) -> bool | None:
    """Ask the VLM whether two crops are the same player (few, ambiguous pairs)."""
    prompt = ("These are two cropped images of soccer players from the same match. "
              "Are they the SAME individual person (same kit, hair, build)? "
              'Return ONLY JSON: {"same": true|false}.')
    try:
        import re as _re
        txt = llm._vision(prompt, [crop_a, crop_b], want_json=True)
        m = _re.search(r'"same"\s*:\s*(true|false)', txt.lower())
        return (m.group(1) == "true") if m else None
    except Exception:
        return None


def fuse_vlm(cache: Path, teams: Path, vlm_fp: dict[int, dict], out: Path,
             jersey: Path | None = None, llm=None, judge: bool = False,
             w_vlm: float = 0.4, w_emb: float = 0.6, threshold: float = 0.50,
             judge_band: float = 0.10, judge_cap: int = 15,
             crops_for_judge: dict[int, list] | None = None) -> Path:
    # The OSNet embedding is the more RELIABLE same-tracklet signal on low-res
    # night crops; the VLM attributes are richer but noisier across a track's
    # crops. So embedding leads (w_emb) and the semantic fingerprint refines it
    # (w_vlm) — VLM-dominant weights fragment/mis-link players here.
    z = np.load(cache, allow_pickle=True)
    track_ids = [int(t) for t in z["track_ids"]]
    embs = z["embs"]
    frames = [set(int(x) for x in f) for f in z["frames"]]
    n = len(track_ids)

    tdf = pd.read_parquet(teams)
    team_of = dict(zip(tdf["track_id"].astype(int), tdf["team"]))
    teams_arr = [team_of.get(t, "O") for t in track_ids]

    num_of = {}
    if jersey is not None and Path(jersey).exists():
        jdf = pd.read_parquet(jersey)
        for _, r in jdf.iterrows():
            if bool(r.get("confident", False)) and int(r["number"]) >= 0:
                num_of[int(r["track_id"])] = int(r["number"])

    INF = 1e9
    D = np.full((n, n), INF)
    ambiguous = []
    for i in range(n):
        for j in range(i + 1, n):
            if teams_arr[i] != teams_arr[j]:
                continue                                  # team hard gate
            if frames[i] & frames[j]:
                continue                                  # coexist -> different
            ni, nj = num_of.get(track_ids[i]), num_of.get(track_ids[j])
            if ni is not None and nj is not None and ni != nj:
                continue                                  # different confident numbers
            d_vlm = _attr_dist(vlm_fp.get(track_ids[i], {}), vlm_fp.get(track_ids[j], {}))
            d_emb = 1.0 - float(embs[i] @ embs[j])
            d = w_vlm * d_vlm + w_emb * d_emb
            if ni is not None and nj is not None and ni == nj:
                d *= 0.3                                  # same confident number -> pull
            D[i, j] = D[j, i] = d
            if abs(d - threshold) <= judge_band:
                ambiguous.append((d, i, j))

    # Optional VLM judge on the few borderline pairs.
    if judge and llm is not None and crops_for_judge:
        ambiguous.sort(key=lambda t: abs(t[0] - threshold))
        used = 0
        for _, i, j in ambiguous:
            if used >= judge_cap:
                break
            ca, cb = crops_for_judge.get(track_ids[i]), crops_for_judge.get(track_ids[j])
            if not ca or not cb:
                continue
            verdict = _vlm_same_player(llm, ca[0], cb[0])
            used += 1
            if verdict is True:
                D[i, j] = D[j, i] = min(D[i, j], threshold * 0.5)
            elif verdict is False:
                D[i, j] = D[j, i] = INF
        print(f"[vlm] judged {used} borderline pairs")

    clusters = _agglomerate(D, threshold, INF)

    rows, id_table = [], []
    per_team_counter: dict[str, int] = {}
    for members in clusters:
        team = _majority([teams_arr[m] for m in members])
        nums = [num_of[track_ids[m]] for m in members if track_ids[m] in num_of]
        if nums:
            number = _majority(nums)
            sid = f"{team}#{number}"
        else:
            per_team_counter[team] = per_team_counter.get(team, 0) + 1
            number = -1
            sid = f"{team}-{per_team_counter[team]}"
        # Representative attributes: from the member with the most descriptive fingerprint.
        rep = max(members, key=lambda m: len(json.dumps(vlm_fp.get(track_ids[m], {}))))
        attrs = vlm_fp.get(track_ids[rep], {})
        n_frames = len(set().union(*[frames[m] for m in members]))
        id_table.append({"stable_id": sid, "team": team, "number": number,
                         "n_raw_tracks": len(members), "n_frames": n_frames,
                         "hair": attrs.get("hair", ""), "shirt": attrs.get("shirt", ""),
                         "shorts": attrs.get("shorts", ""), "socks": attrs.get("socks", ""),
                         "shoes": attrs.get("shoes", ""), "build": attrs.get("build", ""),
                         "raw_track_ids": ",".join(str(track_ids[m]) for m in members)})
        for m in members:
            rows.append({"raw_track_id": track_ids[m], "stable_id": sid,
                         "team": team, "number": number})

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("raw_track_id").to_parquet(out, index=False)
    tbl = out.parent / f"{out.stem}_table.parquet"
    pd.DataFrame(id_table).sort_values(["team", "stable_id"]).to_parquet(tbl, index=False)
    print(f"[fuse-vlm] {n} raw tracks -> {len(clusters)} stable identities (threshold {threshold})")
    print(f"  identities per team: {pd.Series([r['team'] for r in id_table]).value_counts().to_dict()}")
    print(f"  with jersey number: {sum(1 for r in id_table if r['number'] >= 0)}")
    print(f"wrote {out}  and  {tbl}")
    return out


def run(video: Path, tracks: Path, out: Path, teams: Path, jersey: Path | None = None,
        start_frame: int = 0, end_frame: int | None = None,
        samples_per_track: int = 40, device: str = "cuda:0",
        threshold: float = 0.42, cache: Path | None = None,
        reid_weights: str = "osnet_x1_0_msmt17.pt") -> Path:
    cache = cache or out.parent / f"{out.stem}_desc.npz"
    extract(video, tracks, cache, start_frame, end_frame, samples_per_track, device, reid_weights)
    return fuse(cache, teams, out, jersey, threshold=threshold)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse tracklets into stable player identities.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path, help="reid tracklets parquet")
    ap.add_argument("--teams", type=Path, required=True)
    ap.add_argument("--jersey", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/tracks/identities.parquet"))
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--samples-per-track", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.42)
    ap.add_argument("--fuse-only", action="store_true", help="reuse cached descriptors")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/veo.yaml"),
                    help="pipeline config (read llm block to gate VLM fusion)")
    ap.add_argument("--vlm", action="store_true",
                    help="force-enable the VLM fingerprint fusion layer for this run")
    ap.add_argument("--judge", action="store_true",
                    help="let the VLM adjudicate borderline pairs (few calls)")
    ap.add_argument("--vlm-fp", type=Path, default=None, help="VLM fingerprint cache parquet")
    args = ap.parse_args()
    cache = args.cache or args.out.parent / f"{args.out.stem}_desc.npz"

    # Gate the VLM layer on config (llm.enabled + llm.uses.fingerprint); --vlm forces it on.
    import yaml
    from .llm import get_llm, uses
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config)) or {}
    if args.vlm:
        cfg.setdefault("llm", {})["enabled"] = True
        cfg["llm"].setdefault("uses", {})["fingerprint"] = True
    llm = get_llm(cfg)
    do_vlm = bool(llm) and uses(cfg, "fingerprint")

    # Descriptors (OSNet embedding, part-band Lab, frames) are needed either way.
    if not args.fuse_only:
        extract(args.video, args.tracks, cache, args.start_frame, args.end_frame,
                args.samples_per_track, args.device)

    if do_vlm:
        vlm_fp_path = args.vlm_fp or args.out.parent / f"{args.out.stem}_vlmfp.parquet"
        fp = vlm_fingerprints(args.video, args.tracks, vlm_fp_path, llm,
                              args.start_frame, args.end_frame)
        crops = None
        if args.judge:
            df = pd.read_parquet(args.tracks)
            lo = int(args.start_frame)
            hi = int(df["frame"].max()) if args.end_frame is None else int(args.end_frame)
            df = df[(df["frame"] >= lo) & (df["frame"] <= hi)]
            crops = _best_crops_per_track(args.video, df, lo, hi, per_track=1)
        fuse_vlm(cache, args.teams, fp, args.out, args.jersey, llm=llm, judge=args.judge,
                 threshold=args.threshold, crops_for_judge=crops)
    else:
        fuse(cache, args.teams, args.out, args.jersey, threshold=args.threshold)


if __name__ == "__main__":
    main()
