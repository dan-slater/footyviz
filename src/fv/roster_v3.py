"""M4 identity grouping v3 — neutral prompts + full-frame context + co-occurrence veto.

Fixes three validated problems with the earlier grouping:
  1. LEADING PROMPTS: earlier prompts told the model "hoodies = subs/spectators".
     That was wrong (grey-hoodie people are players on a cold night). All prompts
     here are NEUTRAL — everyone is a player unless the visuals prove a GK/ref
     role; no categories, counts, or hypotheses are supplied.
  2. TIGHT CROPS ONLY: the model never saw spatial context. Here each tracklet is
     shown as 2-3 FULL 1080p frames with that player boxed + labelled, so the VLM
     can reason about position (near goal? near boards? who's alongside?). Tight
     crops stay only for the OSNet embedding path.
  3. IMPOSSIBLE MERGES: a co-occurrence matrix is precomputed — two tracklets that
     are ever on screen together AND far apart are DIFFERENT people. This is a hard
     veto in code, independent of what the VLM says, and is stated to the VLM as
     fact so it never proposes an impossible merge.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .roster_v2 import _nums_incompatible, _best_number

COOCCUR_SEP_PX = 100     # co-present + centres farther than this => different people


def cooccur_veto(df: pd.DataFrame, tracks: list[int]):
    """Return (veto:set[frozenset], info:dict) — pairs that can NEVER merge."""
    cx = ((df["x1"] + df["x2"]) / 2)
    cy = ((df["y1"] + df["y2"]) / 2)
    pos = {}  # track -> {frame: (cx,cy)}
    for tid, g in df.assign(cx=cx, cy=cy).groupby("track_id"):
        pos[int(tid)] = dict(zip(g["frame"].astype(int), zip(g["cx"], g["cy"])))
    veto, info = set(), {}
    for a_i in range(len(tracks)):
        for b_i in range(a_i + 1, len(tracks)):
            a, b = tracks[a_i], tracks[b_i]
            shared = set(pos[a]) & set(pos[b])
            if not shared:
                continue
            seps = [float(np.hypot(pos[a][f][0] - pos[b][f][0], pos[a][f][1] - pos[b][f][1]))
                    for f in shared]
            med = float(np.median(seps))
            if med > COOCCUR_SEP_PX:
                veto.add(frozenset((a, b)))
                info[(a, b)] = (len(shared), round(med, 1))
    return veto, info


def render_context(video: Path, df: pd.DataFrame, tracks: list[int], out_dir: Path,
                   scale: float = 1.5, n: int = 3, width: int = 960):
    """Per track: n full frames (sampled across its life) with that player boxed."""
    from .identity import _sample_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    want: dict[int, list] = {}
    for tid in tracks:
        g = df[df["track_id"] == tid].sort_values("frame")
        for f in _sample_frames(g["frame"].to_numpy(), n):
            r = g[g["frame"] == f].iloc[0]
            want.setdefault(int(f), []).append((tid, (r.x1, r.y1, r.x2, r.y2)))
    views: dict[int, list] = {t: [] for t in tracks}
    cap = cv2.VideoCapture(str(video))
    idx = -1
    hi = max(want) if want else -1
    while idx < hi:
        if not cap.grab():
            break
        idx += 1
        items = want.get(idx)
        if not items:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        for tid, (x1, y1, x2, y2) in items:
            f = frame.copy()
            p1 = (int(x1 * scale), int(y1 * scale)); p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(f, p1, p2, (0, 255, 0), 4)
            cv2.putText(f, f"track {tid}", (p1[0], max(28, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / w)))
            views[tid].append(f)
    cap.release()
    for tid, imgs in views.items():
        for k, im in enumerate(imgs):
            cv2.imwrite(str(out_dir / f"t{tid}_{k}.jpg"), im)
    return views


_P_PAIR = (
    "Each image shows a single video frame from one football match with ONE person "
    "outlined by a green box and labelled with a track id. The two sets of images "
    "show track {a} and track {b}. Decide whether these two boxed people are the "
    "SAME individual person or two different people. Judge from appearance (hair, "
    "build, skin, clothing, socks, shoes) and where they are on the pitch. "
    'Return STRICT JSON {{"same": true|false, "confidence": 0.0-1.0, "reason": "<one line>"}}.'
)


def vlm_same(llm, views, a, b, cooccur_note=""):
    imgs = views.get(a, [])[:2] + views.get(b, [])[:2]
    if not imgs:
        return False, 0.0, "no views"
    prompt = _P_PAIR.format(a=a, b=b) + (f"\nFact: {cooccur_note}" if cooccur_note else "")
    from .llm import _parse_json
    try:
        d = _parse_json(llm._vision(prompt, imgs, want_json=True))
        return bool(d.get("same")), float(d.get("confidence", 0)), str(d.get("reason", ""))[:120]
    except Exception as e:
        return False, 0.0, repr(e)[:80]


def cluster_team(tracks, embs, frames, num_of, veto, views, llm,
                 emb_gate=0.55, min_conf=0.6, max_calls=40):
    """Agglomerative merge: closest embedding pair, VLM-verified on context views,
    hard-blocked by co-occurrence veto and confident-number incompatibility."""
    groups = [[t] for t in tracks]
    merges, blocked, calls, tried = [], [], 0, set()

    def cent(g):
        v = np.mean([embs[t] for t in g], axis=0); return v / (np.linalg.norm(v) + 1e-9)

    def rep(g):
        return max(g, key=lambda t: len(frames[t]))

    def can_merge(gi, gj):
        for a in gi:
            for b in gj:
                if frozenset((a, b)) in veto:
                    return False
        nums = {num_of[t] for t in gi + gj if t in num_of}
        return not _nums_incompatible(nums)

    while calls < max_calls:
        cand = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if not can_merge(groups[i], groups[j]):
                    continue
                key = frozenset((rep(groups[i]), rep(groups[j])))
                if key in tried:
                    continue
                d = 1 - float(cent(groups[i]) @ cent(groups[j]))
                if d <= emb_gate:
                    cand.append((d, i, j, key))
        if not cand:
            break
        cand.sort(key=lambda c: c[0])
        _, i, j, key = cand[0]
        tried.add(key)
        same, conf, reason = vlm_same(llm, views, rep(groups[i]), rep(groups[j]))
        calls += 1
        if same and conf >= min_conf and can_merge(groups[i], groups[j]):
            merges.append((rep(groups[i]), rep(groups[j]), round(conf, 2), reason))
            groups[i] += groups[j]; groups.pop(j); tried.clear()
        # else keep trying next-closest untried pair
    return groups, merges, calls
