"""Experiment 13 — GK-role lock as a pre-pass before the neutral grouping (roster_v3).

Run 11 ("neutral + context + veto") grouped every tracklet as if it were an
outfield player, so the goalkeeper (night track 14) was merged into an outfield
row (A-4 = tracks 10, 12, 14). Here the keeper role is decided FIRST:

  1. GEOMETRY (code fact): each track's motion RELATIVE to the others on screen
     (per-frame centre displacement minus the median displacement of co-visible
     tracks, so the camera pan cancels). Keepers barely move relative to play.
  2. CONTEXT (VLM, neutral): the track's full-frame boxed views (data/tracks/
     context_night/, the same views run 11 used) with one neutral question:
     where is this person standing and what role do they have? Geometry is
     given as a measured fact, no hypothesis about who the keeper is.
  3. LOCK: role == goalkeeper AND location == in_goal_mouth AND confidence >= 0.6
     AND relative motion <= the slice median. Locked keepers are excluded from
     the outfield merges and get their own `<team>-GK` identity (keeper tracklets
     of one team merge only if the co-occurrence veto allows it).
  4. GROUP: unchanged roster_v3.cluster_team per team (emb_gate 0.55,
     min_conf 0.6, max_calls 40) on everyone else.

`--control` skips steps 1-3: a same-day re-run of run 11 (the VLM is stochastic,
so exp 13 is compared against this, not only against the July output).

    python -m fv.gk_lock --out data/tracks/identities_gklock_night.parquet
    python -m fv.gk_lock --control --out data/tracks/identities_ctrl11_night.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .llm import _parse_json, get_llm
from .roster import _emit, _name_clusters, load_context
from .roster_v3 import cluster_team, cooccur_veto

_P_ROLE = (
    "Each image is a full video frame from one 5-a-side football match. In every image ONE "
    "person is outlined by a green box labelled 'track {t}'. Look at where that boxed person "
    "is standing on the pitch and what they are doing in play.\n"
    "Measured fact (from tracking, not a guess): relative to the other people on screen this "
    "person moves {m:.1f} px/frame; the median over everyone in this clip is {med:.1f} px/frame.\n"
    'Return STRICT JSON {{"location": "in_goal_mouth" | "on_pitch_outfield" | "off_pitch", '
    '"role": "goalkeeper" | "outfield_player" | "referee" | "other", '
    '"confidence": 0.0-1.0, "reason": "<one line>"}}.'
)


def relative_motion(df: pd.DataFrame, tracks: list[int]) -> dict[int, float]:
    """Mean |own displacement - median co-visible displacement| per frame (px/frame)."""
    d = df[df["track_id"].isin(tracks)].sort_values(["track_id", "frame"]).copy()
    d["cx"], d["cy"] = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
    g = d.groupby("track_id")
    d["dx"], d["dy"] = g["cx"].diff(), g["cy"].diff()
    d.loc[g["frame"].diff() != 1, ["dx", "dy"]] = np.nan          # only consecutive frames
    d = d.dropna(subset=["dx"])
    med = d.groupby("frame")[["dx", "dy"]].transform("median")      # the pan, per frame
    n = d.groupby("frame")["dx"].transform("size")
    d = d[n >= 3]                                                   # need others to estimate pan
    rel = np.hypot(d["dx"] - med.loc[d.index, "dx"], d["dy"] - med.loc[d.index, "dy"])
    return rel.groupby(d["track_id"]).mean().astype(float).to_dict()


def load_views(ctx_dir: Path, tracks: list[int]) -> dict[int, list]:
    views: dict[int, list] = {t: [] for t in tracks}
    for f in sorted(ctx_dir.glob("t*_*.jpg")):
        m = re.match(r"t(\d+)_(\d+)\.jpg", f.name)
        if m and int(m.group(1)) in views:
            views[int(m.group(1))].append(cv2.imread(str(f)))
    return views


def role_of(llm, views, t, m, med, retries=3) -> dict:
    prompt = _P_ROLE.format(t=t, m=m, med=med)
    for _ in range(retries):
        try:
            d = _parse_json(llm._vision(prompt, views[t][:3], want_json=True))
            return {"location": str(d.get("location", "")), "role": str(d.get("role", "")),
                    "confidence": float(d.get("confidence", 0)), "reason": str(d.get("reason", ""))[:160]}
        except Exception as e:                                      # noqa: BLE001
            err = repr(e)[:100]
    return {"location": "", "role": "error", "confidence": 0.0, "reason": err}


def gk_prepass(llm, views, tracks, rel, workers=8):
    med = float(np.median([rel[t] for t in tracks if t in rel]))
    with ThreadPoolExecutor(workers) as ex:
        futs = {t: ex.submit(role_of, llm, views, t, rel.get(t, med), med) for t in tracks if views.get(t)}
    roles = {t: f.result() for t, f in futs.items()}
    locked = [t for t, r in roles.items()
              if r["role"] == "goalkeeper" and r["location"] == "in_goal_mouth"
              and r["confidence"] >= 0.6 and rel.get(t, med) <= med]
    return locked, roles, med


def gk_groups(locked, team_of, veto):
    """Keeper tracklets of one team join one identity unless the veto forbids it."""
    groups: list[list[int]] = []
    for t in sorted(locked):
        for g in groups:
            if team_of.get(g[0], "O") == team_of.get(t, "O") and \
                    all(frozenset((t, u)) not in veto for u in g):
                g.append(t)
                break
        else:
            groups.append([t])
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 13: GK-role lock pre-pass + neutral grouping (night slice).")
    ap.add_argument("--reid", default="data/tracks/reid_night_0_1800.parquet")
    ap.add_argument("--cache", default="data/tracks/identities_night_desc.npz")
    ap.add_argument("--teams", default="data/tracks/teams_reid_night.parquet")
    ap.add_argument("--jersey", default="data/tracks/jersey_reid_night.parquet")
    ap.add_argument("--vlm-fp", default="data/tracks/identities_vlm_night_vlmfp.parquet")
    ap.add_argument("--context-dir", default="data/tracks/context_night")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None, help="json decision log (default: next to --out)")
    ap.add_argument("--control", action="store_true", help="no pre-pass (same-day run-11 control)")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    args = ap.parse_args()

    llm = get_llm({"llm": {"enabled": True, "provider": "gemini", "model": args.model}})
    ctx = load_context(Path(args.cache), Path(args.teams), Path(args.jersey), Path(args.vlm_fp), 15)
    ctx_dir = Path(args.context_dir)
    tracks = sorted({int(re.match(r"t(\d+)_", f.name).group(1)) for f in ctx_dir.glob("t*_*.jpg")})
    df = pd.read_parquet(args.reid)
    views = load_views(ctx_dir, tracks)
    veto, _ = cooccur_veto(df[df["track_id"].isin(tracks)], tracks)
    print(f"[{'control' if args.control else 'exp13'}] {len(tracks)} tracks, {len(veto)} vetoed pairs, model {llm.model}")

    log: dict = {"mode": "control" if args.control else "gk_lock", "tracks": tracks}
    locked: list[int] = []
    if not args.control:
        rel = relative_motion(df, tracks)
        locked, roles, med = gk_prepass(llm, views, tracks, rel)
        log.update(rel_motion={int(k): round(v, 2) for k, v in rel.items()}, rel_motion_median=round(med, 2),
                   roles={int(k): v for k, v in roles.items()}, gk_locked=locked)
        print(f"  relative-motion median {med:.2f} px/frame; GK-voted:",
              [(t, roles[t]["location"], roles[t]["confidence"], round(rel.get(t, 0), 1))
               for t in roles if roles[t]["role"] == "goalkeeper"])
        print(f"  GK LOCKED: {locked}")

    groups, merges_all, calls_all = [], {}, 0
    for team in ("A", "B", "O"):
        ids = [t for t in tracks if ctx["team_of"].get(t, "O") == team and t not in locked]
        if not ids:
            continue
        g, merges, calls = cluster_team(ids, ctx["embs"], ctx["frames"], ctx["num_of"], veto, views, llm)
        print(f"  [team {team}] {len(ids)} tracks -> {len(g)} groups ({len(merges)} merges, {calls} VLM calls)")
        groups += g
        merges_all[team] = merges
        calls_all += calls
    by_track = _name_clusters(groups, ctx)
    for k, g in enumerate(gk_groups(locked, ctx["team_of"], veto)):
        team = ctx["team_of"].get(g[0], "O")
        n_same = sum(1 for h in gk_groups(locked, ctx["team_of"], veto) if ctx["team_of"].get(h[0], "O") == team)
        sid = f"{team}-GK" if n_same == 1 else f"{team}-GK{k + 1}"
        for t in g:
            by_track[t] = sid
    _emit(by_track, ctx, args.out)

    n_ids = len(set(by_track.values()))
    row14 = sorted(t for t, s in by_track.items() if s == by_track.get(14))
    log.update(n_identities=n_ids, vlm_calls_grouping=calls_all, merges=merges_all,
               track14_identity=by_track.get(14), track14_row=row14)
    print(f"  => {n_ids} identities; track 14 -> {by_track.get(14)} with tracks {row14}")
    logp = args.log or args.out.with_suffix(".json")
    logp.write_text(json.dumps(log, indent=1, default=str))
    print(f"  log -> {logp}")


if __name__ == "__main__":
    main()
