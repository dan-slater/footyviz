"""M1 — assign each tracked player to a team by jersey colour.

Track IDs churn hard on Veo panning footage (a re-entering player becomes a new
ID), so colouring the overlay by *team* gives a durable visual anchor that
survives ID swaps. This module clusters track IDs into a few visual groups from
their shirt colour:

    A  -> field team 1 (largest cluster)
    B  -> field team 2 (second largest)
    O  -> "other": goalkeeper / referee / bibs (remaining small clusters)

Pipeline:
  1. For each track, sample up to N of its boxes spread across its lifespan.
  2. Crop the upper-body/shirt region, mask out pitch-green and shadow pixels,
     and take the median colour.
  3. Median the per-crop colours -> one colour feature per track.
  4. KMeans (k=3 or k=4, chosen by silhouette) over the track colours.
  5. Write track_id -> (team, representative BGR colour) to parquet.

The video is decoded ONCE per run (crops for all sampled frames are collected
in a single sequential pass). Per-track features are cached to .npz so
re-clustering (e.g. a different k) skips the decode entirely.

Because the source YouTube video is a *compilation* of several matches with
different kits, a single global clustering over the whole video is muddy — run
with --start-frame/--end-frame to scope clustering to one continuous clip,
which separates cleanly.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- shirt-crop geometry (fractions of the player box) ---
CROP_TOP = 0.15      # skip the head
CROP_BOT = 0.50      # ... down to mid-torso -> isolates the shirt
CROP_INSET = 0.15    # trim left/right to avoid arms / grass bleed

# --- pixel masks (OpenCV HSV: H 0-179, S/V 0-255) ---
GREEN_H = (30, 95)   # pitch green
GREEN_MIN_S = 40
GREEN_MIN_V = 35
DARK_MAX_V = 25      # shadow / black void (kept above this so dark kits survive)


def _shirt_pixels(crop: np.ndarray) -> np.ndarray | None:
    """Return the shirt pixels of a box crop as an (N,3) BGR array, or None."""
    h, w = crop.shape[:2]
    y0, y1 = int(h * CROP_TOP), int(h * CROP_BOT)
    x0, x1 = int(w * CROP_INSET), int(w * (1 - CROP_INSET))
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    region = crop[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (H >= GREEN_H[0]) & (H <= GREEN_H[1]) & (S >= GREEN_MIN_S) & (V >= GREEN_MIN_V)
    dark = V < DARK_MAX_V
    keep = ~(green | dark)
    px = region[keep]
    if len(px) < 15:  # mostly grass/shadow -> fall back to the whole region
        px = region.reshape(-1, 3)
    return px if len(px) else None


def _feature_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """4D colour feature from a representative BGR triple.

    Hue is projected onto the unit circle and weighted by saturation so that
    low-saturation kits (white/grey) don't inject noisy hue; saturation and
    value are appended so white (bright, desaturated) and dark kits separate.
    """
    px = np.uint8([[bgr]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    s_n, v_n = s / 255.0, v / 255.0
    ang = 2 * np.pi * (h / 180.0)
    return np.array([s_n * np.cos(ang), s_n * np.sin(ang), s_n, v_n], dtype=float)


def _sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    """Pick up to n frame indices spread evenly across a track's frames."""
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[np.unique(idx)]


def _cache_path(tracks: Path, lo: int, hi: int, n: int) -> Path:
    key = f"{tracks.resolve()}|{lo}|{hi}|{n}|{CROP_TOP}|{CROP_BOT}|{CROP_INSET}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return tracks.parent / f"{tracks.stem}_teamfeat_{lo}_{hi}_{n}_{digest}.npz"


def _collect_features(
    video: Path, df: pd.DataFrame, lo: int, hi: int, samples: int, cache: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (track_ids, features Nx4, rep_bgr Nx3) sampling the video once."""
    cpath = _cache_path(df.attrs["_tracks_path"], lo, hi, samples)  # keyed on tracks file
    if cache and cpath.exists():
        z = np.load(cpath)
        print(f"[cache] loaded track features from {cpath.name}")
        return z["track_ids"], z["features"], z["rep_bgr"]

    # Which (frame -> [(track_id, box), ...]) do we need to crop?
    needed: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        for f in _sample_frames(g["frame"].to_numpy(), samples):
            row = g[g["frame"] == f].iloc[0]
            box = (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
            needed.setdefault(int(f), []).append((int(tid), box))

    # Decode from frame 0 and count frames with grab() (frame-accurate, matches
    # the absolute frame numbers in the parquet). grab() skips the BGR copy for
    # frames we don't need; we only retrieve() the sampled frames.
    per_track: dict[int, list[np.ndarray]] = {}
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
        H, W = frame.shape[:2]
        for tid, (x1, y1, x2, y2) in items:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 4 or y2 - y1 < 6:
                continue
            px = _shirt_pixels(frame[y1:y2, x1:x2])
            if px is None:
                continue
            per_track.setdefault(tid, []).append(np.median(px, axis=0))
    cap.release()

    track_ids, features, rep_bgr = [], [], []
    for tid, colours in per_track.items():
        med = np.median(np.stack(colours), axis=0)  # BGR
        track_ids.append(tid)
        rep_bgr.append(med)
        features.append(_feature_from_bgr(med))
    track_ids = np.array(track_ids, dtype=int)
    features = np.array(features, dtype=float)
    rep_bgr = np.array(rep_bgr, dtype=float)

    if cache:
        np.savez(cpath, track_ids=track_ids, features=features, rep_bgr=rep_bgr)
        print(f"[cache] wrote track features -> {cpath.name}")
    return track_ids, features, rep_bgr


def _choose_k_and_cluster(features: np.ndarray, k: int | None):
    """Cluster; if k is None, pick k in {3,4} by silhouette (default 3)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(features)
    if k is not None:
        km = KMeans(n_clusters=min(k, n), n_init=10, random_state=0).fit(features)
        return km.labels_, min(k, n), None

    best = None
    for kk in (3, 4):
        if n <= kk:
            continue
        km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(features)
        sil = silhouette_score(features, km.labels_)
        if best is None or sil > best[2] + 1e-6:
            best = (km.labels_, kk, sil)
    if best is None:  # tiny input
        km = KMeans(n_clusters=min(3, n), n_init=10, random_state=0).fit(features)
        return km.labels_, min(3, n), None
    return best


def run(
    video: Path,
    tracks: Path,
    out: Path,
    k: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    samples_per_track: int = 30,
    cache: bool = True,
) -> Path:
    df = pd.read_parquet(tracks)
    df.attrs["_tracks_path"] = Path(tracks)
    lo = 0 if start_frame is None else int(start_frame)
    hi = int(df["frame"].max()) if end_frame is None else int(end_frame)
    win = df[(df["frame"] >= lo) & (df["frame"] <= hi)]
    n_tracks = win["track_id"].nunique()
    print(f"clustering tracks in frames [{lo}, {hi}]: {n_tracks} tracks, {len(win)} detections")

    track_ids, features, rep_bgr = _collect_features(video, win, lo, hi, samples_per_track, cache)
    print(f"got colour features for {len(track_ids)} / {n_tracks} tracks "
          f"(rest had no usable shirt crop)")
    if len(track_ids) < 2:
        raise SystemExit("not enough tracks with colour features to cluster")

    labels, kk, sil = _choose_k_and_cluster(features, k)
    print(f"KMeans k={kk}" + (f" (silhouette {sil:.3f})" if sil is not None else ""))

    # Rank clusters by membership; largest two -> A, B; rest -> O.
    order = pd.Series(labels).value_counts().index.tolist()
    cl_to_team = {}
    for rank, cl in enumerate(order):
        cl_to_team[cl] = "A" if rank == 0 else "B" if rank == 1 else "O"

    # Representative colour per team = median rep_bgr of its member tracks.
    team_colour: dict[str, np.ndarray] = {}
    for team in ("A", "B", "O"):
        mask = np.array([cl_to_team[l] == team for l in labels])
        if mask.any():
            team_colour[team] = np.median(rep_bgr[mask], axis=0)

    rows = []
    for tid, lab in zip(track_ids, labels):
        team = cl_to_team[lab]
        b, g, r = team_colour[team]
        rows.append({"track_id": int(tid), "team": team,
                     "color_b": int(b), "color_g": int(g), "color_r": int(r)})
    out_df = pd.DataFrame(rows).sort_values("track_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    # Report split.
    counts = out_df["team"].value_counts().to_dict()
    print("team split (tracks):", {t: counts.get(t, 0) for t in ("A", "B", "O")})
    for team in ("A", "B", "O"):
        if team in team_colour:
            b, g, r = team_colour[team].astype(int)
            print(f"  team {team}: BGR=({b},{g},{r})  RGB=({r},{g},{b})")
    print(f"wrote team map -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Assign tracks to teams by jersey colour.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/tracks/teams.parquet"))
    ap.add_argument("--k", type=int, default=None, help="force cluster count (default: pick 3/4 by silhouette)")
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--samples-per-track", type=int, default=30)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    run(args.video, args.tracks, args.out, k=args.k,
        start_frame=args.start_frame, end_frame=args.end_frame,
        samples_per_track=args.samples_per_track, cache=not args.no_cache)


if __name__ == "__main__":
    main()
