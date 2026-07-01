"""M4 Layer 1 — appearance re-ID tracking (boxmot BoT-SORT + OSNet).

Plain ByteTrack (M0) mints a fresh ID every time a player leaves the panning
crop or is occluded (~4000 IDs for ~10 players over the match). This stage
re-runs association with an appearance-embedding tracker (BoT-SORT driven by an
OSNet ReID model) so a re-entering player reclaims their track across short
gaps, and camera-motion compensation (ECC) undoes the pan between frames.

We reuse the M0 YOLO detections (data/tracks/*.parquet) rather than re-detecting
— BoT-SORT only needs (dets, frame). Output schema matches detect_track so the
rest of the pipeline (teams, overlay) is unchanged.

Long re-entries (a player off-screen for many seconds) are beyond any online
tracker's memory; those are collapsed later by offline fusion (fv.identity),
which clusters these tracklets by appearance + team + jersey number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

COLUMNS = ["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def _build_tracker(device: str, reid_weights: str, half: bool,
                   track_buffer: int, appearance_thresh: float):
    from boxmot.reid import ReID
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

    reid = ReID(weights=reid_weights, device=device, half=half)
    params = yaml.safe_load(open(get_tracker_config("botsort")))
    params = {p: v["default"] for p, v in params.items()}
    # Keep a lost track alive longer than the 1s default so short pans/occlusions
    # are bridged by the tracker itself; ReID appearance gate stays strict.
    params["track_buffer"] = track_buffer
    params["appearance_thresh"] = appearance_thresh
    tracker = create_tracker("botsort", reid_model=reid.model, device=device,
                             half=half, evolve_param_dict=params)
    return tracker


def run(
    video: Path,
    tracks: Path,
    out: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    device: str = "cuda:0",
    reid_weights: str = "osnet_x1_0_msmt17.pt",
    track_buffer: int = 90,
    appearance_thresh: float = 0.25,
    half: bool = False,
) -> Path:
    df = pd.read_parquet(tracks)
    lo = int(start_frame)
    hi = int(df["frame"].max()) if end_frame is None else int(end_frame)
    df = df[(df["frame"] >= lo) & (df["frame"] <= hi)]
    # Detections per frame as Nx6 [x1,y1,x2,y2,conf,cls].
    by_frame = {
        int(f): g[["x1", "y1", "x2", "y2", "confidence", "class_id"]].to_numpy(dtype=float)
        for f, g in df.groupby("frame")
    }

    tracker = _build_tracker(device, reid_weights, half, track_buffer, appearance_thresh)

    rows: list[dict] = []
    cap = cv2.VideoCapture(str(video))
    idx = -1
    processed = 0
    while idx < hi:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx < lo:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        dets = by_frame.get(idx, np.empty((0, 6), dtype=float))
        res = tracker.update(dets, frame)  # BoT-SORT: needs consecutive frames+imgs
        if len(res):
            xyxy = np.asarray(res.xyxy, dtype=float)
            ids = np.asarray(res.id, dtype=int)
            confs = np.asarray(res.conf, dtype=float)
            clss = np.asarray(res.cls, dtype=int)
            for (x1, y1, x2, y2), tid, cf, cl in zip(xyxy, ids, confs, clss):
                rows.append({"frame": idx, "track_id": int(tid), "class_id": int(cl),
                             "confidence": float(cf), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        processed += 1
        if processed % 300 == 0:
            n_ids = len({r["track_id"] for r in rows})
            print(f"  {processed} frames, {len(rows)} track-rows, {n_ids} ids so far")
    cap.release()

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"wrote {len(out_df)} rows, {out_df['track_id'].nunique()} unique reid tracks "
          f"across {out_df['frame'].nunique()} frames -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-track detections with BoT-SORT + OSNet ReID.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path, help="M0 detections/tracks parquet (source of boxes)")
    ap.add_argument("--out", type=Path, default=Path("data/tracks/reid.parquet"))
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reid-weights", default="osnet_x1_0_msmt17.pt")
    ap.add_argument("--track-buffer", type=int, default=90)
    ap.add_argument("--appearance-thresh", type=float, default=0.25)
    args = ap.parse_args()
    run(args.video, args.tracks, args.out, start_frame=args.start_frame,
        end_frame=args.end_frame, device=args.device, reid_weights=args.reid_weights,
        track_buffer=args.track_buffer, appearance_thresh=args.appearance_thresh)


if __name__ == "__main__":
    main()
