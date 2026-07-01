"""Stage 2+3 — detect players (YOLO11) and track them (ByteTrack).

Writes one row per (frame, tracked player) to a parquet file. Persisting the
tracks means overlay/stats can iterate cheaply without re-running detection.

Frames are indexed by ABSOLUTE frame number so overlay/stats can line up
regardless of the sampling stride.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import supervision as sv
from ultralytics import YOLO

COLUMNS = ["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def run(
    video: Path,
    out: Path,
    model_path: str = "yolo11m.pt",
    device: str = "cuda:0",
    conf: float = 0.30,
    person_class_id: int = 0,
    stride: int = 1,
    max_frames: int | None = None,
) -> Path:
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    generator = sv.get_video_frames_generator(str(video), stride=stride)

    rows: list[dict] = []
    processed = 0
    for i, frame in enumerate(generator):
        if max_frames is not None and processed >= max_frames:
            break
        result = model(
            frame, verbose=False, device=device, conf=conf, classes=[person_class_id]
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        abs_frame = i * stride
        for xyxy, class_id, confidence, track_id in zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence,
            detections.tracker_id,
        ):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            rows.append(
                {
                    "frame": abs_frame,
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }
            )
        processed += 1
        if processed % 500 == 0:
            print(f"  processed {processed} frames, {len(rows)} detections so far")

    df = pd.DataFrame(rows, columns=COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} detections across {df['frame'].nunique()} frames -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect + track players -> parquet.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/tracks/tracks.parquet"))
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    run(
        args.video, args.out, args.model, args.device,
        args.conf, stride=args.stride, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
