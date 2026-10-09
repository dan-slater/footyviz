"""Stage 2+3 — detect players (YOLO11) and track them (ByteTrack).

Writes one row per (frame, tracked player) to a parquet file. Persisting the
tracks means overlay/stats can iterate cheaply without re-running detection.

Frames are indexed by ABSOLUTE frame number so overlay/stats can line up
regardless of the sampling stride.

With ball detection on (default), COCO class 32 "sports ball" is detected in the
same forward pass but NOT tracked: the best ball box per frame goes to a sibling
`<out>_ball.parquet` (frame, confidence, x1..y2). The person parquet stays
person-only so teams / reid / identity / overlay are unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import supervision as sv
from ultralytics import YOLO

COLUMNS = ["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
BALL_COLUMNS = ["frame", "confidence", "x1", "y1", "x2", "y2"]
BALL_CLASS_ID = 32          # COCO "sports ball"


def ball_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}_ball.parquet")


def run(
    video: Path,
    out: Path,
    model_path: str = "yolo11m.pt",
    device: str = "cuda:0",
    conf: float = 0.30,
    person_class_id: int = 0,
    stride: int = 1,
    max_frames: int | None = None,
    ball: bool = True,
    ball_conf: float = 0.10,
    imgsz: int = 640,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    generator = sv.get_video_frames_generator(str(video), stride=stride)
    total = sv.VideoInfo.from_video_path(str(video)).total_frames // max(stride, 1)
    if max_frames is not None:
        total = min(total, max_frames)
    classes = [person_class_id] + ([BALL_CLASS_ID] if ball else [])
    det_conf = min(conf, ball_conf) if ball else conf

    rows: list[dict] = []
    ball_rows: list[dict] = []
    processed = 0
    for i, frame in enumerate(generator):
        if max_frames is not None and processed >= max_frames:
            break
        result = model(
            frame, verbose=False, device=device, conf=det_conf, classes=classes, imgsz=imgsz
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if ball:
            is_ball = detections.class_id == BALL_CLASS_ID
            balls = detections[is_ball]
            if len(balls):
                k = int(balls.confidence.argmax())
                bx1, by1, bx2, by2 = (float(v) for v in balls.xyxy[k])
                ball_rows.append({"frame": i * stride, "confidence": float(balls.confidence[k]),
                                  "x1": bx1, "y1": by1, "x2": bx2, "y2": by2})
            detections = detections[~is_ball]
        detections = detections[detections.confidence >= conf]
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
        if progress is not None and processed % 30 == 0:
            progress(processed, total)

    df = pd.DataFrame(rows, columns=COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} detections across {df['frame'].nunique()} frames -> {out}")
    if ball:
        bdf = pd.DataFrame(ball_rows, columns=BALL_COLUMNS)
        bdf.to_parquet(ball_path(out), index=False)
        print(f"ball detected in {len(bdf)}/{processed} frames -> {ball_path(out)}")
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
    ap.add_argument("--no-ball", action="store_true", help="persons only (skip sports-ball pass)")
    ap.add_argument("--ball-conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    run(
        args.video, args.out, args.model, args.device,
        args.conf, stride=args.stride, max_frames=args.max_frames,
        ball=not args.no_ball, ball_conf=args.ball_conf, imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
