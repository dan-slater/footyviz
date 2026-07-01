"""Stage 7 — burn boxes, motion traces and "Player N" labels onto the video.

Reads the persisted tracks parquet and re-draws over the original frames.
Note: supervision's VideoSink writes video only. Audio (needed for M3 event
detection anyway) is muxed back later with ffmpeg if required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv


def run(
    video: Path,
    tracks: Path,
    out: Path,
    trace_length: int = 30,
    draw_traces: bool = True,
) -> Path:
    df = pd.read_parquet(tracks)
    by_frame = {int(f): g for f, g in df.groupby("frame")}

    info = sv.VideoInfo.from_video_path(str(video))
    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    trace = sv.TraceAnnotator(trace_length=trace_length)

    out.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out), info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(str(video))):
            group = by_frame.get(i)
            if group is not None and len(group):
                detections = sv.Detections(
                    xyxy=group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
                    confidence=group["confidence"].to_numpy(dtype=float),
                    class_id=group["class_id"].to_numpy(dtype=int),
                    tracker_id=group["track_id"].to_numpy(dtype=int),
                )
                labels = [f"Player {tid}" for tid in detections.tracker_id]
                if draw_traces:
                    frame = trace.annotate(frame, detections)
                frame = box.annotate(frame, detections)
                frame = label.annotate(frame, detections, labels=labels)
            sink.write_frame(frame)
    print(f"wrote annotated video -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Burn tracking overlay onto the video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/out/annotated.mp4"))
    ap.add_argument("--trace-length", type=int, default=30)
    ap.add_argument("--no-traces", action="store_true")
    args = ap.parse_args()
    run(args.video, args.tracks, args.out, args.trace_length, not args.no_traces)


if __name__ == "__main__":
    main()
