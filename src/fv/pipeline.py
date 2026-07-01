"""Orchestrate the M0 pipeline for one match: acquire -> detect_track -> overlay.

    fv --url https://youtu.be/XXXX --config configs/veo.yaml
    fv --video data/raw/game.mp4 --max-frames 900   # quick 30s-ish test pass

CLI flags override config values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import acquire, detect_track, overlay


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="football-vision M0 pipeline")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL to download")
    src.add_argument("--video", type=Path, help="local video file (skip download)")
    ap.add_argument("--config", type=Path, default=Path("configs/veo.yaml"))
    ap.add_argument("--workdir", type=Path, default=Path("data"))
    ap.add_argument("--device", help="override detect.device (e.g. cpu, cuda:0)")
    ap.add_argument("--model", help="override detect.model")
    ap.add_argument("--stride", type=int, help="override detect.stride")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (quick test)")
    args = ap.parse_args()

    cfg = load_config(args.config if args.config.exists() else None)
    det = cfg.get("detect", {})
    ovl = cfg.get("overlay", {})

    device = args.device or det.get("device", "cuda:0")
    model = args.model or det.get("model", "yolo11m.pt")
    stride = args.stride or det.get("stride", 1)

    # 1. Acquire
    if args.url:
        print("== acquire ==")
        video = acquire.download(
            args.url, args.workdir / "raw", cfg.get("acquire", {}).get("max_height", 1080)
        )
    else:
        video = args.video
    print(f"video: {video}")

    stem = video.stem
    tracks = args.workdir / "tracks" / f"{stem}.parquet"
    out = args.workdir / "out" / f"{stem}_annotated.mp4"

    # 2+3. Detect + track
    print("== detect + track ==")
    detect_track.run(
        video, tracks,
        model_path=model, device=device, conf=det.get("conf", 0.30),
        person_class_id=det.get("person_class_id", 0),
        stride=stride, max_frames=args.max_frames,
    )

    # 7. Overlay
    print("== overlay ==")
    overlay.run(
        video, tracks, out,
        trace_length=ovl.get("trace_length", 30),
        draw_traces=ovl.get("draw_traces", True),
    )
    print(f"\nDone. Annotated video: {out}")


if __name__ == "__main__":
    main()
