# football-vision

Pull amateur football matches (Veo auto-panning single-camera footage) from
YouTube, track every player, and burn a stats overlay onto the video —
boxes, motion traces, and `Player N` labels — plus (later) extracted stats
like possession, heatmaps, goals and assists.

## Input assumptions

- **Source:** the auto-panned Veo highlight video from YouTube (a virtual crop
  of a fixed 180° camera that pans/zooms to follow the ball).
- Because the visible view moves every frame, pitch mapping uses **per-frame
  homography** (M2), not a static transform.
- Players leave/re-enter frame during pans, so expect some **ID swaps**; whole-
  match identity is best-effort. (Upgrade: if you can export Veo's full
  panoramic feed, the camera is effectively static and everything gets easier.)

## Setup

```bash
# on zulzi-gpu (has CUDA); locally use --device cpu (slow)
uv venv && source .venv/bin/activate     # or: python -m venv .venv
uv pip install -e .                       # or: pip install -e .
```

`ultralytics` downloads the YOLO11 weights (`yolo11m.pt`) on first run.

## Run (M0 — watchable overlay)

```bash
# From a YouTube URL, full pipeline:
fv --url "https://youtu.be/XXXXXXXX"

# Quick test on a short slice of a local file (first ~900 frames):
fv --video data/raw/game.mp4 --max-frames 900 --device cuda:0

# Individual stages:
python -m fv.acquire "https://youtu.be/XXXX" --out-dir data/raw
python -m fv.detect_track data/raw/game.mp4 --out data/tracks/game.parquet
python -m fv.overlay data/raw/game.mp4 data/tracks/game.parquet --out data/out/game.mp4
```

Output: `data/out/<name>_annotated.mp4`.

## Layout

```
src/fv/
  acquire.py       # 1. yt-dlp download
  detect_track.py  # 2+3. YOLO11 detect + ByteTrack -> tracks.parquet
  overlay.py       # 7. draw boxes/labels/traces -> annotated mp4
  pipeline.py      # orchestrates the above (the `fv` command)
  events/          # 6. goals/assists (M3, stubbed)
  # teams.py, pitch.py, stats.py land in M1/M2
configs/veo.yaml   # model/threshold/overlay knobs
```

## Roadmap

- **M0 (this):** download → detect → track → overlay with `Player N` labels.
- **M1:** team assignment (jersey color → embedding), label polish, batch runs.
- **M2:** per-frame pitch homography → positions, heatmaps, distances, possession.
- **M3:** events — audio-spike candidates → ball-near-goal check → assisted
  one-click confirm of scorer/assister (+ scoreboard OCR if footage has it).
- **M4:** accuracy — fine-tune YOLO on your kits, appearance-embedding re-ID to
  cut ID swaps, football-specific detector (players + ball + referee).

## Notes

- Detection currently uses a COCO **person** detector (self-contained). Swap in
  a football-specific YOLO checkpoint to also get the ball/referee.
- Overlay video has no audio; audio is handled in M3 for event detection.
