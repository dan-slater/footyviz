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

## Run (M1 — team colours)

Track IDs churn on panning footage, so colour the overlay by *team* instead —
a durable anchor that survives ID swaps. `fv.teams` clusters each track's shirt
colour (HSV, pitch-green/shadow masked) with KMeans into A / B / other (GK/ref).

```bash
# Assign teams for one continuous clip (scope with --start/--end-frame; the
# source is a compilation so a single global clustering is muddy — cluster per clip).
python -m fv.teams data/raw/game.mp4 data/tracks/game.parquet \
    --out data/tracks/game_teams.parquet --start-frame 0 --end-frame 1000

# Overlay coloured by team ("A | 1237" / "B | 1209"); omit --teams for plain "Player N".
python -m fv.overlay data/raw/game.mp4 data/tracks/game.parquet \
    --out data/out/game_teams.mp4 --teams data/tracks/game_teams.parquet
```

## Run (M4 — stable player identity)

```bash
W="--start-frame 0 --end-frame 1800"   # a continuous clip
# 1. Re-track with appearance ReID (BoT-SORT + OSNet) fed by the M0 detections.
python -m fv.reid_track data/raw/game.mp4 data/tracks/game.parquet --out data/tracks/reid.parquet $W
# 2. Team map + (optional) jersey numbers for the reid tracklets.
python -m fv.teams  data/raw/game.mp4 data/tracks/reid.parquet --out data/tracks/reid_teams.parquet  $W
python -m fv.jersey data/raw/game.mp4 data/tracks/reid.parquet --out data/tracks/reid_jersey.parquet $W
# 3. Fuse tracklets -> stable identities (fingerprint + OSNet + team gate + jersey tiebreak).
python -m fv.identity data/raw/game.mp4 data/tracks/reid.parquet $W \
    --teams data/tracks/reid_teams.parquet --jersey data/tracks/reid_jersey.parquet \
    --out data/tracks/identities.parquet --threshold 0.55
# 4. Overlay labelled by stable id, coloured by team.
python -m fv.overlay data/raw/game.mp4 data/tracks/reid.parquet --out data/out/ident.mp4 \
    --teams data/tracks/reid_teams.parquet --identities data/tracks/identities.parquet
```

Output: `data/out/<name>_annotated.mp4`.

## Layout

```
src/fv/
  acquire.py       # 1. yt-dlp download
  detect_track.py  # 2+3. YOLO11 detect + ByteTrack -> tracks.parquet
  overlay.py       # 7. draw boxes/labels/traces -> annotated mp4 (--teams/--identities)
  pipeline.py      # orchestrates the above (the `fv` command)
  teams.py         # M1. jersey-colour -> team A/B/other
  reid_track.py    # M4. BoT-SORT + OSNet appearance re-ID tracking
  jersey.py        # M4. jersey-number OCR per track (weak identity signal)
  identity.py      # M4. fingerprint+embedding fusion -> stable player identities
  events/          # 6. goals/assists (M3, stubbed)
  # pitch.py, stats.py land in M2
configs/veo.yaml   # model/threshold/overlay knobs
```

## Roadmap

- **M0:** download → detect → track → overlay with `Player N` labels.
- **M1 (done, color):** team assignment by jersey colour (`fv.teams`), team-coloured
  overlay. Upgrade path: jersey **embedding** for cross-clip/robust team ID, label polish, batch runs.
- **M4 (done, stable identity):** collapse the ~4000 churning ByteTrack IDs into a
  small stable set of per-player identities. Layers: `fv.reid_track` (BoT-SORT +
  OSNet appearance ReID) → `fv.identity` (part-based appearance fingerprint —
  hair/skin/shirt/shorts/socks/shoes dominant colour + build — fused with the OSNet
  embedding, team as a hard gate, temporal-overlap cannot-link, `fv.jersey` number
  as a weak tie-breaker) → overlay labelled by stable id (`A#7` / `A-3`).
- **M2:** per-frame pitch homography → positions, heatmaps, distances, possession.
- **M3:** events — audio-spike candidates → ball-near-goal check → assisted
  one-click confirm of scorer/assister (+ scoreboard OCR if footage has it).
- **M4:** accuracy — fine-tune YOLO on your kits, appearance-embedding re-ID to
  cut ID swaps, football-specific detector (players + ball + referee).

## Notes

- Detection currently uses a COCO **person** detector (self-contained). Swap in
  a football-specific YOLO checkpoint to also get the ball/referee.
- Overlay video has no audio; audio is handled in M3 for event detection.
