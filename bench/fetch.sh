#!/usr/bin/env bash
# footyviz-bench: fetch the 840-1080 s source windows for every match in bench/manifest.json.
#
# The footage is NOT redistributed (licence: bench/manifest.json). Run this with your own
# YouTube access. YouTube bot-walls repeated downloads; pass cookies from your own browser:
#     COOKIES="--cookies-from-browser chrome" bench/fetch.sh
# Cookies stay on the machine that runs this script; never copy them anywhere.
#
# What it does, per match (exactly what the control plane runs on the Mac):
#   yt-dlp --download-sections "*840-1080"   # a STREAM COPY: no re-encode
#   -> $DATA/windows/<eyeball_tier>_<video_id>_840-1080.mp4
# A stream copy starts on the keyframe at or before 840 s (835.835 s for Rookie's fixed
# 150-frame GOP at 29.97 fps), and its timestamps are re-zeroed with no edit list. So the
# 120 s labelled clip is NOT cut here: `python -m fv.bench cut` re-times each window
# (manifest source.src_t0, BENCHMARK.md 7) and frame-accurately re-encodes clips/<id>.mp4.
#
# Requirements: yt-dlp, a JS runtime for YouTube's n-challenge (deno on PATH), ffmpeg.
# Usage: bench/fetch.sh [ID ...]     (default: every match in the manifest)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${FVBENCH_DATA:-$HOME/footyviz-bench-data}"
COOKIES="${COOKIES:-}"
FMT='bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]'
mkdir -p "$DATA/windows"

python3 - "$@" <<'EOF' | while read -r id tier vid; do
import json, sys
m = json.load(open("bench/manifest.json"))
want = set(sys.argv[1:])
for x in m["matches"]:
    if want and x["id"] not in want:
        continue
    print(x["id"], x["eyeball_tier"], x["video_id"])
EOF
  out="$DATA/windows/${tier}_${vid}_840-1080.mp4"
  if [[ -s "$out" ]]; then echo "[$id] have $out"; continue; fi
  echo "[$id] fetching $vid 840-1080 s -> $out"
  # shellcheck disable=SC2086
  yt-dlp $COOKIES -f "$FMT" --merge-output-format mp4 \
    --download-sections "*840-1080" \
    -o "$out" "https://www.youtube.com/watch?v=$vid"
  # sanity: about 244 s, 1080p
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate:format=duration \
    -of default=nw=1 "$out" || true
done

echo "next: PYTHONPATH=src python -m fv.bench cut && python -m fv.bench measure"
