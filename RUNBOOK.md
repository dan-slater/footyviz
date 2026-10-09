# footyviz (football-vision) — RUNBOOK

Operational guide: environment, how to run each stage, config, outputs, gotchas.
For project *state / decisions*, see `HANDOVER.md`. For a visual architecture tour, see the
5-tab dossier artifact: https://claude.ai/code/artifact/4a561329-21be-4376-acb9-2c92c80756d7

---

## 0. Where things live

| | Location |
|---|---|
| **Canonical repo + all data** | `zulzi-gpu:~/football-vision` (git; HEAD `e064dd6` = docs, identity work `a467ef7`) |
| Local Mac copy | `/Users/ds/football-vision` — original M0 scaffold + `system-overview.html` + pulled `data/out/*.mp4` only. **NOT the full code** — the identity/LLM work lives on zulzi. |
| GPU | RTX 4090, 24 GB, on `zulzi-gpu` (Tailscale `100.120.149.46`) |
| venv | `~/football-vision/.venv` (uv-managed, **Python 3.12** — 3.14 has no torch wheels) |
| Keys | repo-local gitignored `.env` only (`GEMINI_API_KEY=`, `OPENAI_API_KEY=`; template `.env.example`), auto-loaded by `fv.llm`. **Never** source another project's `.env`. The VLM layer is off by default and `fv.job` never enables it. |

### Box access
- Normal: `ssh zulzi-gpu` (Tailscale).
- **If direct SSH times out** (Tailscale hiccup on the Mac), hop: `ssh devboxv2 'ssh zulzi-gpu "…"'` — devboxv2 stays on the tailnet and can reach zulzi.
- Disk: ~325 G free (2026-10). Still render only to H.264 (the overlay writer does this by default now).

### Driving the box
Work is done through a persistent `claude` factory line (`ORCH-fv`) via the
`remote-tmux-orchestrator` skill (`~/.claude/skills/remote-tmux-orchestrator/factory.sh`).
Feed prompt files with `factory.sh task zulzi-gpu fv <file>`; read with `view`; handle install
prompts with `approve`/`deny`. Everything below can also be run by hand over `ssh`.

---

## 1. Environment (one-time / rebuild)

```bash
cd ~/football-vision
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv missing
~/.local/bin/uv venv --python 3.12 .venv
~/.local/bin/uv pip install -e .
# extras for the research identity stack + optional API (boxmot 16.0.11 is pinned in pyproject:
# boxmot 25.x moved ReID/tracker_zoo and breaks reid_track/identity):
~/.local/bin/uv pip install -e ".[api,vlm,sr]" easyocr
# torch CUDA build (if cuda.is_available() is False):
~/.local/bin/uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# deno (for yt-dlp's YouTube n-challenge -> unlocks 1080p):
curl -fsSL https://deno.land/install.sh | sh             # PATH: ~/.deno/bin
```
Verify GPU: `.venv/bin/python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"`

---

## 2. Running the pipeline

### One job end to end (the product entry point)
```bash
.venv/bin/python -m fv.job --url "https://www.youtube.com/watch?v=hhaWMwLlugE" \
  --start-s 666.667 --end-s 726.667 --out data/jobs/demo_day     # or --url data/raw/<file>.mp4
```
~55 s for a 60 s clip on the 4090 (detect 25 s, reid+identity 15 s, stats 5 s, render 10 s).
Writes `overlay.mp4` (H.264, ≤2.2 Mb/s ≈ 9 MB/min), `poster.jpg`, `roster.jpg`, `crop_*.jpg`,
`stats.json` (schema: `API.md`), plus intermediates (`tracks.parquet`, `tracks_ball.parquet`,
`reid.parquet`, `teams.parquet`, `identities*.parquet`). Cached source: `data/raw/<id>_720.mp4`.
**zulzi's IP is bot-walled by YouTube** ("confirm you're not a bot"): new URLs need
`--cookies cookies.txt`; existing sources are reused from `data/raw/` without a network call.
Whistles use the clip's audio, else the sibling `<id>_1080.mp4` (the 720p file has no audio).

### Job API (optional, local)
`fv.api` = FastAPI queue over `fv.job` (one GPU job at a time, subprocess per job, state in
`data/jobs/<id>/{job.json,progress.json,job.log}`), contract in `API.md`. On zulzi it runs in
tmux session **`footyviz-api`** on **127.0.0.1:8790**, log `~/orch-scratch/footyviz-api.log`:
```bash
tmux new-session -d -s footyviz-api -c ~/football-vision \
  "PYTHONWARNINGS=ignore .venv/bin/python -m fv.api --port 8790 2>&1 | tee -a ~/orch-scratch/footyviz-api.log"
curl -s 127.0.0.1:8790/health
curl -s -XPOST 127.0.0.1:8790/jobs -H 'content-type: application/json' \
  -d '{"url":"data/raw/hhaWMwLlugE_720.mp4","start_s":0,"end_s":20}'      # -> {"id": ...}
curl -s 127.0.0.1:8790/jobs/<id>                                          # poll
curl -s 127.0.0.1:8790/jobs/<id>/files/stats.json
```
Via the API a local `url` must live under `data/raw/`. Max clip `FOOTYVIZ_MAX_CLIP_S` (180).
Restart = `tmux kill-session -t footyviz-api` then the line above (queued jobs re-queue; a job
that was mid-run is marked error).

### Demo drop for the web line
`~/orch-scratch/footyviz-demo/{day,night}/` = copies of `data/jobs/demo2_{day,night}/`
(overlay.mp4, poster.jpg, roster.jpg, stats.json, crop_*.jpg). Day = frames 20000–21800 (dusk),
night = frames 0–1800.

All commands from `~/football-vision`, using `.venv/bin/python`. Crops for identity/OCR come
from the **1080p** file; fast passes use 720p.

### Acquire
```bash
# 720p combined (no ffmpeg merge needed):
.venv/bin/yt-dlp -f "best[ext=mp4][height<=720]" -o data/raw/%(id)s_720.mp4 "<URL>"
# 1080p (YouTube ceiling here) with audio, via imageio-ffmpeg + deno:
FFLOC=$(.venv/bin/python -c "import imageio_ffmpeg,os;print(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))")
PATH="$HOME/.deno/bin:$PATH" .venv/bin/yt-dlp --ffmpeg-location "$FFLOC" \
  -f "bestvideo[height<=1080]+bestaudio/best" --merge-output-format mp4 \
  -o data/raw/%(id)s_1080.mp4 "<URL>"
```

### Detect + track  →  `data/tracks/<name>.parquet`
```bash
.venv/bin/python -m fv.detect_track data/raw/<file>.mp4 --out data/tracks/<name>.parquet \
  --model yolo11m.pt --device cuda:0 --conf 0.30
# quick slice: add --max-frames 900
```
~120 fps on the 4090. Parquet columns: `frame track_id class_id confidence x1 y1 x2 y2`
(frame = **absolute** index).

### Teams  →  A / B / O
```bash
.venv/bin/python -m fv.teams data/raw/<file>.mp4 data/tracks/<name>.parquet \
  --out data/tracks/teams_<name>.parquet
```

### Identity (the hard stage)  — modules: reid_track / jersey / sr / participant / roster_v3
Latest/best entry point is the neutral-prompt + context + co-occurrence path (`roster_v3.py` /
`identity.py`). Requires the Gemini key exported:
```bash
# put GEMINI_API_KEY=... in ~/football-vision/.env (gitignored); fv.llm loads it
# (exact invocation is wired in roster_v3/identity — the fv line knows it; see git log for the
#  command used per run. Outputs: data/tracks/identities_<variant>_night.parquet (+ _table).)
```
Sub-steps that exist as modules: `reid_track.py` (BoT-SORT+OSNet), `jersey.py` (easyocr number
votes), `sr.py` (Real-ESRGAN via spandrel), `participant.py` (player/GK/ref/spectator — **use
geometry as ground truth, prompt neutrally**), `roster_v3.py` (multi-view grids → holistic VLM
grouping → co-occurrence veto).

### Events
```bash
.venv/bin/python -m fv.events.whistle data/raw/<file>_1080.mp4 --out data/out/whistles.csv
# scoreboard: investigated & ruled out 2026-07-02 (likely a drinks advert, no recoverable digits) — see HANDOVER. Evidence: data/out/sb3_*.jpg
```

### Overlay render  →  H.264
```bash
.venv/bin/python -m fv.overlay data/raw/<file>.mp4 data/tracks/<identities>.parquet \
  --out data/out/<name>_fifa.mp4 --style fifa    # or --style box (debug)
# if it writes mp4v, transcode to h264 and DELETE the mp4v (disk!).
```

---

## 3. Config — `configs/veo.yaml`
```yaml
detect:   { model: yolo11m.pt, device: cuda:0, conf: 0.30, stride: 1 }
overlay:  { style: fifa }            # fifa | box
llm:
  enabled: false                     # MASTER SWITCH — pipeline runs fully without it
  provider: gemini                   # gemini | openai | local(ollama)
  model: gemini-3.1-pro-preview
  uses: { fingerprint: true, events: true, scoreboard: false, report: true }
```

---

## 4. Key outputs (on zulzi `data/`)
- `tracks/full.parquet` — full-match detections (~488k rows).
- `tracks/reid_night_0_1800.parquet` — night-slice tracklets used for identity work.
- `tracks/identities_neutral_night.parquet` / `identities_sr2_night(_table)` — latest rosters.
- `tracks/jersey_night_0_1800.parquet` — number OCR (col `confident` matters!).
- `out/*.mp4` — renders (`ident_neutral_fifa_h264.mp4` is the latest FIFA roster).
- `out/roster_montage_{A,B,O}.jpg` — per-identity crop grids (audit view).
- `out/participant_classification.csv`, `out/whistles.csv`.
- `data/models/RealESRGAN_x4plus.pth`.

### Review dashboard (regenerable)
`build_dashboard.py` builds a **self-contained HTML review console** — every stage's grids/frames + all per-variant overlay videos base64-inlined into one <16 MB fragment for a claude.ai Artifact. Videos are compressed to 300p/10fps, images to ~800px/q72 (the Artifact hard cap is 16 MB; full-res stays on the box; the 600 MB full-match video is deliberately not inlined).
```bash
.venv/bin/python build_dashboard.py            # -> data/out/dashboard.html (recomputes ID counts from parquet)
```
Then from the Mac: `scp zulzi-gpu:~/football-vision/data/out/dashboard.html .` and redeploy to the SAME Artifact URL (`https://claude.ai/code/artifact/9b09eb2c-1f14-4e83-b9c0-4d055ceb6be8`). To add a new identity experiment, append it to the `VARIANTS` list and re-run.

### Experiment tracker (PRIMARY living lab doc — tabbed)
`build_exp_tracker.py` builds a **tabbed** self-contained HTML: one tab per identity run
(setup / result / notes) + an Overview chart. This is the doc we iterate on.
```bash
.venv/bin/python build_exp_tracker.py     # -> data/out/dashboard_experiments.html
```
Then Mac: `scp zulzi-gpu:~/football-vision/data/out/dashboard_experiments.html .` → redeploy Artifact
`https://claude.ai/code/artifact/220dbdc8-10a9-4d8a-aa00-86d65f7aa6e1`. **Add a new experiment** = run the
pipeline variant, add a `VARIANTS` entry (run#, label, tag, count-parquet, video, frames, montages, setup-note),
re-run. Reuses the video cache in `data/out/_dash_cache/`.

### PhD report (on the Mac, currently paused)
`/Users/ds/football-vision-report` (git, Mac). Compile: `latexmk -pdf -bibtex -interaction=nonstopmode main.tex`
(local MacTeX). Written: Ch1, Ch9, App A, App B, `references.bib`. Written by **Mac-local factory lines**
(`~/orch-mac-lines/macline.sh spawn <name> <cwd> <model>`; `sayfile`/`view`/`approve`). See its `PLAN.md`/`STATE.md`.

---

## 5. Gotchas (bite-marks)
- **Python 3.12 only** (uv-pinned). System python is 3.14 → no torch wheels.
- **`basicsr` is broken** with modern torch (`functional_tensor` import). Use **`spandrel`** to load Real-ESRGAN. Don't reinstall basicsr (it left a 6 G cache once).
- **boxmot**: pinned 16.0.11 — 25.x moved `ReID`/`tracker_zoo`; 16.0.11 `tracker.update` returns an (N,8) ndarray (handled in reid_track).
- **YouTube bot wall** on zulzi: pass `--cookies` to `fv.job`, or reuse `data/raw/`.
- **yt-dlp 1080p** needs deno (n-challenge) + imageio-ffmpeg (`--ffmpeg-location`).
- **Gemini is a leading-prompt risk** — ask neutrally, feed geometry as fact (see HANDOVER lessons).
- **Overlay VideoSink is video-only** — audio handled separately (events).
- **Factory lines: `ORCH-fv` and `ORCH-sb` were retired 2026-07-02** (they were ~647 k / ~248 k tokens — very heavy). Spin a **fresh** line for new work; don't revive the old ones. `metro` on zulzi is a different project — leave it.
