# footyviz

Turn an amateur football video (a YouTube or Veo link) into:

- **`overlay.mp4`**: the clip with a broadcast-style marker over every player, coloured by team (H.264, plays on phones)
- **`stats.json`**: possession, distance covered, per-team heatmaps, a player roster and whistle events
- **`roster.jpg`**: the best crop of every player the pipeline found, grouped by team
- **`poster.jpg`**: one representative overlaid frame

It runs on your own machine. You use your own GPU, and your own API keys if you turn the optional vision-LLM layer on.
Open source (MIT). Built and tested on a 39-minute Cape Town 5-a-side filmed by a Veo
auto-panning camera ([the test match](https://www.youtube.com/watch?v=hhaWMwLlugE)).

> **Honesty line.** Identity is capped by the footage: on the reference night slice the
> pipeline found ~23 stable identities vs ~14–16 true players (dark floodlit pitch, near-identical
> kits, ~35 px figures, constant camera pan). **Team-level stats do not depend on perfect
> identity**; per-player numbers do. Every `stats.json` carries this line in `identity_note`,
> and every overlay has it burned in as a footer.

## Run it on your match

```bash
git clone https://github.com/dan-slater/footyviz && cd footyviz
curl -LsSf https://astral.sh/uv/install.sh | sh        # skip if you already have uv
uv venv --python 3.12 .venv
uv pip install -e .
.venv/bin/python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU only')"
.venv/bin/python -m fv.job --url "https://www.youtube.com/watch?v=hhaWMwLlugE" --start-s 667 --end-s 727 --out out/my_match
.venv/bin/python -m json.tool out/my_match/stats.json | head -60
```

Swap in your own match URL and a window of up to 180 s (`--start-s` / `--end-s`, in seconds of
the source video). `--url` also takes a local video file. The run prints progress per stage and
leaves `overlay.mp4`, `poster.jpg`, `roster.jpg`, `stats.json` (and `crop_*.jpg`) in `--out`.
The first run downloads model weights (YOLO11m, OSNet; ~60 MB).

- **GPU:** an NVIDIA GPU with CUDA is strongly recommended (a 60 s clip takes ~1–2 min on an RTX 4090).
  Python 3.12 is used because torch has no wheels for 3.14 yet. If the check above prints `CPU only` on a
  machine that has an NVIDIA GPU, install the CUDA build of torch:
  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
- **CPU fallback:** add `--device cpu` to the `fv.job` command. It works but is **slow**: expect
  tens of minutes for a 60 s clip. Use a short window (`--end-s` 20–30 s after `--start-s`) to try it.
- `--no-identity` skips the re-identification + identity stage (faster). The roster then lists raw tracker IDs.

## What's in `stats.json`

Full field-by-field schema: **[API.md](API.md#statsjson-schema_version-1)**. In short:

| key | what | how much to trust it |
|---|---|---|
| `teams` | the two outfield kits A/B: colour (hex) + name | good: kit colour clustering holds day and night |
| `possession` | A / B / unresolved fractions | proxy: nearest player to the detected ball. `ball_frames_frac` says how often the ball was seen |
| `distance` | per team + per roster player, pixels and estimated metres | rough: no pitch homography, so metres assume a 25 m visible pitch width (±50%) |
| `heatmaps` | 32×18 grid per team, image coordinates | good for "where on screen did each team play"; not pitch coordinates |
| `roster` | one entry per stable identity (id, team, frames, crop, jersey number if confident) | see the honesty line |
| `events` | referee-whistle candidates (audio) | candidates, not confirmed events |
| `pipeline` | model / tracker versions, run time | |

## How it works

```
URL ──yt-dlp──▶ 720p source ──cut──▶ clip
  ▶ YOLO11m (persons + ball) ─▶ ByteTrack ─▶ BoT-SORT + OSNet re-ID
  ▶ kit-colour KMeans → teams A / B / O(ther: keeper, referee)
  ▶ identity fusion (part-based colour fingerprint + OSNet, team hard-gate, co-occurrence cannot-link)
  ▶ stats (possession proxy, pan-compensated distance, heatmaps, whistle audio)
  ▶ FIFA-style overlay → H.264
```

All of it is classical CV plus pretrained models. Nothing calls an external API unless you
enable the optional VLM layer below.

Modules (`src/fv/`): `job.py` (the end-to-end runner), `acquire.py`, `detect_track.py`,
`reid_track.py`, `teams.py`, `identity.py`, `stats.py`, `overlay.py`, `events/whistle.py`,
`api.py` (optional), and the research-grade identity stack `jersey.py` / `sr.py` /
`participant.py` / `roster*.py` / `llm.py`.

## Optional: local HTTP API

Only if you want a browser front on your own machine. It queues `fv.job` runs, one at a time on the GPU,
and binds **127.0.0.1 only**:

```bash
uv pip install -e ".[api]"
.venv/bin/python -m fv.api --port 8790
curl -s -XPOST localhost:8790/jobs -H 'content-type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=hhaWMwLlugE","start_s":667,"end_s":727}'
```

Contract (routes, statuses, Range-enabled file serving, `stats.json` schema): [API.md](API.md).

## Optional: vision-LLM roster review

The research identity stack can ask a vision LLM (Gemini, OpenAI, or a local Ollama model) to
review player crops: holistic roster-grid grouping, keeper/role checks, jersey reading. It is
**off by default** (`llm.enabled: false` in `configs/veo.yaml`), and `fv.job` never turns it on.

To use it, you need **your own** `GEMINI_API_KEY` or `OPENAI_API_KEY`:

```bash
cp .env.example .env        # then fill in your key; .env is gitignored
uv pip install -e ".[vlm,sr]"
```

Then set `llm.enabled: true` in `configs/veo.yaml`, or pass `--vlm` to `python -m fv.identity`.
See [RUNBOOK.md](RUNBOOK.md) for the identity experiments and [HANDOVER.md](HANDOVER.md) for
what worked (neutral prompts, whole-grid review, context crops, a geometric co-occurrence veto).

## Demo

Two pre-rendered 60 s results from the test match, a golden-hour **day** slice and a floodlit
**night** slice, are shown on the project page. To reproduce the day slice, run the
`fv.job` command above. For the night slice, use `--start-s 0 --end-s 60`.

## Research notes

This started as a research project on monocular player identity in amateur footage. The
experiment log, design decisions and lessons are in [HANDOVER.md](HANDOVER.md) (links to the
experiment tracker and the dissertation write-up) and [RUNBOOK.md](RUNBOOK.md).

## Licence

MIT, see [LICENSE](LICENSE).
