# football-vision — HANDOVER / STATE

Living state doc. Pairs with `RUNBOOK.md` (how to run) and the dossier artifact
(https://claude.ai/code/artifact/4a561329-21be-4376-acb9-2c92c80756d7).
Last updated: 2026-07-02.

## What this is
A monocular sports-CV pipeline that pulls an **amateur Veo football match off YouTube** and
produces tracked players, team labels, whistle-based events, and a broadcast-style overlay — with
an optional, toggleable **vision-LLM** reasoning layer. Test video: a 39.5-min Cape Town 5-a-side,
`https://www.youtube.com/watch?v=hhaWMwLlugE` (starts daylight, ends floodlit night).

## Status at a glance
| Capability | State | Notes |
|---|---|---|
| Acquire (yt-dlp) | ✅ done | 720p + 1080p (YouTube ceiling) + audio |
| Detect + track (YOLO11 + ByteTrack/BoT-SORT) | ✅ done | ~120 fps, 488k dets, 98.6% frames |
| Team assignment (A white / B pink / O blue+GK) | ✅ done | holds day & night |
| Whistle events | ✅ done | 12 blasts, 2nd half, some goal-pairs → `whistles.csv` |
| FIFA overlay (chevrons, no boxes/traces) | ✅ done | `overlay.py --style fifa` |
| Vision-LLM layer (Gemini/OpenAI/local, toggle) | ✅ done | off by default; Gemini validated live |
| Jersey-number OCR | 🟡 partial | reads where legible (#6/#7/#8/#11/#26, "VDM 6", "Elliot 7") |
| **Player identity** | 🔴 at ceiling | ~23–24 stable vs true **~14–16** on the night slice |
| Scoreboard OCR | 🔴 unresolved | physical board top-center IS there, too low-res/bloomed |
| Goals & assists | ⬜ designed | whistle + audio-spike + ball-near-goal path, not built |
| Stats (possession/heatmap/distance) | ⬜ next up | doesn't need perfect IDs |

## The identity story (most of the effort)
Collapse thousands of churning track IDs → a stable per-player roster. On the night slice
(55–61 raw tracklets), best result is **~23–24 stable identities** vs a true **~14–16**
(5-a-side ≈ 5–8/team + keepers). Progression (non-monotonic on purpose): CV-only 19 (over-merged)
→ VLM variants 25–27 → +participant filter 21 → 1080p+reprompt 24 → +super-res 23 → +number-anchor
fix 23 → neutral+context+veto 24.

**The count is capped by the footage, not the method:** dark, floodlit, near-identical kits,
players in cold-weather hoodies, ~35 px figures, heavy auto-pan. The methodology is sound and built.

### Techniques that work / bugs fixed
- **Co-occurrence veto** (kept): two tracklets on-screen together & >100 px apart cannot be one
  person → hard-blocks impossible merges. (Proved track 6 ≠ track 20: 317 px apart, 464 shared frames.)
- **Confident-number anchoring** (fixed bug): a low-confidence "1" (partial "11") was hard-splitting
  one player into A#1 + A#11. Fix: only `confident=True` numbers anchor; single digits are
  prefix-compatible with multi-digit. A#1+A#11 now merge. **Check `jersey_*.parquet` col `confident`.**
- **Holistic grid review > pairwise**: hand the VLM the whole team roster grid at once → it spots
  duplicates/roles a pairwise sweep misses. Verified: 44+20 merged, track 6 unbundled.
- **Neutral prompting + full-frame context crops** (built into `roster_v3`): earlier prompts LED
  the model ("hoodies = subs") → it confirmed the wrong hypothesis. **Grey hoodies ARE players**
  (cold night). Ask neutrally; feed geometry as fact; give it boxed full-frame views (a keeper is
  defined by standing in goal — invisible in a tight crop).
- **Real super-res**: use **spandrel** to load RealESRGAN_x4plus (NOT basicsr — broken). Sharper
  crops proved clarity was no longer the limiter.

### Known remaining bugs / gaps
- **Double `A-4` / keeper mislabel**: the goalkeeper (track 14) tangles into a duplicate row — clean defect to fix.
- Grey-hoodie players still over/under-split (near-identical → genuinely hard).
- `O` bucket (blue kit / GK / misc) inflates the count; team assignment strains on the 3rd-kit/keeper.

## Lessons banked
1. Don't lead the VLM — neutral prompts, evidence decides.
2. Holistic (whole-grid) review beats pairwise comparison.
3. Context crops (boxed full frame) for role/position; tight crops for the OSNet embedding.
4. Geometry (can't-be-two-places) is the reliable arbiter when appearance is ambiguous.
5. Super-res = spandrel, not basicsr.
6. Lower ID count isn't always better — a correct 23 beats an over-merged 19.
7. GPU is decode-bound at 120 fps (47% util) — headroom is NVDEC/batching/TensorRT-FP8, not a bigger GPU.

## Open decisions / next steps (pick one to continue)
1. **Daytime slice test** — *highest value.* Sharper, kits/numbers legible; tests whether ~23 is the clip or the method.
2. **Move to stats** — possession / heatmaps / distance; team-level ready without perfect IDs.
3. **Fix the double-A-4 / keeper labeling bug** — small, clean.
4. **Own Gemini key** — currently borrowing `~/projectx-restaurant-portal/.env`; give the project its own gitignored key. (User was checking which GCP project the key belongs to — "sorted separately".)
5. **Scoreboard** — user to point at a timestamp where the top-center lit board's digits are readable; then crop+OCR/Gemini there.
6. **Reseed the ORCH-fv factory line** — it's ~600 k tokens, very heavy.

## Infra recap
- Repo + data: `zulzi-gpu:~/football-vision` (canonical, git HEAD `a467ef7`).
- GPU: single RTX 4090. venv `.venv` (uv, py3.12).
- Access: `ssh zulzi-gpu`; fallback hop `ssh devboxv2 'ssh zulzi-gpu …'`.
- Driven via `ORCH-fv` factory line (remote-tmux-orchestrator skill).
- Gemini `gemini-3.1-pro-preview`, key from restaurant-portal `.env`.
- Dossier artifact source: `/Users/ds/football-vision/system-overview.html`.
