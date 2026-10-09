# footyviz (football-vision) — HANDOVER / STATE

Living state doc. Pairs with `RUNBOOK.md` (how to run) and the dossier artifact
(https://claude.ai/code/artifact/4a561329-21be-4376-acb9-2c92c80756d7).
Last updated: 2026-10-09.

## 2026-10-09 — footyviz: open-source product + runner (ORCH-fv-pipeline line)
**Product decision (Daniel):** no public jobs. The repo (github.com/dan-slater/footyviz, MIT) is the
product; people run it on their own box with their own keys. The web page (sibling line
ORCH-fv-web, `~/code/footyviz-web`) is a demo show-off page built against `API.md` + the demo drop.

**Built (all committed on main, not pushed):**
- `API.md`: contract v1 incl. the full `stats.json` schema (the web line mocks against it).
- `fv.job`: the main CLI. URL or local file + window → `overlay.mp4` (H.264, honesty footer),
  `poster.jpg`, `roster.jpg`, `crop_*.jpg`, `stats.json`; prints progress, writes `progress.json`.
- `fv.stats`: possession proxy (nearest player to YOLO "sports ball"), pan-compensated distance
  (phase correlation; metres from an assumed 25 m visible width), 32×18 team heatmaps, roster.
- `fv.detect_track`: ball (COCO 32) detected in the same pass → `<tracks>_ball.parquet` (untracked).
- `fv.overlay`: H.264 writer (capped 2.2 Mb/s), footer, poster, cleaner label fallback.
- `fv.api`: optional local queue (127.0.0.1:8790, tmux `footyviz-api`) with Range serving.
- `.env` (gitignored) + `.env.example` for keys; the borrowed restaurant-portal key pointer is gone.
- README rewritten as the product page ("Run it on your match"); project renamed footyviz 0.2.0.

**Proven:** venv rebuilt (torch 2.14 cu130, 4090 OK); detect_track smoke (300 frames, 8.7 s);
`fv.job` dusk slice (f20000–21800) 21 identities, night slice (f0–1800) 27 identities, ~55 s
each, overlay ~8.9 MB each → `~/orch-scratch/footyviz-demo/{day,night}/`; API: health, POST,
poll → done, stats.json, Range 206, CORS, traversal 404, `/etc/passwd` rejected.

**Open / caveats:**
- The match has **no real daylight**: the "day" demo is dusk. Decision 1 (daytime test) can't be
  answered from this video; it needs a second match.
- YouTube bot-walls zulzi's IP → new URLs need `--cookies`; `source.title` falls back to the video id.
- Team clustering still lets spectators / ref into A/B on some clips (night roster B-8/B-13).
- Possession is a proxy: COCO ball detector hits 63–76% of frames incl. some false positives.
- The CV-only identity in `fv.job` (no VLM) gives 21/27, not the VLM-reviewed ~23 reference.

**Exp 13 (GK-role pre-pass): IN PROGRESS, not yet run.** Key is in `.env`; `gemini-3.1-pro-preview` answers.
Reconstructed run-11 baseline (the July driver is lost; `roster_v3` has no CLI):
- Track set = the 57 tracks in `data/tracks/context_night/` (= `identities_neutral_night.parquet`
  rows) and includes spectators 19 and 28. Run 11 = 24 identities. **Keeper track 14 is merged into A-4
  with outfield tracks 10 and 12**, which is the tangle to fix.
- `data/out/participant_classification.csv` already labels 14 = GK (tight-crop VLM, "gloves");
  track 16 (O) also got a "GOALKEEPER" reply that was discarded as unparseable. `motion` column: 14 = 62.6 px
  vs hundreds for outfield (geometry signal).
- Plan: driver = `roster.load_context(identities_night_desc.npz, teams_reid_night, jersey_night_0_1800,
  identities_vlm_night_vlmfp)` + `roster_v3.cooccur_veto(reid_night_0_1800)` + views loaded from
  `context_night/t{tid}_{k}.jpg` → **pre-pass**: low-motion candidates + neutral VLM role question on
  the context views ("where is the boxed person standing?") → lock GK(s), exclude them, then
  `roster_v3.cluster_team` per team (emb_gate 0.55, min_conf 0.6, max_calls 40), GK = own `<team>-GK`
  identity, `roster._name_clusters` + `_emit` → `identities_gklock_night.parquet`. Also re-run a
  same-day control without the pre-pass (VLM is stochastic). Then add VARIANTS run 13 in build_exp_tracker.py.

## 2026-07-02 SESSION UPDATE — dashboards, PhD report, pivot back to experiments
**Two living HTML docs (self-contained claude.ai Artifacts, regenerable):**
- **Experiment tracker** (PRIMARY / current focus): https://claude.ai/code/artifact/220dbdc8-10a9-4d8a-aa00-86d65f7aa6e1
  — one tab per identity run (setup / result / notes) + an Overview chart. Generator:
  `zulzi:~/football-vision/build_exp_tracker.py` (commit 4df4673). Regenerate: run it →
  pull `data/out/dashboard_experiments.html` → redeploy the Artifact. Add a run = add a `VARIANTS` entry.
- **Review dashboard** (earlier, per-stage view): https://claude.ai/code/artifact/9b09eb2c-1f14-4e83-b9c0-4d055ceb6be8
  — generator `build_dashboard.py`.

**PhD dissertation report (PAUSED — pivoted back to experiments):** lives on the **Mac** at
`/Users/ds/football-vision-report` (git). Dissertation-style; compiles to 53 pp via local MacTeX
(`latexmk -pdf -bibtex main.tex`; Mac also has tectonic). WRITTEN: Ch1 Introduction, Ch9 Evaluation/Results
(+2 plots), Appendix A Config/Repro, Appendix B VLM Prompts, `references.bib` (96 verified entries),
related-work notes. Ch2–8, 10–11 are one-page stubs. Read-link (typeset pages):
https://claude.ai/code/artifact/6315e372-0d53-40ea-b651-ea4968c8b515. Report ground-truth + per-chapter
briefs in its `PLAN.md`; live board in `STATE.md`; real implementation reference in `notes/impl/`
(`src/fv/*.py`, `veo.yaml`, versions) + result CSVs in `data/`.

**Orchestration infra (this session):** report/writing work runs as **Mac-local factory lines** via
`~/orch-mac-lines/macline.sh` (local tmux + fence, NO ssh/sudo) editing the Mac repo; **zulzi-gpu is for
experiments only** (remote-tmux-orchestrator `factory.sh` drives zulzi lines). The Mac + zulzi are shared
with other control planes — scope kills to `ORCH-fvr-*`, never touch others.

**NEXT (pivot):** back to experimentation — run new identity-pipeline variants on zulzi (change the setup,
e.g. the GK-role pre-pass for the double-A-4/keeper bug; the daytime-slice test), add each as a tab to the
experiment tracker, iterate. Tools/data unchanged from below.

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
| Scoreboard OCR | ⛔ ruled out (pending a timestamp) | top-center lit board is most likely a **drinks advert**, not a score — ~140 px even at sharpest pan, bloomed, no digits recoverable under the bloom. Investigated hard on the `ORCH-sb` line; flips only if a frame shows two clearly-different digits |
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
3. **Fix the double-A-4 / keeper labeling bug** — small, clean. *Concrete approach (from the fv line):* **lock the GK role first** (keeper = defined by standing-in-goal geometry, via context crops), then **exclude that keeper track from the outfield grey-hoodie merges**, and only *then* run neutral grouping — i.e. run the GK-role lock as a pre-pass before grouping so the goalkeeper (track 14) can't tangle into a duplicate outfield row.
4. **Own Gemini key** — *done in code 2026-10-09:* keys come only from the repo's gitignored `.env` (`.env.example`). The repo has no key yet; the VLM path needs one put there.
5. **Scoreboard** — *investigated & ruled out (2026-07-02).* The top-center lit board is most likely a **drinks advert**, not a scoreboard: ~140 px even at the sharpest pan, bloomed, and registration/temporal-averaging recovers no digits under the bloom (there's nothing there to read). Only reopen if you point at a specific timestamp where you can personally see two clearly-different digits (e.g. a "2" or "3") — then I zoom straight to it. Evidence crops on zulzi `data/out/`: `sb3_panel_compare.jpg` (registered straddle test), `sb3_montage.jpg` (top-12 sharpest), `sb3_best_00_f720_t1107.jpg` (crispest), `sb3_panel_*.jpg`, `sb3_digits_*.jpg`. No code changed for this.
6. **~~Reseed the ORCH-fv factory line~~** — *done 2026-07-02:* the heavy `ORCH-fv` (~647 k) and `ORCH-sb` (~248 k) lines were retired. Spin a **fresh** line on demand (see Infra recap); don't revive the old ones.

## Infra recap
- Repo + data: `zulzi-gpu:~/football-vision` (canonical, git HEAD `e064dd6` = docs; identity work is `a467ef7`). Untracked in root: `draw_pair.py` (a stray, not part of the pipeline).
- GPU: single RTX 4090. venv `.venv` (uv, py3.12).
- Access: `ssh zulzi-gpu`; fallback hop `ssh devboxv2 'ssh zulzi-gpu …'`.
- Driven via a `remote-tmux-orchestrator` factory line. **The old `ORCH-fv`/`ORCH-sb` lines were retired 2026-07-02** — spin a fresh line when you need one (`factory.sh` / the skill), don't revive the heavy ones. (`metro` on zulzi belongs to a different project — leave it alone.)
- Gemini `gemini-3.1-pro-preview`, key from this repo's gitignored `.env` (none set as of 2026-10-09).
- Dossier artifact source: `/Users/ds/football-vision/system-overview.html`.
