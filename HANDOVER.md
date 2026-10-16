# footyviz (football-vision) — HANDOVER / STATE

Living state doc. Pairs with `RUNBOOK.md` (how to run) and the dossier artifact
(https://claude.ai/code/artifact/4a561329-21be-4376-acb9-2c92c80756d7).
Last updated: 2026-10-09.

> ## ⚽ footyviz: public library + site, exp 13 done, benchmark designed (2026-10-16) — READ THIS FIRST
>
> **Goal:** football-vision is now **footyviz**, a public open-source library with a demo site. The current thread is **footyviz-bench**: a bad-to-good benchmark of amateur Veo clips to evaluate approaches, swap AI models, and later fine-tune our own.
>
> **⚠️ UNCOMMITTED:** only this block in `HANDOVER.md` (main, zulzi). No code WIP anywhere. Commit it with the next real change.
>
> **⚠️ zulzi rebooted ~2026-10-14. ALL tmux sessions are gone:** ORCH-fv-pipeline, ORCH-fv-web, ORCH-fv-bench, `footyviz-api` (:8790), `footyviz-web-dev` (:5190), and the surf-lab ORCH-mapkit/ORCH-sim/surflab-dev. Committed work survives. Transcripts are in `~/.claude/projects/-home-danielslater-{football-vision,football-vision-bench,code-footyviz-web}/*.jsonl`. Respawn lines fresh; never assume a line is alive.
>
> **What shipped (all pushed; GitHub = zulzi):**
> - `dan-slater/footyviz` **main ac779d7**: `fv.job` CLI runner, `fv.stats`, optional local `fv.api`, `fv.gk_lock` (exp 13), README as the product page, API.md stats.json schema v1. MIT.
> - `dan-slater/footyviz-web` **main 162de99**: SvelteKit static showcase (dusk + night demos, heatmaps, roster, honesty line, "Run it on your match"). LIVE at **https://footyviz.danielslater.dev** (CF Pages project `footyviz`, direct-upload from the Mac; `bun run build` then `bunx wrangler pages deploy build --project-name=footyviz --branch=main` with CLOUDFLARE_API_KEY/EMAIL legacy auth, account 084af6c1…).
> - `dan-slater/footyviz` **bench 9ae3887** (worktree `~/football-vision-bench`): `bench/BENCHMARK.md` design v0.2 + probes of all four eval windows. Design only, no harness code yet.
> - **Exp 13 (GK-role lock):** keeper track 14 now has its own A-GK row instead of being merged with outfield players. Counts: exp 13 = 40, same-day control = 43, July run 11 = 24. **VLM run-to-run variance dominates identity counts.** Always compare against a same-day control. Follow-up: track 12 (probably the keeper's second tracklet) failed the motion gate (3.9 vs 3.25); loosen the gate and re-run.
>
> **Product decisions (Daniel, 2026-10-09):** no public jobs. The site is a demo showcase and people run the library themselves. The VLM layer stays in the repo, off by default, using the user's own key in a gitignored `.env`.
>
> **Benchmark data:**
> - Source: YouTube channel **Rookie (@TRG_Rookie)**, 1,782 Cape Town Veo uploads. Thumbnail triage of all of them is in `~/footyviz-bench-data/thumb_scores.csv`.
> - Eval windows (source 840–1080 s, stream-copied) are in `~/footyviz-bench-data/windows/`: D1 c6IV1mQ3GjM, D2 gjd12dkMNNI, B1 mzcJ8_wyOaw, C1 GQfni7JdYvY. A1 = the reference hhaWMwLlugE, already local.
> - **Train matches NOT fetched:** A2 O4YGl9Ps93A, B2 vS4NP3wkBIA, C2 2u32W0O7ZfQ, D3 YwkdvL6Un3g. D3 was chosen by eye: clear day, blue v black kits, no shared team with eval.
>
> **Blockers:**
> 1. **YouTube bot-walls zulzi AND (since 2026-10-09) the Mac** after about a dozen downloads. Fix = `yt-dlp --cookies-from-browser <browser>` on the Mac. Daniel's call, not yet given. Cookies stay on the Mac; never ship them to zulzi.
> 2. The Mac→zulzi copy runs at only about 0.25–0.75 MB/s and the tailnet drops mid-copy. So copy 4-minute windows, not full matches. Verify size and retry.
>
> **Open decisions awaiting Daniel (BENCHMARK.md §8, 7 items):**
> 1. The metric set.
> 2. The label budget: 3.2 h now and 2.7 h dense later, OR D1 + A1 first (about 70 min), which is recommended.
> 3. The team-disjoint split rule.
> 4. The VLM budget ($30–60) and N = 3 repeats.
> 5. Whether to label the A0 slice.
> 6. One window per eval match, and whether whole-match runs go on the board.
> 7. Tiers on two measured axes (image quality + kit clarity), or keep the eyeball tiers.
>
> Also still open: the cookies call; whether to ask the Rookie channel before publishing benchmark-derived material (the raw footage stays private; the public repo gets only ids, timestamps and labels).
>
> **Gotchas:**
> - The Rookie channel mixes in rugby, padel and school 11-a-side festivals (minors), all excluded. Its titles carry team names and the final score.
> - The "day" demo slice of the reference match is really dusk.
> - The lines' "Usage limit reached" means re-login with the other account (see the tmux-orchestrator skill). zulzi `claude` was last logged in as nia@nsuna.com.
> - The idle `❯` ghost text in a line is an auto-suggestion, never an approval.
>
> **Next actions:**
> 1. Get Daniel's §8 answers and the cookies call.
> 2. Fetch + relay the train windows (or full matches overnight).
> 3. Respawn ORCH-fv-bench on `~/football-vision-bench` (inline-only brief, commit on `bench`, never push) for Phase 2: manifest + fetch script, pre-labeller, labelling page on 127.0.0.1:5191 (tmux `footyviz-label`), `fv.bench run/score`, baselines.
> 4. Exp 13 follow-up (motion gate).
> 5. Check the site's video playback on a real phone.
>
> **Pointers:** `bench/BENCHMARK.md` (bench branch), the 2026-10-09 block below, RUNBOOK.md "Job runner" + "Job API", API.md, `~/dan-hub/AGENTS-BOARD.md` (footyviz entries), Mac memory `footyviz-project.md`.

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

**Exp 13 (GK-role pre-pass): DONE 2026-10-09** (`python -m fv.gk_lock`, VARIANTS run 13).
- Pre-pass: neutral role question on each track's 3 context views + pan-cancelled relative-motion
  gate (≤ slice median 3.25 px/frame). VLM voted keeper for 12 and 14 (both in-goal, 0.99); gate kept 14 only.
- **Keeper left the tangle:** run 11 A-4 = {10,12,14} → same-day control A-6 = {14,48} → exp 13 **A-GK = {14}**.
- Counts: **exp 13 = 40, same-day control = 43** (`--control`, identical code minus the pre-pass), July run 11 = 24.
  **The control does not reproduce 24**: same inputs (333 vetoed pairs match the July commit), but the VLM now
  rejects most merges and every team hits the 40-call cap. Identity counts are dominated by VLM run-to-run
  variance; compare variants against a same-day control. The B/O differences (17→13, 4→5) are that noise; the
  pre-pass only changes team A (22 = 22, but 14 is now its own GK row instead of merged with outfield 48).
- Follow-up: track 12 ("diving to defend", 0.99) failed the motion gate (3.9) and is probably the keeper's second
  tracklet. Loosen the gate (e.g. ≤ 1.5× median, or skip it at conf ≥ 0.95), then re-run. Also raise max_calls or
  log rejected-merge reasons to see why acceptance dropped since July.
- Outputs (data/, not committed): `tracks/identities_{gklock,ctrl11}_night{,_table}.parquet` + `.json` decision
  logs; `out/ident_gklock_fifa_h264.mp4`, `out/ident_gklock_frame{0200,1100}.jpg`.

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
