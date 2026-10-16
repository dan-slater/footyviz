# footyviz-bench: Phase 2 log (ORCH-fv-bench)

Newest last. Commands run from the repo root with `PYTHONPATH=src ~/football-vision/.venv/bin/python -m fv.bench ...`.

## 2026-10-16 — D1: manifest + fetch
- `bench/manifest.json` (schema `fvbench.manifest/1`): 9 matches (5 eval: A1 B1 C1 D1 D2; 4 train: A2 B2 C2 D3), each with eyeball tier, kit tier (clean/bibs/none; train kits pending eyeball except D3), split, normalised teams, venue, source (window/full, fps, `src_t0`), labelled window (900–1020, none shifted) and clip file. A0 dropped (§8.1 #5).
- `python -m fv.bench check`: manifest sanity + team-disjoint rule, exact and fuzzy (shared distinctive word), exit 1 on failure. Passes: 8 eval teams vs 8 train teams. Negative test (a "salahds" variant on D3) fails loudly.
- `bench/fetch.sh`: yt-dlp `--download-sections "*840-1080"` stream copy per match, cookies via `$COOKIES` (never shipped).
- Reference full match symlinked as `raw/A_hhaWMwLlugE.mp4`.
- Box note: `nvidia-smi` reports an NVML driver/library mismatch since the reboot, but torch CUDA works (2.14.1+cu130, `cuda.is_available()` True). So "GPU idle" cannot be checked via nvidia-smi right now.

## 2026-10-16 — D2: `fv.bench measure`
- `src/fv/bench/measure.py`: every 4 s over the labelled window (30 frames), bottom 8 % excluded, full res: luma, RMS contrast, Laplacian var, saturation, Lab a/b, file bitrate. Writes `quality` + `image_tier` into the manifest. Bands: low = luma < 55 or lap < 250; mid = luma < 90; high otherwise. ~10 s for all five.
- Numbers agree with the §0.1 probe within a few % (strip size differs).

| id | eye | kit tier | luma | RMS | lap | sat | Lab a/b | Mb/s | image tier |
|---|---|---|---|---|---|---|---|---|---|
| A1 | A | bibs | 47.0 | 36.5 | 154 | 79.0 | -5.6 / 6.3 | 2.13 | low |
| B1 | B | clean | 36.8 | 25.7 | 100 | 96.5 | -6.1 / 5.1 | 2.32 | low |
| C1 | C | bibs | 58.5 | 37.3 | 833 | 92.1 | -1.4 / -6.3 | 3.98 | mid |
| D1 | D | none | 95.6 | 38.8 | 869 | 79.9 | -7.7 / 10.3 | 4.43 | high |
| D2 | D | clean | 107.1 | 41.0 | 585 | 40.8 | -3.9 / 4.8 | 4.13 | high |

The two axes cross well: image {low: A1 B1, mid: C1, high: D1 D2} × kit {clean: B1 D2, bibs: A1 C1, none: D1}. Train windows not landed yet; re-run `measure` when they do.

## 2026-10-16 — re-timing validated; `fv.bench cut`; data change
- **Data (control plane):** the cookies grant was withdrawn. Only A2 (O4YGl9Ps93A) relayed; B2, C2 and D3 are `status: pending` in the manifest and the harness skips them. A2 measured: image tier **low** (luma 27.7, lap 185). By eye: night, orange bibs vs black-and-white shirts (kit tier bibs), venue V2 (D1's orange clubhouse).
- `src/fv/bench/cut.py`: clips are cut by **frame index** from `src_t0` (nearest frame to 900 s ± ½ frame: D1 → window frame 1923, source 899.9991 s; A1 → source frame 27000), `-frames:v round(120·fps)`, libx264 crf 12 + audio. It records n_frames, src start and pHash (frames 0 / mid / last) in the manifest. D1 + A1 cut, 3596 / 3600 frames.
- **§7 validation on the reference** (stream-copy 840–1080 from the full file, re-time, recut, per-frame pHash vs A1 cut direct):
  - copy *with* an edit list (ffmpeg 7 default): src_t0 = 840.0, 3600/3600 frames, pHash distance **0** on every frame ✓.
  - copy *without* an edit list (`-use_editlist 0`, like the relay): frame 0 is the I-frame at 837.7, but two open-GOP leading B-frames are dropped, which leaves a **pts gap after frame 0**. Index re-timing is then off by **2 frames** (mean pHash 3.1; at offset +2 it is 0.1).
  - **The Rookie eval windows do not have this:** frame 0 is the I-frame at pts 0.056 and pts are uniform (checked B1 C1 D1 D2), so the 835.835 s / frame-1923 derivation holds.
  - `cut.window_check` now enforces it: frame 0 must be a keyframe and there must be no pts gaps, otherwise an eval cut is refused.
- **A2 relay fails the check**: it starts mid-GOP, with P/B frames up to the first keyframe at window index 25 (pts 0.834). It has 7,319 frames like the others, so 835.835 is likely, but it is `src_t0_verified: false`. It is a train clip, so this is not blocking; cut warns.

## 2026-10-16 — D3: pre-labeller (`fv.bench prelabel`)
- `src/fv/bench/prelabel.py` (not a leaderboard config, per the anti-bias rule):
  - yolo11x @ 1280, conf 0.10, every frame (`fv.detect_track`, ByteTrack);
  - BoT-SORT + OSNet tracklets (`fv.reid_track`);
  - kit k-means teams + CV identity fusion (`fv.teams`, `fv.identity`) → proposed people;
  - RT-DETR-x on keyframes, union where IoU < 0.5.
- Keyframes every 90 frames (40/clip). 3 blind keyframes per clip come from a seeded RNG and are exported with no boxes.
- Weights: `~/footyviz-bench-data/models/{yolo11x,rtdetr-x}.pt` (downloaded via ultralytics). Worktree symlinks `yolo11m.pt` and `data/models` → main checkout (both gitignored).
- Output:
  - `~/footyviz-bench-data/prelabels/<id>/` holds the parquet intermediates;
  - `~/footyviz-bench-data/labels/<id>/` holds `frames/kf_*.jpg`, `crops/t<track>.jpg` (one exemplar per tracklet) and `prelabel.json` (keyframe boxes with track/group/team guess, tracklets, groups, 1 Hz ball points for L2 later).

| clip | boxes on 37 visible kf | tracklets | proposed people | blind kf | wall time |
|---|---|---|---|---|---|
| D1 | 386 | 94 | 29 | 270, 720, 2700 | 2.4 min (detect 67 s, reid 38 s, teams+id 21 s, RT-DETR 10 s) |
| A1 | 287 | 116 | 32 | 90, 1440, 2430 | ~2.2 min |

## 2026-10-16 — D4: labelling page (`fv.bench label`)
- Runs in tmux **`footyviz-label`** on **127.0.0.1:5191**. From the Mac: `ssh -L 5191:127.0.0.1:5191 zulzi-gpu`, then open http://localhost:5191/#D1. The server log is `~/footyviz-bench-data/label-server.log`.
- `src/fv/bench/label.py` (stdlib `http.server`, no build step) + `label_static/index.html` (plain JS). On first open, proposed groups that appear on keyframes are seeded as P01…Pn, with the k-means team guess mapped A → T1, B → T2, O → none. The page autosaves `labels/<id>/work.json`. The server derives `clip.json` (fvbench.clip/1) + `keyframes.json` (fvbench.keyframes/1) on every save (`src/fv/bench/labels.py`).
- **Gallery (L0)**: one card per person, holding that person's tracklet exemplar crops. Hover a crop for a large preview.
  - Select: click (shift/ctrl adds).
  - Move: drag a crop onto a card, or `M` (move to the focused card), `N` (move to a new person), `X` (move to ignore).
  - Edit the person: `1`/`2`/`0` sets the team, `E` keeper, `F` referee, `J` jersey.
- **Keyframes (L1/L1b)**: the full-res frame with boxes, a 2× loupe, and the people list on the right.
  - Select and adjust: `Tab` (or click) selects a box; drag a corner to resize; `A`+drag adds a box.
  - Assign: type digits + `Enter` to set the P-id (`Shift` applies it to the whole tracklet), or click a person in the list. `N` makes a new person.
  - Mark: `I` ignore (press again to cycle off_pitch/ghost/tiny/truncated), `U` unsure (press again to cycle team), `X` deletes a false box.
  - `Space`/`Enter` marks the keyframe done and jumps to the next undone one. `←→` navigate.
  - Blind keyframes are flagged in purple and show no pre-label boxes.
- Active seconds are logged per view and per keyframe (input in the last 60 s), which turns the §3 cost table into measurements.
- Checked headless with Chromium (playwright-core): gallery + keyframe flows, autosave, derive; no console errors. The test state was wiped afterwards.
- Seen on D1: the proposed people include bench subs sitting by the orange wall (they become ignore in the gallery). The k-means "O" cluster is 9 people with no team, which is expected on a no-kit clip.

## 2026-10-16 — D5: `fv.bench run` / `score` / `export-labels`; first CV batch
- `src/fv/bench/run.py`: method yaml (`bench/methods/`, `inherits:` shallow per-stage override) → stage adapters over `fv.detect_track` / `fv.reid_track` / `fv.teams` / `fv.identity` (unmodified) → `runs/<method>@<batch>/<clip>/r<k>/{dets,ball}.parquet + meta.json` and `run.json` (commit, config + hash, versions, GPU). Methods: `b0-cv-job`, `b4-bytetrack`, `b4-botsort`, `b3-yolo11x-1280` (not run yet).
- `src/fv/bench/score.py` scores done keyframes only:
  - T1: greedy IoU 0.5 with an ignore class; R, P, AP50, AP75, recall by height bucket.
  - T3: Hungarian team permutation reused downstream; outfield accuracy, keeper accuracy (O or own team counts), false-participant rate.
  - T4: all-pairs P/R/F1 over cross-keyframe pairs (closed-form counts), keyframe IDF1, identity count error per team with ≥ 2 s phantoms.
  - Grouping per clip / image tier / kit tier / overall. `--write-results` appends a block to `bench/RESULTS.md` with run, date, commit, config and label version.
- **Scorer self-test** (synthetic GT built from the D1 pre-labels):
  - identical predictions → every metric 1.0;
  - over-split (raw track ids) → pair P 1.0, pair R 0.345, kf IDF1 0.565, count error 14.5;
  - swapped team letters → 1.0, so the permutation is not penalised.
- `fv.bench export-labels`: derived `clip.json` / `keyframes.json` → `bench/labels/<id>/` + `VERSION` (JSON only, for git).
- **Batch `2026-10-16-base`** (deterministic, ×1, all 5 eval clips, no errors): B0 81–88 s/clip (detect 23, reid 38–41, teams 3–5, identity 15–19), ByteTrack-only 26–28 s, BoT-SORT-only 64–68 s. These are scored as soon as labels exist. The table is in RESULTS.md.

## 2026-10-16 — D6: VLM baseline, scoped, not run
- The run-11 control = `fv.gk_lock --control` and run 13 = `fv.gk_lock`. Both take path flags (`--reid --cache --teams --jersey --vlm-fp --context-dir`) but default to night-slice files.
- A per-clip adapter needs four steps per clip and repeat: `fv.jersey` → `identity.vlm_fingerprints` (VLM) → `roster_v3.render_context` → `gk_lock`. That's about an hour of wiring.
- Deferred because it is only worth running (and spending on) once the target clip is labelled. Open question to Daniel: does "B1" in "VLM N=3 on B1 only" mean clip B1 (mzcJ8, not in the D1+A1 label pass) or baseline row B1 (= run 11)?
- Spend so far: **$0** (no VLM calls; the Gemini key has not been loaded).
