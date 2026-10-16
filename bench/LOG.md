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
