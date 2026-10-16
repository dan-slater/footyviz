# footyviz-bench: Phase 2 log (ORCH-fv-bench)

Newest last. Commands run from the repo root with `PYTHONPATH=src ~/football-vision/.venv/bin/python -m fv.bench ...`.

## 2026-10-16 — D1: manifest + fetch
- `bench/manifest.json` (schema `fvbench.manifest/1`): 9 matches (5 eval: A1 B1 C1 D1 D2; 4 train: A2 B2 C2 D3), each with eyeball tier, kit tier (clean/bibs/none; train kits pending eyeball except D3), split, normalised teams, venue, source (window/full, fps, `src_t0`), labelled window (900–1020, none shifted) and clip file. A0 dropped (§8.1 #5).
- `python -m fv.bench check`: manifest sanity + team-disjoint rule, exact and fuzzy (shared distinctive word), exit 1 on failure. Passes: 8 eval teams vs 8 train teams. Negative test (a "salahds" variant on D3) fails loudly.
- `bench/fetch.sh`: yt-dlp `--download-sections "*840-1080"` stream copy per match, cookies via `$COOKIES` (never shipped).
- Reference full match symlinked as `raw/A_hhaWMwLlugE.mp4`.
- Box note: `nvidia-smi` reports an NVML driver/library mismatch since the reboot, but torch CUDA works (2.14.1+cu130, `cuda.is_available()` True). So "GPU idle" cannot be checked via nvidia-smi right now.
