# footyviz-bench results

Written by `python -m fv.bench score --run <run> --write-results`. Metric definitions: BENCHMARK.md §2.
Each row group: per clip, per image tier, per kit tier, overall (mean over clips, equal weight); ± is sd over repeats.
Every block records run name, date, commit, config hash + config, label version and repeats.

**No scores yet:** no keyframes labelled (D1 + A1 are first, §8.1 #2). Runs already on the box, ready to score:

| run | date | commit | config | repeats | clips | wall s / clip |
|---|---|---|---|---|---|---|
| b0-cv-job@2026-10-16-base | 2026-10-16 | 0c85aaf | yolo11m@640 c0.30, ByteTrack → BoT-SORT/OSNet, kmeans teams, cv_fusion 0.42 | 1 | A1 B1 C1 D1 D2 | 81–88 |
| b4-bytetrack@2026-10-16-base | 2026-10-16 | 0c85aaf | yolo11m@640 c0.30, ByteTrack only, identity = track id | 1 | A1 B1 C1 D1 D2 | 26–28 |
| b4-botsort@2026-10-16-base | 2026-10-16 | 0c85aaf | yolo11m@640 c0.30, BoT-SORT/OSNet, identity = track id | 1 | A1 B1 C1 D1 D2 | 64–68 |
| b3-yolo11x-1280@2026-10-16-base | 2026-10-17 | cc77f88 | B0 with yolo11x@1280 | 1 | A1 B1 C1 D1 D2 | 125–130 |

Unlabelled sanity (predicted identities per clip, A/B/O): B0 A1 35 (17/10/8), B1 23 (16/6/1), C1 41 (12/19/10), D1 24 (12/9/3), D2 25 (15/8/2). B3 (yolo11x@1280): A1 30, B1 25, C1 38, D1 31, D2 31. BoT-SORT alone: 84–121 tracks per clip; ByteTrack alone: 179–220.

VLM rows (b1-run11 control, b2-run13) are wired but **not run**: B1 is unlabelled and the spend waits for Daniel's go after the D1 + A1 pass.
