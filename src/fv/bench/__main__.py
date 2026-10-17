"""footyviz-bench CLI (run from the repo root with PYTHONPATH=src).

    python -m fv.bench check                       # team-disjoint split + manifest sanity
    python -m fv.bench measure [--clip D1 ...]     # image-tier stats -> manifest
    python -m fv.bench cut [--clip D1 ...]         # 120 s labelled clips from windows/full sources
    python -m fv.bench prelabel --clip D1 A1       # keyframe crops + proposed person IDs
    python -m fv.bench label                       # labelling page on 127.0.0.1:5191
    python -m fv.bench export-labels               # derived GT json -> bench/labels/ (git)
    python -m fv.bench run   --method bench/methods/<m>.yaml --batch <tag> [--clip ..] [--repeats N]
    python -m fv.bench score --run <run-name> [--write-results]
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fv.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="manifest sanity + team-disjoint splits (exit 1 on failure)")
    p = sub.add_parser("measure", help="image-tier measurements -> manifest")
    p.add_argument("--clip", nargs="*")
    p.add_argument("--no-write", action="store_true")
    p = sub.add_parser("cut", help="cut labelled 120 s clips")
    p.add_argument("--clip", nargs="*")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("prelabel", help="keyframe crops + proposed IDs for labelling")
    p.add_argument("--clip", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("label", help="serve the labelling page")
    p.add_argument("--port", type=int, default=5191)
    p = sub.add_parser("export-labels", help="copy derived GT json into bench/labels/ for git")
    p.add_argument("--clip", nargs="*")
    p = sub.add_parser("run", help="run a method on eval clips")
    p.add_argument("--method", required=True)
    p.add_argument("--batch", required=True, help="batch tag, e.g. 2026-10-16-base")
    p.add_argument("--clip", nargs="*")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-spend", action="store_true", help="permit paid VLM calls (stochastic methods)")
    p.add_argument("--dry-vlm", action="store_true", help="fake VLM client, $0: wiring test only")
    p = sub.add_parser("score", help="score a run against labels")
    p.add_argument("--run", required=True, nargs="+", help="run name(s) under runs/")
    p.add_argument("--write-results", action="store_true", help="append to bench/RESULTS.md")
    p.add_argument("--labels", default=None, help="labels dir (default ~/footyviz-bench-data/labels)")
    a = ap.parse_args(argv)

    if a.cmd == "check":
        from . import manifest
        return manifest.check()
    if a.cmd == "measure":
        from . import measure
        return measure.main(a.clip, write=not a.no_write)
    if a.cmd == "cut":
        from . import cut
        return cut.main(a.clip, force=a.force)
    if a.cmd == "prelabel":
        from . import prelabel
        return prelabel.main(a.clip, force=a.force)
    if a.cmd == "label":
        from . import label
        return label.main(a.port)
    if a.cmd == "export-labels":
        from . import labels
        return labels.export(a.clip)
    if a.cmd == "run":
        from . import run
        return run.main(a.method, a.batch, a.clip, a.repeats, force=a.force,
                        allow_spend=a.allow_spend, dry_vlm=a.dry_vlm)
    if a.cmd == "score":
        from . import score
        from pathlib import Path
        return score.main(a.run, write=a.write_results,
                          labels=Path(a.labels).expanduser() if a.labels else score.LABELS)
    return 2


if __name__ == "__main__":
    sys.exit(main())
