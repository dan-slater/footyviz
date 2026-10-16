"""bench/manifest.json: load, save, resolve paths, team-disjoint check."""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "bench" / "manifest.json"
DATA = Path("~/footyviz-bench-data").expanduser()

# Words too generic to flag a shared team on their own.
_GENERIC = {"fc", "utd", "united", "the", "city", "club", "athletic", "kids", "sc", "afc",
            "real", "inter", "sporting", "rovers", "town"}


def load(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text())


def save(m: dict, path: Path = MANIFEST) -> None:
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")


def match(m: dict, mid: str) -> dict:
    for x in m["matches"]:
        if x["id"] == mid:
            return x
    raise KeyError(f"no match {mid!r} in manifest")


def matches(m: dict, split: str | None = None, ids: list[str] | None = None) -> list[dict]:
    out = [x for x in m["matches"] if split in (None, x["split"])]
    if ids:
        out = [x for x in out if x["id"] in ids]
    return out


def data_path(rel: str) -> Path:
    return DATA / rel


def fps(x: dict) -> float | None:
    f = x["source"].get("fps")
    return float(Fraction(f)) if f else None


def norm_team(name: str, aliases: dict[str, str]) -> str:
    n = name.lower()
    n = re.sub(r"\d{4}[- ]\d{2}[- ]\d{2}|\d{2} \d{2} \d{4}", " ", n)
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()  # names are hand-cleaned of scores ("sky 5" is a name)
    return aliases.get(n, n)


def team_overlap(m: dict) -> list[str]:
    """Problems with the team-disjoint rule (BENCHMARK.md 4). Empty list = OK."""
    al = m.get("team_aliases", {})
    teams = {"eval": {}, "train": {}}
    for x in m["matches"]:
        for t in x["teams"]:
            teams[x["split"]].setdefault(norm_team(t, al), []).append(x["id"])
    probs = []
    for t, ids in teams["eval"].items():
        if t in teams["train"]:
            probs.append(f"team {t!r} in eval {ids} AND train {teams['train'][t]}")
    # fuzzy: a distinctive word shared between an eval and a train team (catches unaliased
    # spellings like "salahds wins scrums" vs "salahds win scram")
    def words(t):
        return {w for w in t.split() if len(w) >= 4 and w not in _GENERIC}
    for te, ie in teams["eval"].items():
        for tt, it in teams["train"].items():
            common = words(te) & words(tt)
            if common and te != tt:
                probs.append(f"possible same team: eval {te!r} {ie} ~ train {tt!r} {it} "
                             f"(shared {sorted(common)}); add a team_alias or swap the match")
    return probs


def check(m: dict | None = None) -> int:
    m = m or load()
    ids = [x["id"] for x in m["matches"]]
    probs = [f"duplicate id {i}" for i in set(ids) if ids.count(i) > 1]
    vids = [x["video_id"] for x in m["matches"]]
    probs += [f"video {v} listed twice" for v in set(vids) if vids.count(v) > 1]
    for x in m["matches"]:
        if x["split"] not in ("eval", "train"):
            probs.append(f"{x['id']}: bad split {x['split']!r}")
        w = x["label_window"]
        if round(w["end_s"] - w["start_s"], 3) != 120.0:
            probs.append(f"{x['id']}: label window is not 120 s")
    probs += team_overlap(m)
    if probs:
        print("MANIFEST CHECK FAILED:", file=sys.stderr)
        for p in probs:
            print("  - " + p, file=sys.stderr)
        return 1
    al = m.get("team_aliases", {})
    for s in ("eval", "train"):
        ts = sorted({norm_team(t, al) for x in m["matches"] if x["split"] == s for t in x["teams"]})
        print(f"{s:5s} ({len(ts)} teams): {', '.join(ts)}")
    print("OK: train and eval are team-disjoint")
    return 0
