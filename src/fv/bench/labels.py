"""Label state -> GT files (BENCHMARK.md 3, "Label files").

The label page saves its raw state as labels/<id>/work.json. `derive` turns prelabel.json +
work.json into the schema files the scorer reads: clip.json (fvbench.clip/1, L0 gallery) and
keyframes.json (fvbench.keyframes/1, L1 + L1b).

work.json:
  people:     {"P01": {"team": "T1"|"T2"|null, "role": "outfield"|"keeper"|"referee", "jersey": str|null}}
  track_pid:  {"<track>": "P01" | "ignore"}             gallery: tracklet -> person (merge/split)
  box:        {"<bid>": {"role"?, "pid"?, "why"?, "team"?}}   per-box overrides on keyframes
  added:      {"<frame>": [{"bid", "box", "role", "pid", "why", "team"}]}
  deleted:    ["<bid>", ...]                             false-positive boxes
  kf_done:    {"<frame>": seconds}                       keyframes Daniel finished
  seconds:    {"gallery": s, "keyframes": s}
  teams:      {"T1": kit text, "T2": kit text}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import manifest as M

LABELS = M.DATA / "labels"


def empty_work() -> dict:
    return {"people": {}, "track_pid": {}, "box": {}, "added": {}, "deleted": [],
            "kf_done": {}, "seconds": {"gallery": 0, "keyframes": 0}, "teams": {}}


def load_work(cid: str) -> dict:
    p = LABELS / cid / "work.json"
    w = empty_work()
    if p.exists():
        w.update(json.loads(p.read_text()))
    return w


def resolve_box(b: dict, w: dict) -> dict:
    """One keyframe box -> {box, role, pid, why?, team?, status}."""
    own = {k: b[k] for k in ("role", "pid", "why", "team") if b.get(k) is not None}  # added boxes
    ov = {**own, **w["box"].get(b["bid"], {})}
    tp = w["track_pid"].get(str(b["track"])) if b.get("track") is not None else None
    role = ov.get("role")
    pid = ov.get("pid")
    if role is None:
        role = "ignore" if tp == "ignore" else "player"
    if pid is None and tp not in (None, "ignore"):
        pid = tp
    out = {"box": [round(v, 1) for v in ov.get("box", b["box"])], "role": role}
    if role == "ignore":
        out["why"] = ov.get("why", "off_pitch")
    else:
        if pid is None:
            pid = "unsure" if ov else None
        person = w["people"].get(pid or "", {})
        if person.get("role") == "referee":
            out["role"] = "referee"
        out["pid"] = pid
        if pid in (None, "unsure"):
            out["team"] = ov.get("team")
    out["status"] = "verified" if (ov or tp is not None or b.get("added")) else "prelabel"
    return out


def kf_boxes(pl: dict, w: dict, kf: dict) -> list[dict]:
    dele = set(w["deleted"])
    boxes = [b for b in kf["boxes"] if b["bid"] not in dele]
    boxes += [dict(b, added=True, track=None) for b in w["added"].get(str(kf["frame"]), [])
              if b["bid"] not in dele]
    return [dict(resolve_box(b, w), bid=b["bid"]) for b in boxes]


def derive(cid: str) -> dict:
    d = LABELS / cid
    pl = json.loads((d / "prelabel.json").read_text())
    w = load_work(cid)
    frames = []
    seen: dict[str, list] = {}
    for kf in pl["keyframes"]:
        boxes = kf_boxes(pl, w, kf)
        for b in boxes:
            if b["role"] != "ignore" and b.get("pid") not in (None, "unsure"):
                seen.setdefault(b["pid"], []).append((kf["frame"], b["box"]))
        done = str(kf["frame"]) in w["kf_done"]
        frames.append({"frame": kf["frame"], "src_s": kf["src_s"], "blind": kf["blind"],
                       "done": done, "seconds": w["kf_done"].get(str(kf["frame"]), 0),
                       "boxes": [{k: v for k, v in b.items() if k != "status"} | {"bid": b["bid"]}
                                 for b in boxes]})
    people = []
    for pid in sorted(seen):
        p = w["people"].get(pid, {})
        fs = [f for f, _ in seen[pid]]
        ex = max(seen[pid], key=lambda fb: fb[1][3] - fb[1][1])
        people.append({"pid": pid, "team": p.get("team"), "role": p.get("role", "outfield"),
                       "jersey": p.get("jersey"), "exemplar": {"frame": ex[0], "box": ex[1]},
                       "on_pitch": [[min(fs), max(fs)]], "n_kf": len(set(fs))})
    tc = {"T1": 0, "T2": 0, "referee": 0}
    for p in people:
        if p["role"] == "referee":
            tc["referee"] += 1
        elif p["team"] in tc:
            tc[p["team"]] += 1
    n_done = sum(f["done"] for f in frames)
    status = "verified" if n_done == len(frames) else "partial"
    prov = {"status": status, "labeller": "daniel", "tool": "fvlabel 0.1",
            "prelabel": pl["prelabeller"], "date": time.strftime("%Y-%m-%d"),
            "seconds": w["seconds"], "kf_done": n_done, "kf_total": len(frames)}
    kits = pl.get("kits") or {}
    clip = {"schema": "fvbench.clip/1", "clip_id": cid, "video_id": pl["video_id"],
            "start_s": pl["start_s"], "end_s": pl["end_s"], "start_src_s": pl["start_src_s"],
            "fps": pl["fps"], "n_frames": pl["n_frames"], "width": pl["width"], "height": pl["height"],
            "teams": {t: {"kit": w["teams"].get(t) or kits.get(t)} for t in ("T1", "T2")},
            "people": people, "true_count": tc, "provenance": prov}
    kfj = {"schema": "fvbench.keyframes/1", "clip_id": cid, "stride": pl["stride"],
           "fps": pl["fps"], "frames": frames, "provenance": prov}
    (d / "clip.json").write_text(json.dumps(clip, indent=1))
    (d / "keyframes.json").write_text(json.dumps(kfj, indent=1))
    return {"status": status, "kf_done": n_done, "people": len(people), "true_count": tc}
