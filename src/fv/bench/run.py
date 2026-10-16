"""`fv.bench run`: run one method config on every ready eval clip (BENCHMARK.md 5).

    python -m fv.bench run --method bench/methods/b0-cv-job.yaml --batch 2026-10-16-base

Writes ~/footyviz-bench-data/runs/<method>@<batch>/<clip>/r<k>/:
    dets.parquet   frame, x1,y1,x2,y2, conf, track_id, identity_id, team (A|B|O), role
    ball.parquet   frame, x, y, conf
    meta.json      git sha, config + hash, versions, wall time per stage, GPU, vlm, seed
    work/          stage intermediates
plus run.json at the run root. Stages are adapters over the fv modules (parquet in, parquet out);
the fv modules themselves are not modified.

Method yaml:
    name, stochastic (bool), repeats (int)
    detector: {weights, imgsz, conf, ball_conf}
    tracker:  {name: bytetrack | botsort, params: {track_buffer, appearance_thresh}}
               bytetrack = fv.detect_track's ByteTrack only; botsort = + fv.reid_track (OSNet)
    teams:    {method: kmeans_color, k: null}
    identity: {method: none | cv_fusion, params: {threshold}}
               (roster_v3 / gk_lock need per-clip adapters: not wired yet)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from . import manifest as M

RUNS = M.DATA / "runs"
DEFAULT_WEIGHTS = {"yolo11m.pt": M.REPO / "yolo11m.pt"}
OSNET = M.REPO / "data" / "models" / "osnet_x1_0_msmt17.pt"


def load_method(path: str) -> dict:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text())
    if "inherits" in cfg:
        base = load_method(str(p.parent / cfg.pop("inherits")))
        for k, v in cfg.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        cfg = base
    return cfg


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def git_sha() -> str:
    r = subprocess.run(["git", "-C", str(M.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(M.REPO), "status", "--porcelain", "src"], capture_output=True, text=True)
    return r.stdout.strip() + ("+dirty" if d.stdout.strip() else "")


def weights(name: str) -> str:
    for cand in (DEFAULT_WEIGHTS.get(name), M.DATA / "models" / name):
        if cand and Path(cand).exists():
            return str(cand)
    return name


def run_clip(cfg: dict, clip: Path, out: Path, device: str = "cuda:0") -> dict:
    from .. import detect_track, teams
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    t = {}
    d = cfg["detector"]
    s = time.time()
    tracks = work / "tracks.parquet"
    detect_track.run(clip, tracks, model_path=weights(d["weights"]), device=device,
                     conf=d.get("conf", 0.30), ball=True, ball_conf=d.get("ball_conf", 0.10),
                     imgsz=d.get("imgsz", 640))
    t["detect"] = time.time() - s
    person = tracks
    tr = cfg["tracker"]
    if tr["name"] == "botsort":
        from .. import reid_track
        s = time.time()
        person = work / "reid.parquet"
        p = tr.get("params", {})
        reid_track.run(clip, tracks, person, device=device, reid_weights=str(OSNET),
                       track_buffer=p.get("track_buffer", 90),
                       appearance_thresh=p.get("appearance_thresh", 0.25))
        t["reid"] = time.time() - s
    elif tr["name"] != "bytetrack":
        raise SystemExit(f"unknown tracker {tr['name']!r}")
    s = time.time()
    team_map = work / "teams.parquet"
    teams.run(clip, person, team_map, k=cfg.get("teams", {}).get("k"))
    t["teams"] = time.time() - s
    idm = cfg.get("identity", {}).get("method", "none")
    ident = None
    if idm == "cv_fusion":
        from .. import identity
        s = time.time()
        ident = work / "identities.parquet"
        identity.run(clip, person, ident, team_map, device=device, cache=work / "identity_desc.npz",
                     reid_weights=str(OSNET),
                     threshold=cfg["identity"].get("params", {}).get("threshold", 0.42))
        t["identity"] = time.time() - s
    elif idm != "none":
        raise SystemExit(f"identity method {idm!r} has no per-clip adapter yet (roster_v3/gk_lock: TODO)")

    # -> prediction format
    df = pd.read_parquet(person)
    tm = pd.read_parquet(team_map)
    team_of = dict(zip(tm["track_id"].astype(int), tm["team"]))
    df["team"] = df["track_id"].map(lambda x: team_of.get(int(x), "O"))
    if ident is not None:
        idf = pd.read_parquet(ident)
        sid = dict(zip(idf["raw_track_id"].astype(int), idf["stable_id"]))
        df["identity_id"] = df["track_id"].map(lambda x: sid.get(int(x), f"t{int(x)}"))
        iteam = dict(zip(idf["stable_id"], idf["team"]))
        df["team"] = df["identity_id"].map(lambda x: iteam.get(x)).fillna(df["team"])
    else:
        df["identity_id"] = df["track_id"].map(lambda x: f"t{int(x)}")
    df["role"] = None
    df = df.rename(columns={"confidence": "conf"})[
        ["frame", "x1", "y1", "x2", "y2", "conf", "track_id", "identity_id", "team", "role"]]
    df.to_parquet(out / "dets.parquet", index=False)
    b = pd.read_parquet(detect_track.ball_path(tracks))
    pd.DataFrame({"frame": b["frame"], "x": (b.x1 + b.x2) / 2, "y": (b.y1 + b.y2) / 2,
                  "conf": b["confidence"]}).to_parquet(out / "ball.parquet", index=False)
    return {k: round(v, 1) for k, v in t.items()}


def versions() -> dict:
    import importlib.metadata as md
    import torch
    out = {}
    for p in ("ultralytics", "supervision", "boxmot", "torch"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


def main(method: str, batch: str, ids: list[str] | None, repeats: int | None, force: bool = False) -> int:
    cfg = load_method(method)
    reps = repeats or (cfg.get("repeats", 3) if cfg.get("stochastic") else 1)
    name = f"{cfg['name']}@{batch}"
    root = RUNS / name
    root.mkdir(parents=True, exist_ok=True)
    m = M.load()
    clips = [x for x in M.matches(m, split="eval", ids=ids) if x.get("status") == "ready"]
    sha, ver = git_sha(), versions()
    (root / "run.json").write_text(json.dumps({
        "run": name, "method": cfg["name"], "batch": batch, "date": time.strftime("%Y-%m-%d"),
        "commit": sha, "config": cfg, "config_hash": cfg_hash(cfg), "repeats": reps,
        "clips": [x["id"] for x in clips], "versions": ver}, indent=1))
    for x in clips:
        clip = M.data_path(x["clip_file"])
        if not clip.exists():
            print(f"[{x['id']}] no clip; run `fv.bench cut --clip {x['id']}`")
            continue
        for k in range(1, reps + 1):
            out = root / x["id"] / f"r{k}"
            if (out / "meta.json").exists() and not force:
                print(f"[{name}] {x['id']} r{k} exists, skipped")
                continue
            out.mkdir(parents=True, exist_ok=True)
            s = time.time()
            timing = run_clip(cfg, clip, out)
            meta = {"run": name, "clip": x["id"], "repeat": k, "commit": sha, "config_hash": cfg_hash(cfg),
                    "versions": ver, "wall_s": round(time.time() - s, 1), "stage_s": timing,
                    "vlm": None, "seed": None, "date": time.strftime("%Y-%m-%d %H:%M")}
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[{name}] {x['id']} r{k}: {meta['wall_s']} s {timing}")
    print(f"-> {root}")
    return 0
