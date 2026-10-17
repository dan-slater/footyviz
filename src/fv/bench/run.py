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
    identity: {method: none | cv_fusion | roster_v3 | gk_lock, params: {threshold}}
               roster_v3 = run 11 (the identity control), gk_lock = run 13; both via fv.bench.vlm
    vlm:      {provider, model, fingerprints, cluster: {emb_gate, min_conf, max_calls},
               price_usd_per_mtok: {input, output}}
    control:  <method yaml> run first in the same batch (same-day control, BENCHMARK.md 5)

Stochastic (VLM) methods refuse to run without --allow-spend (real Gemini calls, cost logged
in meta.json["vlm"] and <run>/spend.json) or --dry-vlm (fake client, $0, wiring test only).
The deterministic CV stages live in runs/_shared/<clip>-<cv hash>/ and are reused.
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


def cv_key(cfg: dict) -> str:
    return cfg_hash({k: cfg.get(k) for k in ("detector", "tracker", "teams")})


def cv_stages(cfg: dict, clip: Path, work: Path, device: str = "cuda:0") -> tuple[Path, Path, Path, dict]:
    """Deterministic detect -> (re)track -> teams. Skips stages whose output exists."""
    from .. import detect_track, teams
    work.mkdir(parents=True, exist_ok=True)
    t = {}
    d = cfg["detector"]
    tracks = work / "tracks.parquet"
    if not tracks.exists():
        s = time.time()
        detect_track.run(clip, tracks, model_path=weights(d["weights"]), device=device,
                         conf=d.get("conf", 0.30), ball=True, ball_conf=d.get("ball_conf", 0.10),
                         imgsz=d.get("imgsz", 640))
        t["detect"] = time.time() - s
    person = tracks
    tr = cfg["tracker"]
    if tr["name"] == "botsort":
        from .. import reid_track
        person = work / "reid.parquet"
        if not person.exists():
            s = time.time()
            p = tr.get("params", {})
            reid_track.run(clip, tracks, person, device=device, reid_weights=str(OSNET),
                           track_buffer=p.get("track_buffer", 90),
                           appearance_thresh=p.get("appearance_thresh", 0.25))
            t["reid"] = time.time() - s
    elif tr["name"] != "bytetrack":
        raise SystemExit(f"unknown tracker {tr['name']!r}")
    team_map = work / "teams.parquet"
    if not team_map.exists():
        s = time.time()
        teams.run(clip, person, team_map, k=cfg.get("teams", {}).get("k"))
        t["teams"] = time.time() - s
    return tracks, person, team_map, t


VLM_METHODS = ("roster_v3", "gk_lock")


def run_clip(cfg: dict, clip: Path, out: Path, device: str = "cuda:0", shared: Path | None = None,
             llm=None, meter=None) -> tuple[dict, dict | None]:
    from .. import detect_track
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    idm = cfg.get("identity", {}).get("method", "none")
    tracks, person, team_map, t = cv_stages(cfg, clip, shared or work, device)
    ident, vlog = None, None
    if idm == "cv_fusion":
        from .. import identity
        s = time.time()
        ident = work / "identities.parquet"
        identity.run(clip, person, ident, team_map, device=device, cache=work / "identity_desc.npz",
                     reid_weights=str(OSNET),
                     threshold=cfg["identity"].get("params", {}).get("threshold", 0.42))
        t["identity"] = time.time() - s
    elif idm in VLM_METHODS:
        from . import vlm
        if cfg["tracker"]["name"] != "botsort":
            raise SystemExit("VLM identity methods need tracker botsort (reid tracklets)")
        p = vlm.prepare_shared(clip, shared, device)
        t.update(p["timing"])
        r = vlm.run_repeat(idm, cfg.get("vlm", {}), clip, p, out, llm, meter)
        ident, vlog = r["identities"], r["log"]
        t.update(vlog["stage_s"])
    elif idm != "none":
        raise SystemExit(f"unknown identity method {idm!r}")

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
    return {k: round(v, 1) for k, v in t.items()}, vlog


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


def estimate_calls(cfg: dict, clips: list[dict], reps: int) -> str:
    """Upper-bound VLM call count from the B0 tracklet counts (fingerprints 1/track, GK role
    1/track, grouping <= max_calls per team x 3 teams)."""
    v, idm = cfg.get("vlm", {}), cfg["identity"]["method"]
    n_tr = 120  # B0 batch 2026-10-16: 84-121 BoT-SORT tracks per eval clip
    per = (n_tr if v.get("fingerprints", True) else 0) + (n_tr if idm == "gk_lock" else 0) + \
        3 * v.get("cluster", {}).get("max_calls", 40)
    return f"<= ~{per} calls per clip-repeat x {len(clips)} clips x {reps} repeats = <= ~{per * len(clips) * reps} calls"


def main(method: str, batch: str, ids: list[str] | None, repeats: int | None, force: bool = False,
         allow_spend: bool = False, dry_vlm: bool = False, _is_control: bool = False) -> int:
    cfg = load_method(method)
    stochastic = bool(cfg.get("stochastic"))
    reps = repeats or (cfg.get("repeats", 3) if stochastic else 1)
    m = M.load()
    clips = [x for x in M.matches(m, split="eval", ids=ids) if x.get("status") == "ready"]
    is_vlm = cfg.get("identity", {}).get("method") in VLM_METHODS
    if is_vlm and not (allow_spend or dry_vlm):
        print(f"REFUSED: {cfg['name']} makes paid VLM calls ({cfg['vlm']['model']}); "
              f"{estimate_calls(cfg, clips, reps)}.\n"
              f"  Re-run with --allow-spend once approved, or --dry-vlm to test the wiring for $0.")
        return 3
    if cfg.get("control") and not _is_control:
        ctl = str(Path(method).parent / cfg["control"])
        print(f"[{cfg['name']}] same-day control first: {cfg['control']} in batch {batch}")
        rc = main(ctl, batch, ids, repeats, force, allow_spend, dry_vlm, _is_control=True)
        if rc:
            return rc
    if dry_vlm and is_vlm:
        batch = batch + "-dry"  # never mixes with paid runs on the board
    name = f"{cfg['name']}@{batch}"
    root = RUNS / name
    root.mkdir(parents=True, exist_ok=True)
    sha, ver = git_sha(), versions()
    (root / "run.json").write_text(json.dumps({
        "run": name, "method": cfg["name"], "batch": batch, "date": time.strftime("%Y-%m-%d"),
        "commit": sha, "config": cfg, "config_hash": cfg_hash(cfg), "repeats": reps,
        "stochastic": stochastic, "dry_vlm": bool(dry_vlm and is_vlm), "control": cfg.get("control"),
        "clips": [x["id"] for x in clips], "versions": ver}, indent=1))
    for x in clips:
        clip = M.data_path(x["clip_file"])
        if not clip.exists():
            print(f"[{x['id']}] no clip; run `fv.bench cut --clip {x['id']}`")
            continue
        shared = RUNS / "_shared" / f"{x['id']}-{cv_key(cfg)}" if is_vlm else None
        for k in range(1, reps + 1):
            out = root / x["id"] / f"r{k}"
            if (out / "meta.json").exists() and not force:
                print(f"[{name}] {x['id']} r{k} exists, skipped")
                continue
            out.mkdir(parents=True, exist_ok=True)
            llm = meter = None
            if is_vlm:
                from . import vlm
                llm, meter = vlm.make_llm(cfg.get("vlm", {}), dry=dry_vlm)
            s = time.time()
            timing, vlog = run_clip(cfg, clip, out, shared=shared, llm=llm, meter=meter)
            spend = None
            if meter is not None:
                spend = meter.summary() | {"cap_hit": vlog["cap_hit"], "grouping_calls": vlog["grouping_calls"],
                                           "dry": bool(dry_vlm), "cache": "off"}
            meta = {"run": name, "clip": x["id"], "repeat": k, "commit": sha, "config_hash": cfg_hash(cfg),
                    "versions": ver, "wall_s": round(time.time() - s, 1), "stage_s": timing,
                    "vlm": spend, "identity_log": vlog, "shared": str(shared) if shared else None,
                    "seed": None, "date": time.strftime("%Y-%m-%d %H:%M")}
            (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
            msg = f" vlm {spend['calls']} calls ${spend['cost_usd_est']:.2f} cap_hit {spend['cap_hit']}" if spend else ""
            print(f"[{name}] {x['id']} r{k}: {meta['wall_s']} s {timing}{msg}")
    if is_vlm:
        rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/r*/meta.json"))]
        tot = {"run": name, "clip_repeats": len(rows),
               "calls": sum(r["vlm"]["calls"] for r in rows if r.get("vlm")),
               "cost_usd_est": round(sum(r["vlm"]["cost_usd_est"] for r in rows if r.get("vlm")), 2),
               "cap_hits": {f"{r['clip']}/r{r['repeat']}": r["vlm"]["cap_hit"] for r in rows
                            if r.get("vlm") and r["vlm"]["cap_hit"]}}
        (root / "spend.json").write_text(json.dumps(tot, indent=1))
        print(f"[{name}] spend: {tot['calls']} calls, ~${tot['cost_usd_est']:.2f} (est.)")
    print(f"-> {root}")
    return 0
