"""Per-clip adapter for the VLM identity methods (BENCHMARK.md 5, 6: rows B1 run 11, B2 run 13).

`fv.gk_lock` is wired to night-slice files; this rebuilds its inputs for any bench clip and runs
the same steps, without modifying the fv modules:

  shared (deterministic, once per clip + CV config; reused by repeats and by control + method):
    detect_track -> reid_track -> teams -> identity.extract (OSNet desc cache)
    -> jersey OCR (fv.jersey) -> context views (roster_v3.render_context, scale 1.0: bench boxes
       are already in clip pixels; the night run's 1.5 mapped 720p boxes onto 1080p frames)
  per repeat (stochastic):
    identity.vlm_fingerprints (fresh, no cache)          [vlm.fingerprints, default on]
    gk_lock pre-pass (relative motion + neutral VLM role) [method gk_lock only]
    roster_v3.cluster_team per team (emb_gate 0.55, min_conf 0.6, max_calls 40)
    -> roster._name_clusters / _emit -> identities.parquet

Note: roster.load_context needs the fingerprint file, but cluster_team and the GK pre-pass never
read fp_of (only the old roster approaches A/B do). Fingerprints therefore cost about one call
per tracklet without changing the identities; `vlm.fingerprints: false` writes an empty file.

Spend: a Meter wraps Gemini's generate_content and records calls, tokens (prompt / output /
thinking) and an estimated cost per stage; it goes into meta.json["vlm"]. The key is read from
~/football-vision/.env by path, only when a real (non-dry) client is built, and never printed.
`dry: true` uses a fake client (fixed JSON, zero cost) to test the wiring end to end.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pandas as pd

from . import manifest as M

KEY_ENV = Path("~/football-vision/.env").expanduser()
OSNET = M.REPO / "data" / "models" / "osnet_x1_0_msmt17.pt"
MIN_FRAMES = 15


# --------------------------------------------------------------------------- spend meter
class Meter:
    def __init__(self, model: str, price: dict):
        self.model, self.price = model, price
        self.stage = "?"
        self.lock = threading.Lock()
        self.by_stage: dict[str, dict] = {}

    def add(self, usage) -> None:
        g = lambda k: int(getattr(usage, k, 0) or 0) if usage is not None else 0  # noqa: E731
        with self.lock:
            s = self.by_stage.setdefault(self.stage, {"calls": 0, "in": 0, "out": 0, "think": 0})
            s["calls"] += 1
            s["in"] += g("prompt_token_count")
            s["out"] += g("candidates_token_count")
            s["think"] += g("thoughts_token_count")

    def summary(self) -> dict:
        tot = {k: sum(s[k] for s in self.by_stage.values()) for k in ("calls", "in", "out", "think")}
        cost = (tot["in"] * self.price["input"] + (tot["out"] + tot["think"]) * self.price["output"]) / 1e6
        return {"model": self.model, "calls": tot["calls"], "tokens_in": tot["in"],
                "tokens_out": tot["out"], "tokens_thinking": tot["think"],
                "cost_usd_est": round(cost, 4), "price_usd_per_mtok": self.price,
                "by_stage": self.by_stage}


def make_llm(vcfg: dict, dry: bool):
    from ..llm import LLMClient, get_llm
    price = vcfg.get("price_usd_per_mtok", {"input": 2.0, "output": 12.0})
    if dry:
        class DryLLM(LLMClient):
            def _vision(self, prompt, images, want_json):
                meter.add(None)
                return json.dumps({"same": False, "confidence": 0.5, "reason": "dry",
                                   "location": "on_pitch_outfield", "role": "outfield_player",
                                   "hair": "dark", "shirt": "plain", "shorts": "dark"})
        llm = DryLLM("dry")
        meter = Meter("dry", {"input": 0.0, "output": 0.0})
        return llm, meter
    if not os.environ.get("GEMINI_API_KEY"):
        from dotenv import dotenv_values
        key = dotenv_values(KEY_ENV).get("GEMINI_API_KEY")
        if not key:
            raise SystemExit(f"GEMINI_API_KEY not found in {KEY_ENV}")
        os.environ["GEMINI_API_KEY"] = key
    llm = get_llm({"llm": {"enabled": True, "provider": vcfg.get("provider", "gemini"),
                           "model": vcfg["model"]}})
    meter = Meter(llm.model, price)
    models = llm._client.models
    orig = models.generate_content

    def counted(*a, **kw):
        resp = orig(*a, **kw)
        meter.add(getattr(resp, "usage_metadata", None))
        return resp
    models.generate_content = counted
    return llm, meter


# --------------------------------------------------------------------------- shared stages
def prepare_shared(clip: Path, shared: Path, device: str = "cuda:0") -> dict:
    """identity desc cache + jersey + context views on top of the shared CV stages."""
    from .. import identity, jersey, roster_v3
    reid = shared / "reid.parquet"
    p = {"reid": reid, "teams": shared / "teams.parquet", "cache": shared / "identity_desc.npz",
         "jersey": shared / "jersey.parquet", "context": shared / "context"}
    t = {}
    if not p["cache"].exists():
        s = time.time()
        identity.extract(clip, reid, p["cache"], device=device, reid_weights=str(OSNET))
        t["identity_extract"] = time.time() - s
    if not p["jersey"].exists():
        s = time.time()
        jersey.run(clip, reid, p["jersey"], device=device)
        t["jersey"] = time.time() - s
    if not (p["context"] / ".done").exists():
        s = time.time()
        df = pd.read_parquet(reid)
        n = df.groupby("track_id").size()
        keep = sorted(int(x) for x in n[n >= MIN_FRAMES].index)
        roster_v3.render_context(clip, df, keep, p["context"], scale=1.0)
        (p["context"] / ".done").write_text(json.dumps({"tracks": keep}))
        t["context"] = time.time() - s
    p["timing"] = {k: round(v, 1) for k, v in t.items()}
    return p


# --------------------------------------------------------------------------- one repeat
def run_repeat(method: str, vcfg: dict, clip: Path, p: dict, out: Path, llm, meter: Meter) -> dict:
    """method: roster_v3 (= run 11, the control) | gk_lock (= run 13). Returns a log dict."""
    import re

    from .. import gk_lock as G
    from ..identity import vlm_fingerprints
    from ..roster import _emit, _name_clusters, load_context
    from ..roster_v3 import cluster_team, cooccur_veto
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    fp = work / "vlm_fp.parquet"
    t = {}
    s = time.time()
    meter.stage = "fingerprints"
    if vcfg.get("fingerprints", True):
        vlm_fingerprints(clip, p["reid"], fp, llm, cache=False)
    if not fp.exists():
        pd.DataFrame({"track_id": pd.Series(dtype=int)}).to_parquet(fp, index=False)
    t["fingerprints"] = time.time() - s

    ctx = load_context(p["cache"], p["teams"], p["jersey"], fp, MIN_FRAMES)
    tracks = sorted({int(re.match(r"t(\d+)_", f.name).group(1)) for f in p["context"].glob("t*_*.jpg")})
    df = pd.read_parquet(p["reid"])
    views = G.load_views(p["context"], tracks)
    veto, _ = cooccur_veto(df[df["track_id"].isin(tracks)], tracks)
    log: dict = {"mode": method, "n_tracks": len(tracks), "n_veto": len(veto)}
    locked: list[int] = []
    if method == "gk_lock":
        s = time.time()
        meter.stage = "gk_prepass"
        rel = G.relative_motion(df, tracks)
        locked, roles, med = G.gk_prepass(llm, views, tracks, rel)
        log.update(gk_locked=locked, rel_motion_median=round(med, 2),
                   gk_votes={int(k): v for k, v in roles.items() if v["role"] == "goalkeeper"})
        t["gk_prepass"] = time.time() - s
    s = time.time()
    meter.stage = "grouping"
    cp = vcfg.get("cluster", {})
    max_calls = cp.get("max_calls", 40)
    groups, calls, cap_hits, merges = [], {}, [], {}
    for team in ("A", "B", "O"):
        ids = [x for x in tracks if ctx["team_of"].get(x, "O") == team and x not in locked]
        if not ids:
            continue
        g, mg, c = cluster_team(ids, ctx["embs"], ctx["frames"], ctx["num_of"], veto, views, llm,
                                emb_gate=cp.get("emb_gate", 0.55), min_conf=cp.get("min_conf", 0.6),
                                max_calls=max_calls)
        groups += g
        calls[team] = c
        merges[team] = [list(m[:3]) for m in mg]
        if c >= max_calls:
            cap_hits.append(team)
    t["grouping"] = time.time() - s
    by_track = _name_clusters(groups, ctx)
    gg = G.gk_groups(locked, ctx["team_of"], veto)
    for k, g in enumerate(gg):
        team = ctx["team_of"].get(g[0], "O")
        n_same = sum(1 for h in gg if ctx["team_of"].get(h[0], "O") == team)
        for x in g:
            by_track[x] = f"{team}-GK" if n_same == 1 else f"{team}-GK{k + 1}"
    ident = work / "identities.parquet"
    _emit(by_track, ctx, ident)
    log.update(grouping_calls=calls, cap_hit=cap_hits, merges=merges,
               n_identities=len(set(by_track.values())), stage_s={k: round(v, 1) for k, v in t.items()})
    (work / "vlm_log.json").write_text(json.dumps(log, indent=1, default=str))
    return {"identities": ident, "log": log}
