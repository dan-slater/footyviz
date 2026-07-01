"""M4 comparison — two VLM-driven ways to fuse tracklets into a stable roster.

Both consume the CV descriptor cache (fv.identity extract) + the cached VLM
fingerprints (fv.identity.vlm_fingerprints) and add NO per-crop re-description.
They exist to be compared against the CV-only fusion (fv.identity.fuse).

Shared pre-step: drop noise tracklets (lifetime < min_frames) so flicker
detections don't inflate the roster.

Approach A — bucketed attributes (no new API calls):
    parse each fingerprint's free text into DISCRETE buckets (hair colour/length,
    beard, skin, shirt colour/sleeve, shorts, socks, build) and cluster on the
    bucket vector, weighting the discriminative fields. Team hard-gate, temporal
    cannot-link, jersey number confirms/breaks ties.

Approach B — VLM-as-clusterer (bounded new API calls):
    build one labelled montage per team and ask the VLM which track-ids are the
    same player. Team gate is implicit (one montage per team); temporal
    cannot-link is enforced afterwards (two ids that coexist can't be one player).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .identity import _agglomerate

TEAM_ORDER = ("A", "B", "O")


# --------------------------------------------------------------------------- #
# shared context
# --------------------------------------------------------------------------- #
def load_context(cache: Path, teams: Path, jersey: Path | None, vlm_fp: Path,
                 min_frames: int = 15):
    z = np.load(cache, allow_pickle=True)
    ids = [int(t) for t in z["track_ids"]]
    frames = {ids[i]: set(int(x) for x in f) for i, f in enumerate(z["frames"])}
    embs = {ids[i]: z["embs"][i] for i in range(len(ids))}

    dropped = [t for t in ids if len(frames[t]) < min_frames]
    keep = [t for t in ids if len(frames[t]) >= min_frames]

    tm = pd.read_parquet(teams)
    team_of = dict(zip(tm["track_id"].astype(int), tm["team"]))

    num_of = {}
    if jersey is not None and Path(jersey).exists():
        jdf = pd.read_parquet(jersey)
        for _, r in jdf.iterrows():
            if bool(r.get("confident", False)) and int(r["number"]) >= 0:
                num_of[int(r["track_id"])] = int(r["number"])

    fdf = pd.read_parquet(vlm_fp)
    attr_cols = [c for c in fdf.columns if c != "track_id"]
    fp_of = {}
    for _, r in fdf.iterrows():
        fp_of[int(r["track_id"])] = {c: (str(r[c]) if isinstance(r[c], str) else "") for c in attr_cols}

    return {"keep": keep, "dropped": dropped, "frames": frames, "embs": embs,
            "team_of": team_of, "num_of": num_of, "fp_of": fp_of}


def _emit(clusters_by_track: dict[int, str], ctx, out: Path, id_table_extra=None):
    """clusters_by_track: raw_track_id -> stable_id. Writes identities parquet + table."""
    rows = [{"raw_track_id": t, "stable_id": s, "team": ctx["team_of"].get(t, "O"),
             "number": ctx["num_of"].get(t, -1)} for t, s in clusters_by_track.items()]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("raw_track_id").to_parquet(out, index=False)
    # table
    from collections import defaultdict
    members = defaultdict(list)
    for t, s in clusters_by_track.items():
        members[s].append(t)
    tbl = []
    for sid, ts in members.items():
        team = _majority([ctx["team_of"].get(t, "O") for t in ts])
        nums = [ctx["num_of"][t] for t in ts if t in ctx["num_of"]]
        n_frames = len(set().union(*[ctx["frames"][t] for t in ts]))
        tbl.append({"stable_id": sid, "team": team,
                    "number": _majority(nums) if nums else -1,
                    "n_raw_tracks": len(ts), "n_frames": n_frames,
                    "raw_track_ids": ",".join(map(str, sorted(ts)))})
    tp = out.parent / f"{out.stem}_table.parquet"
    pd.DataFrame(tbl).sort_values(["team", "stable_id"]).to_parquet(tp, index=False)
    n_ids = len(members)
    per_team = pd.Series([r["team"] for r in tbl]).value_counts().to_dict()
    print(f"  -> {len(clusters_by_track)} tracks -> {n_ids} stable ids  per-team {per_team}  "
          f"(numbered {sum(1 for r in tbl if r['number'] >= 0)})")
    return out


def _majority(xs):
    return pd.Series(xs).value_counts().index[0] if len(xs) else -1


def _name_clusters(groups: list[list[int]], ctx) -> dict[int, str]:
    """Assign stable ids: team + jersey number if any, else team-local index."""
    by_track = {}
    counter = {}
    for members in groups:
        team = _majority([ctx["team_of"].get(t, "O") for t in members])
        nums = [ctx["num_of"][t] for t in members if t in ctx["num_of"]]
        if nums:
            sid = f"{team}#{_majority(nums)}"
        else:
            counter[team] = counter.get(team, 0) + 1
            sid = f"{team}-{counter[team]}"
        for t in members:
            by_track[t] = sid
    return by_track


# --------------------------------------------------------------------------- #
# Approach A — bucketed attributes
# --------------------------------------------------------------------------- #
_COLORS = ["white", "black", "navy", "blue", "red", "pink", "orange", "green",
           "yellow", "purple", "maroon", "grey", "gray", "silver", "beige",
           "blond", "brown", "dark", "light"]
_COLOR_CANON = {"gray": "grey", "navy": "blue", "blonde": "blond"}


def _color(text: str) -> str:
    t = (text or "").lower()
    for c in _COLORS:
        if c in t:
            return _COLOR_CANON.get(c, c)
    return ""


def _bucket(fp: dict) -> dict:
    hair = (fp.get("hair", "") or "").lower()
    facial = (fp.get("facial_hair", "") or "").lower()
    return {
        "hair_color": _color(hair),
        "hair_len": ("long" if "long" in hair else "short" if "short" in hair
                     else "med" if ("medium" in hair or "mid" in hair) else ""),
        "beard": ("y" if ("beard" in hair or "beard" in facial or
                          ("facial" in facial and "none" not in facial and facial)) else
                  "n" if facial in ("none", "clean", "clean-shaven", "clean shaven") else ""),
        "skin": ("dark" if "dark" in (fp.get("skin", "").lower()) else
                 "medium" if "medium" in (fp.get("skin", "").lower()) or "tan" in (fp.get("skin", "").lower()) else
                 "fair" if ("fair" in (fp.get("skin", "").lower()) or "light" in (fp.get("skin", "").lower())) else ""),
        "shirt_color": _color(fp.get("shirt", "")),
        "shirt_sleeve": ("long" if "long" in (fp.get("shirt", "").lower()) else
                         "short" if "short" in (fp.get("shirt", "").lower()) else ""),
        "shorts_color": _color(fp.get("shorts", "")),
        "socks_color": _color(fp.get("socks", "")),
        "build": ("stocky" if any(w in (fp.get("build", "").lower()) for w in ("stocky", "heavy", "muscular", "sturdy", "large")) else
                  "lean" if any(w in (fp.get("build", "").lower()) for w in ("lean", "slim", "thin", "slight")) else
                  "average" if fp.get("build") else ""),
    }


# field -> weight in the bucket distance (discriminative fields weigh more).
_BUCKET_W = {"socks_color": 1.3, "hair_color": 1.0, "beard": 1.0, "build": 0.8,
             "shirt_sleeve": 0.9, "shorts_color": 0.8, "shirt_color": 0.4,
             "skin": 0.5, "hair_len": 0.5}


def _bucket_dist(a: dict, b: dict) -> float:
    num = den = 0.0
    for k, w in _BUCKET_W.items():
        va, vb = a.get(k, ""), b.get(k, "")
        if not va or not vb:
            continue
        num += w * (0.0 if va == vb else 1.0)
        den += w
    return num / den if den > 0 else 0.5


def fuse_buckets(ctx, out: Path, threshold: float = 0.30) -> Path:
    keep = ctx["keep"]
    buckets = {t: _bucket(ctx["fp_of"].get(t, {})) for t in keep}
    n = len(keep)
    INF = 1e9
    D = np.full((n, n), INF)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = keep[i], keep[j]
            if ctx["team_of"].get(ti, "O") != ctx["team_of"].get(tj, "O"):
                continue
            if ctx["frames"][ti] & ctx["frames"][tj]:
                continue
            ni, nj = ctx["num_of"].get(ti), ctx["num_of"].get(tj)
            if ni is not None and nj is not None and ni != nj:
                continue
            d = _bucket_dist(buckets[ti], buckets[tj])
            if ni is not None and nj is not None and ni == nj:
                d *= 0.2
            D[i, j] = D[j, i] = d
    clusters = _agglomerate(D, threshold, INF)
    groups = [[keep[i] for i in c] for c in clusters]
    print(f"[bucket] threshold {threshold}")
    return _emit(_name_clusters(groups, ctx), ctx, out)


# --------------------------------------------------------------------------- #
# Approach B — VLM-as-clusterer (montage grouping per team)
# --------------------------------------------------------------------------- #
def _montage(track_ids: list[int], crop_dir: Path, cell=(96, 150), cols=6):
    cells, order = [], []
    for t in track_ids:
        p = crop_dir / f"t{t}.jpg"
        if not p.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, cell)
        band = np.full((26, cell[0], 3), 20, np.uint8)
        cv2.putText(band, f"{t}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cells.append(np.vstack([band, im]))
        order.append(t)
    if not cells:
        return None, []
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 40))
        rows.append(np.hstack(row))
    return np.vstack(rows), order


_P_GROUP = (
    "This grid shows cropped images of soccer players from ONE team in a single "
    "match; each crop is labelled with an integer ID in green. Some IDs are the "
    "SAME physical player reappearing (judge by hair, facial hair, skin, build, "
    "socks, shoes, shorts shade — NOT by pose or motion blur). Group IDs that are "
    "the same person. Return ONLY JSON {\"groups\": [[id,id,...], ...]}. Every ID "
    "shown must appear in exactly one group; a player seen once is a singleton group."
)


def _vlm_groups(llm, montage_img, ids: list[int]) -> list[list[int]]:
    txt = llm._vision(_P_GROUP + f"\nThe IDs present are: {ids}.", [montage_img], want_json=True)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    groups = []
    if m:
        try:
            groups = json.loads(m.group(0)).get("groups", [])
        except json.JSONDecodeError:
            groups = []
    # keep only known ids; ensure every id is covered (uncovered -> singleton)
    seen = set()
    clean = []
    for g in groups:
        gg = [int(x) for x in g if int(x) in ids and int(x) not in seen]
        for x in gg:
            seen.add(x)
        if gg:
            clean.append(gg)
    for t in ids:
        if t not in seen:
            clean.append([t])
    return clean


def _split_temporal(group: list[int], frames: dict) -> list[list[int]]:
    """Partition a VLM group into temporally-consistent identities.

    The VLM says these track-ids are one player, but two tracks that coexist in
    any frame cannot be — so greedily merge tracks into components, allowing a
    merge only when the resulting component has NO internal frame overlap. This
    avoids the transitive-chaining bug where a-b and b-c merge a and c even
    though a and c are on screen together.
    """
    comps = [[t] for t in group]
    fset = {t: frames[t] for t in group}
    merged = True
    while merged:
        merged = False
        for a in range(len(comps)):
            fa = set().union(*[fset[t] for t in comps[a]])
            for b in range(a + 1, len(comps)):
                fb = set().union(*[fset[t] for t in comps[b]])
                if not (fa & fb):
                    comps[a] += comps[b]
                    comps.pop(b)
                    merged = True
                    break
            if merged:
                break
    return comps


def fuse_vlm_cluster(ctx, out: Path, llm, crop_dir: Path) -> Path:
    groups_all = []
    for team in TEAM_ORDER:
        ids = [t for t in ctx["keep"] if ctx["team_of"].get(t, "O") == team]
        if not ids:
            continue
        mont, order = _montage(ids, crop_dir)
        if mont is None:
            groups_all += [[t] for t in ids]
            continue
        raw = _vlm_groups(llm, mont, order)
        # enforce temporal cannot-link, and never merge across confident numbers
        for g in raw:
            for sub in _split_temporal(g, ctx["frames"]):
                # split further if two confident different numbers landed together
                by_num = {}
                free = []
                for t in sub:
                    nn = ctx["num_of"].get(t)
                    (by_num.setdefault(nn, []) if nn is not None else free).append(t)
                if by_num:
                    parts = list(by_num.values())
                    parts[0] += free
                    groups_all += parts
                else:
                    groups_all.append(sub)
        print(f"[vlmcluster] team {team}: {len(ids)} tracks -> {len(raw)} VLM groups")
    return _emit(_name_clusters(groups_all, ctx), ctx, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare bucket vs VLM-cluster roster fusion.")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--teams", type=Path, required=True)
    ap.add_argument("--jersey", type=Path, default=None)
    ap.add_argument("--vlm-fp", type=Path, required=True)
    ap.add_argument("--crop-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/tracks"))
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--bucket-threshold", type=float, default=0.30)
    ap.add_argument("--approach", choices=["a", "b", "both"], default="both")
    ap.add_argument("--config", type=Path, default=Path("configs/veo.yaml"))
    args = ap.parse_args()

    ctx = load_context(args.cache, args.teams, args.jersey, args.vlm_fp, args.min_frames)
    print(f"dropped {len(ctx['dropped'])} noise tracks (<{args.min_frames}f): {sorted(ctx['dropped'])}")
    print(f"clustering {len(ctx['keep'])} tracks")

    if args.approach in ("a", "both"):
        print("== Approach A: bucketed attributes ==")
        fuse_buckets(ctx, args.out_dir / "identities_bucket_night.parquet", args.bucket_threshold)
    if args.approach in ("b", "both"):
        print("== Approach B: VLM-as-clusterer ==")
        import yaml
        from .llm import get_llm
        cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
        cfg.setdefault("llm", {})["enabled"] = True
        cfg["llm"].setdefault("uses", {})["fingerprint"] = True
        llm = get_llm(cfg)
        fuse_vlm_cluster(ctx, args.out_dir / "identities_vlmcluster_night.parquet", llm, args.crop_dir)


if __name__ == "__main__":
    main()
