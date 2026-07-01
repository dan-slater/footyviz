"""M4 participant filter — classify each tracklet PLAYER / REFEREE / SPECTATOR / OTHER.

Not every "person" detection is a player: people behind the advertising boards,
on the perimeter path, or in street clothes get tracked too and pollute the
identity roster. This stage LABELS every tracklet (it never silently drops them)
so the classification can be eyeballed and identity re-clustering can run on
players only.

Three fused signals:
  - GEOMETRY: feet position (bbox bottom) and whether the patch just below the
    feet is pitch-green. Feet high up the frame (near the boards) or on
    non-green perimeter => likely spectator. We have no homography, so "pitch"
    is approximated by a green mask under the feet (HSV) — noted, not exact.
  - CROWD DISTANCE (ball proxy): we have no ball track, so we use each tracklet's
    mean distance to the per-frame centroid of all detections. Play clusters
    around the ball; perimeter spectators sit far from that centre.
  - VLM: gemini asked directly whether the clearest crop is a player/referee/
    spectator. Strong signal, used as the lead.

Outputs a per-track table (label + reason + the raw signals) so every call is
auditable.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .identity import _sample_frames

GREEN_H = (30, 95)
GREEN_MIN_S = 40
GREEN_MIN_V = 35

_P_CLASS = (
    "This is a cropped image from a single-camera amateur soccer match video. "
    "Classify the main person: an on-pitch PLAYER (wearing team kit/bib), a "
    "REFEREE (distinct official kit), or a SPECTATOR/bystander (street clothes, "
    "standing off the field behind boards/fence), or OTHER if unclear. "
    'Return STRICT JSON: {"class": "player|referee|spectator|other", "reason": "<short>"}.'
)


def _vlm_classify(llm, crop, retries: int = 3) -> tuple[str, str]:
    import time
    last = "error", ""
    for attempt in range(retries):
        try:
            txt = llm._vision(_P_CLASS, [crop], want_json=True)
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                d = json.loads(m.group(0))
                cls = str(d.get("class", "other")).lower().strip()
                if cls in ("player", "referee", "spectator", "other"):
                    return cls, str(d.get("reason", ""))[:120]
            last = "other", (txt or "")[:80]
        except Exception as e:
            last = "error", repr(e)[:80]
            time.sleep(2 * (attempt + 1))  # transient rate-limit backoff (in-process)
    return last


def _green_frac(patch: np.ndarray) -> float:
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = (H >= GREEN_H[0]) & (H <= GREEN_H[1]) & (S >= GREEN_MIN_S) & (V >= GREEN_MIN_V)
    return float(g.mean())


def classify(video: Path, tracks: Path, teams: Path, vlm_fp: Path | None, out_csv: Path,
             llm, start_frame: int = 0, end_frame: int | None = None,
             samples: int = 12, crop_out: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(tracks)
    lo = int(start_frame)
    hi = int(df["frame"].max()) if end_frame is None else int(end_frame)
    df = df[(df["frame"] >= lo) & (df["frame"] <= hi)]
    H_frame = 720

    tm = pd.read_parquet(teams)
    team_of = dict(zip(tm["track_id"].astype(int), tm["team"]))

    # Crowd-centroid distance per detection (ball proxy).
    cx = (df["x1"] + df["x2"]) / 2
    cy = (df["y1"] + df["y2"]) / 2
    df = df.assign(cx=cx, cy=cy)
    cent = df.groupby("frame")[["cx", "cy"]].transform("mean")
    df = df.assign(cdist=np.hypot(df["cx"] - cent["cx"], df["cy"] - cent["cy"]))

    # Per-track geometry from boxes.
    geom = {}
    for tid, g in df.groupby("track_id"):
        feet_y = float(np.median(g["y2"]) / H_frame)
        motion = float(np.hypot(g["cx"].max() - g["cx"].min(), g["cy"].max() - g["cy"].min()))
        geom[int(tid)] = {"feet_y": feet_y, "cdist": float(g["cdist"].mean()),
                          "motion": motion, "n_frames": int(g["frame"].nunique())}

    # One decode pass: clearest crop + green-below-feet per track.
    needed = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        for f in _sample_frames(g["frame"].to_numpy(), samples):
            r = g[g["frame"] == f].iloc[0]
            needed.setdefault(int(f), []).append((int(tid), (float(r.x1), float(r.y1), float(r.x2), float(r.y2))))
    best_crop, green = {}, {}
    cap = cv2.VideoCapture(str(video))
    idx = -1
    while idx < hi:
        if not cap.grab():
            break
        idx += 1
        items = needed.get(idx)
        if items is None:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        Hf, Wf = frame.shape[:2]
        for tid, (x1, y1, x2, y2) in items:
            x1i, y1i, x2i, y2i = max(0, int(x1)), max(0, int(y1)), min(Wf, int(x2)), min(Hf, int(y2))
            if x2i - x1i < 5 or y2i - y1i < 10:
                continue
            bh = y2i - y1i
            below = frame[y2i:min(Hf, y2i + int(0.18 * bh)),
                          x1i + int(0.25 * (x2i - x1i)):x2i - int(0.25 * (x2i - x1i))]
            green.setdefault(tid, []).append(_green_frac(below))
            area = (x2i - x1i) * bh
            sharp = cv2.Laplacian(cv2.cvtColor(frame[y1i:y2i, x1i:x2i], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if tid not in best_crop or (area * sharp) > best_crop[tid][0]:
                best_crop[tid] = (area * sharp, frame[y1i:y2i, x1i:x2i].copy())
    cap.release()

    if crop_out:
        crop_out.mkdir(parents=True, exist_ok=True)

    # thresholds for the crowd-dist "far from action" flag
    cdists = np.array([geom[t]["cdist"] for t in geom])
    far_thr = float(np.percentile(cdists, 75)) if len(cdists) else 1e9

    rows = []
    tids = sorted(geom)
    for i, tid in enumerate(tids, 1):
        gm = geom[tid]
        gb = float(np.median(green.get(tid, [0.0])))
        crop = best_crop.get(tid, (0, None))[1]
        vlm_cls, vlm_reason = (_vlm_classify(llm, crop) if crop is not None else ("other", "no crop"))
        if crop_out and crop is not None:
            cv2.imwrite(str(crop_out / f"t{tid}.jpg"), crop)

        # geometry flags
        perimeter = gm["feet_y"] < 0.30 or gb < 0.20      # feet near boards or not on grass
        far = gm["cdist"] > far_thr
        stationary = gm["motion"] < 60

        # Fuse: VLM leads; geometry can override an ambiguous "player" to spectator.
        label, why = "PLAYER", []
        if vlm_cls == "referee":
            label, why = "REFEREE", [f"vlm=referee({vlm_reason})"]
        elif vlm_cls == "spectator":
            label, why = "SPECTATOR", [f"vlm=spectator({vlm_reason})"]
        elif vlm_cls == "player":
            label, why = "PLAYER", [f"vlm=player({vlm_reason})"]
            if perimeter and far and stationary:
                label, why = "SPECTATOR", why + ["but geometry: perimeter+far+stationary"]
        else:  # other/error/unknown -> decide by geometry
            if perimeter and (far or stationary):
                label, why = "SPECTATOR", [f"vlm={vlm_cls}", "geometry perimeter"]
            else:
                label, why = "PLAYER", [f"vlm={vlm_cls}", "geometry on-pitch"]

        # Flag the calls a human should double-check: VLM gave no clean answer
        # (decided by geometry), or VLM said off-pitch yet the feet are clearly on
        # grass near the action.
        uncertain = (vlm_cls in ("error", "other")
                     or (label == "SPECTATOR" and gb > 0.85 and not far))
        rows.append({"track_id": tid, "team": team_of.get(tid, "?"), "label": label,
                     "reason": "; ".join(why), "feet_y": round(gm["feet_y"], 3),
                     "green_below": round(gb, 2), "mean_ball_dist": round(gm["cdist"], 1),
                     "motion": round(gm["motion"], 1), "n_frames": gm["n_frames"],
                     "vlm_says": vlm_cls, "uncertain": bool(uncertain)})
        if i % 15 == 0:
            print(f"  classified {i}/{len(tids)}")

    out = pd.DataFrame(rows).sort_values(["label", "track_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[participant] wrote {out_csv}")
    print(out["label"].value_counts().to_dict())
    return out


def montage_by_class(df: pd.DataFrame, crop_dir: Path, out_dir: Path):
    for label in ["PLAYER", "REFEREE", "SPECTATOR", "OTHER"]:
        tids = df[df["label"] == label]["track_id"].tolist()
        cells = []
        for t in tids:
            p = crop_dir / f"t{t}.jpg"
            im = cv2.imread(str(p)) if p.exists() else None
            if im is None:
                continue
            im = cv2.resize(im, (90, 150))
            band = np.full((22, 90, 3), 20, np.uint8)
            cv2.putText(band, f"{t}", (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cells.append(np.vstack([band, im]))
        if not cells:
            continue
        cols = 8
        while len(cells) % cols:
            cells.append(np.full_like(cells[0], 40))
        rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
        cv2.imwrite(str(out_dir / f"participant_montage_{label.lower()}.jpg"), np.vstack(rows))
        print(f"[participant] montage {label}: {len(tids)} tracks")


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify tracklets PLAYER/REFEREE/SPECTATOR/OTHER.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--teams", type=Path, required=True)
    ap.add_argument("--vlm-fp", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=Path("data/out/participant_classification.csv"))
    ap.add_argument("--crop-out", type=Path, default=Path("data/tracks/participant_crops"))
    ap.add_argument("--montage-dir", type=Path, default=Path("data/out"))
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--config", type=Path, default=Path("configs/veo.yaml"))
    args = ap.parse_args()

    import yaml
    from .llm import get_llm
    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    cfg.setdefault("llm", {})["enabled"] = True
    cfg["llm"]["provider"] = "gemini"
    cfg["llm"]["model"] = args.model
    cfg["llm"].setdefault("uses", {})["fingerprint"] = True
    llm = get_llm(cfg)
    print(f"[participant] model = {llm.model}")

    df = classify(args.video, args.tracks, args.teams, args.vlm_fp, args.out_csv, llm,
                  args.start_frame, args.end_frame, crop_out=args.crop_out)
    montage_by_class(df, args.crop_out, args.montage_dir)


if __name__ == "__main__":
    main()
