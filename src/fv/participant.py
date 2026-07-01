"""M4 participant filter — classify each tracklet and MARK it (never drop).

Categories: OUTFIELD_PLAYER | GOALKEEPER | REFEREE | SPECTATOR. Every tracklet is
labelled and written out with its evidence so a human can verify; identity
re-clustering then runs on the participants (outfield + keeper) only.

Two lessons baked in:
  - Crops are pulled from the 1080p source (bbox coords scaled from the 720p
    tracker by `scale`, default 1.5) so the VLM sees 2.25x the pixels.
  - The VLM gets GEOMETRY as a hard prior (feet-on-pitch via a green mask under
    the feet, motion, distance-to-crowd-centroid) and is told that dark/plain
    kits are normal — earlier it hallucinated "spectator" for on-pitch keepers
    and dark-kit players. If feet are on the green and the person moves with
    play, they are a PARTICIPANT, never a spectator.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .identity import _sample_frames

GREEN_H = (30, 95)
GREEN_MIN_S = 40
GREEN_MIN_V = 35

_P_CLASS = (
    "You are analysing a crop from DARK, low-resolution amateur NIGHT 5-a-side "
    "football. Dark, plain or casual-LOOKING kits are NORMAL for players here — "
    "do NOT infer 'spectator' from dark or casual clothing.\n"
    "Classify the main person as exactly one of: OUTFIELD_PLAYER, GOALKEEPER, "
    "REFEREE, SPECTATOR.\n"
    "HARD RULES:\n"
    "- If their feet are ON the green pitch AND they move with play, they are a "
    "PARTICIPANT (OUTFIELD_PLAYER / GOALKEEPER / REFEREE) — NEVER a spectator.\n"
    "- Label SPECTATOR only if clearly BEYOND the boundary: feet OFF the green, "
    "standing behind the advertising boards / fence.\n"
    "- GOALKEEPER: a participant who stays in/near a goal mouth, rarely goes to "
    "midfield, often a distinct kit.\n"
    "- REFEREE: on-pitch, moves with play, kit matches NEITHER team.\n"
    "Use the geometry facts given. Return STRICT JSON: "
    '{"class": "OUTFIELD_PLAYER|GOALKEEPER|REFEREE|SPECTATOR", "reason": "<short>"}.'
)

_VALID = {"outfield_player", "goalkeeper", "referee", "spectator"}


def _green_frac(patch: np.ndarray) -> float:
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = (H >= GREEN_H[0]) & (H <= GREEN_H[1]) & (S >= GREEN_MIN_S) & (V >= GREEN_MIN_V)
    return float(g.mean())


def extract_hires_crops(video: Path, df: pd.DataFrame, lo: int, hi: int,
                        scale: float = 1.5, per_track: int = 3, samples: int = 14,
                        min_h: int = 60):
    """Decode the hi-res video once; per track return best crops + green-below-feet.

    bbox coords come from the (720p) tracker and are multiplied by `scale` to index
    the 1080p frame. Returns (crops: tid->list[BGR], green: tid->median float)."""
    needed: dict[int, list[tuple]] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        for f in _sample_frames(g["frame"].to_numpy(), samples):
            r = g[g["frame"] == f].iloc[0]
            needed.setdefault(int(f), []).append(
                (int(tid), (r.x1 * scale, r.y1 * scale, r.x2 * scale, r.y2 * scale)))
    cand: dict[int, list[tuple]] = {}
    green: dict[int, list[float]] = {}
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
            if y2i - y1i < min_h or x2i - x1i < int(min_h * 0.3):
                continue
            bh = y2i - y1i
            below = frame[y2i:min(Hf, y2i + int(0.18 * bh)),
                          x1i + int(0.25 * (x2i - x1i)):x2i - int(0.25 * (x2i - x1i))]
            green.setdefault(tid, []).append(_green_frac(below))
            crop = frame[y1i:y2i, x1i:x2i]
            sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            cand.setdefault(tid, []).append(((x2i - x1i) * bh, sharp, crop.copy()))
    cap.release()
    crops = {}
    for tid, lst in cand.items():
        lst.sort(key=lambda t: t[0], reverse=True)            # largest first
        top = lst[: max(per_track * 2, per_track)]
        top.sort(key=lambda t: t[1], reverse=True)            # then sharpest
        crops[tid] = [c for _, _, c in top[:per_track]]
    green_med = {t: float(np.median(v)) for t, v in green.items()}
    return crops, green_med


def _vlm_classify(llm, crop, hint: str, retries: int = 3) -> tuple[str, str]:
    last = "error", ""
    for attempt in range(retries):
        try:
            txt = llm._vision(_P_CLASS + "\n" + hint, [crop], want_json=True)
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                d = json.loads(m.group(0))
                cls = str(d.get("class", "")).lower().strip()
                if cls in _VALID:
                    return cls, str(d.get("reason", ""))[:120]
            last = "other", (txt or "")[:80]
        except Exception as e:
            last = "error", repr(e)[:80]
            time.sleep(2 * (attempt + 1))
    return last


# category -> (mark label, role)
_CAT_MAP = {
    "outfield_player": ("PLAYER", "OUTFIELD"),
    "goalkeeper": ("GK", "GK"),
    "referee": ("REF", "REF"),
    "spectator": ("SPECTATOR", "SPECTATOR"),
}


def classify(video_hires: Path, tracks: Path, teams: Path, out_csv: Path, llm,
             start_frame: int = 0, end_frame: int | None = None, scale: float = 1.5,
             crop_out: Path | None = None):
    df = pd.read_parquet(tracks)
    lo = int(start_frame)
    hi = int(df["frame"].max()) if end_frame is None else int(end_frame)
    df = df[(df["frame"] >= lo) & (df["frame"] <= hi)]
    H_frame = 720

    tm = pd.read_parquet(teams)
    team_of = dict(zip(tm["track_id"].astype(int), tm["team"]))

    cx = (df["x1"] + df["x2"]) / 2
    cy = (df["y1"] + df["y2"]) / 2
    df = df.assign(cx=cx, cy=cy)
    cent = df.groupby("frame")[["cx", "cy"]].transform("mean")
    df = df.assign(cdist=np.hypot(df["cx"] - cent["cx"], df["cy"] - cent["cy"]))

    geom = {}
    for tid, g in df.groupby("track_id"):
        geom[int(tid)] = {
            "feet_y": float(np.median(g["y2"]) / H_frame),
            "cdist": float(g["cdist"].mean()),
            "motion": float(np.hypot(g["cx"].max() - g["cx"].min(), g["cy"].max() - g["cy"].min())),
            "n_frames": int(g["frame"].nunique()),
        }

    crops, green = extract_hires_crops(video_hires, df, lo, hi, scale, per_track=3)
    if crop_out:
        crop_out.mkdir(parents=True, exist_ok=True)

    rows = []
    tids = sorted(geom)
    for i, tid in enumerate(tids, 1):
        gm = geom[tid]
        gb = float(green.get(tid, 0.0))
        cs = crops.get(tid, [])
        crop = cs[0] if cs else None
        if crop_out and crop is not None:
            cv2.imwrite(str(crop_out / f"t{tid}.jpg"), crop)

        feet_on_pitch = gb >= 0.5
        moves = gm["motion"] > 150
        hint = (f"Geometry for this person: feet_on_pitch={'yes' if feet_on_pitch else 'no'} "
                f"(green_below={gb:.2f}), motion={gm['motion']:.0f}px, "
                f"mean_ball_dist={gm['cdist']:.0f}px.")
        vlm_cls, reason = (_vlm_classify(llm, crop, hint) if crop is not None else ("error", "no crop"))

        label, role = _CAT_MAP.get(vlm_cls, ("PLAYER", "OUTFIELD"))
        why = [f"vlm={vlm_cls}({reason})"]
        # HARD rule: on-pitch + moving => participant, override a wrong 'spectator'.
        if label == "SPECTATOR" and feet_on_pitch and moves:
            label, role = "PLAYER", "OUTFIELD"
            why.append("OVERRIDE: feet_on_pitch+moving -> participant")
        if vlm_cls in ("error", "other"):
            label, role = ("PLAYER", "OUTFIELD") if feet_on_pitch else ("SPECTATOR", "SPECTATOR")
            why.append("vlm unclear -> geometry")

        uncertain = (vlm_cls in ("error", "other")
                     or (label == "SPECTATOR" and feet_on_pitch)
                     or (label == "REF"))
        rows.append({"track_id": tid, "team": team_of.get(tid, "?"), "label": label, "role": role,
                     "reason": "; ".join(why), "feet_on_pitch": feet_on_pitch,
                     "feet_y": round(gm["feet_y"], 3), "green_below": round(gb, 2),
                     "mean_ball_dist": round(gm["cdist"], 1), "motion": round(gm["motion"], 1),
                     "n_frames": gm["n_frames"], "vlm_says": vlm_cls, "uncertain": bool(uncertain)})
        if i % 15 == 0:
            print(f"  classified {i}/{len(tids)}")

    out = pd.DataFrame(rows).sort_values(["label", "track_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[participant] wrote {out_csv}")
    print("labels:", out["label"].value_counts().to_dict())
    return out


def montage_by_class(df: pd.DataFrame, crop_dir: Path, out_dir: Path):
    for label in ["PLAYER", "GK", "REF", "SPECTATOR"]:
        tids = df[df["label"] == label]["track_id"].tolist()
        cells = []
        for t in tids:
            p = crop_dir / f"t{t}.jpg"
            im = cv2.imread(str(p)) if p.exists() else None
            if im is None:
                continue
            im = cv2.resize(im, (110, 180))
            band = np.full((24, 110, 3), 20, np.uint8)
            cv2.putText(band, f"{t}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
    ap = argparse.ArgumentParser(description="Classify tracklets (1080p crops + geometry-primed VLM).")
    ap.add_argument("video_hires", type=Path, help="1080p source video")
    ap.add_argument("tracks", type=Path, help="720p reid tracks parquet")
    ap.add_argument("--teams", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=Path("data/out/participant_classification.csv"))
    ap.add_argument("--crop-out", type=Path, default=Path("data/tracks/participant_crops_1080"))
    ap.add_argument("--montage-dir", type=Path, default=Path("data/out"))
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.5)
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
    print(f"[participant] model = {llm.model}, scale = {args.scale}")

    df = classify(args.video_hires, args.tracks, args.teams, args.out_csv, llm,
                  args.start_frame, args.end_frame, args.scale, crop_out=args.crop_out)
    montage_by_class(df, args.crop_out, args.montage_dir)


if __name__ == "__main__":
    main()
