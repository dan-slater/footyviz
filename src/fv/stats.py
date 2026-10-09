"""Stats — possession proxy, distance, heatmaps and roster from tracks + teams.

Everything in `stats.json` (schema: API.md) is computed here from artefacts the
earlier stages already wrote:

    tracks     person tracks parquet (frame, track_id, x1..y2), frames 0-based in the clip
    teams      fv.teams output (track_id -> A/B/O + kit colour)
    ball       optional fv.detect_track `<tracks>_ball.parquet` (best ball box per frame)
    identities optional fv.identity output (raw_track_id -> stable_id) + its _table

Team-level numbers (possession, team distance, heatmaps) do not need stable
identity. Per-player numbers use the identity pass when present, else fall back
to raw tracker IDs (`roster_level: "track"`).

There is no pitch homography: positions are image coordinates of a camera that
pans. Distance removes the pan with a global phase-correlation estimate and
converts pixels to metres by an *assumed* visible pitch width — written into the
`method` text so the number is never mistaken for a measured one.

One sequential decode of the clip gives both the camera shift and the best
crop per roster entry.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GRID = (32, 18)                 # heatmap cols, rows
ASSUMED_VISIBLE_WIDTH_M = 25.0  # visible pitch width across the frame (5-a-side, Veo follow-cam)
MAX_SPEED_FW_PER_S = 0.5        # drop steps faster than half a frame-width per second
SMOOTH_WIN = 9                  # frames, centred rolling mean on foot points
POSSESSION_REACH_H = 3.0        # ball must be within this many player-heights of the owner
POSSESSION_HOLD_S = 1.0         # owner persists this long through ball-less frames
SHIFT_SCALE = 4                 # phase correlation runs at 1/4 resolution
CROP_CANDIDATES = 8             # frames considered per roster entry for its best crop

IDENTITY_NOTE = ("Identity is capped by the footage: this pipeline resolved {n} stable "
                 "identities; on the reference night slice it found ~23 stable vs ~14–16 "
                 "true players. Team-level stats do not depend on perfect identity.")

# Used when the identity pass was skipped: the roster is raw tracker IDs, not identities.
IDENTITY_NOTE_TRACKS = ("Identity pass skipped: the roster lists {n} raw tracker IDs, not players "
                        "(one player usually spans several). On the reference night slice the full "
                        "pipeline found ~23 stable vs ~14–16 true players. Team-level stats do not "
                        "depend on perfect identity.")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def colour_name(rgb) -> str:
    """Plain-English name for a kit colour (good enough for a legend)."""
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hd = h * 360
    if v < 0.22:
        return "Black"
    if s < 0.16:
        return "White" if v > 0.62 else "Grey"
    if (hd >= 290 or hd < 15) and s < 0.55 and v > 0.5:
        return "Pink"
    for lim, name in ((15, "Red"), (40, "Orange"), (68, "Yellow"), (165, "Green"),
                      (200, "Cyan"), (255, "Blue"), (290, "Purple"), (335, "Pink"), (361, "Red")):
        if hd < lim:
            return name
    return "Red"


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def _safe(sid: str) -> str:
    return sid.replace("#", "n").replace("/", "_")


def _sharpness(crop: np.ndarray) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# roster
# --------------------------------------------------------------------------- #
def _roster_groups(df: pd.DataFrame, team_of: dict, identities: Path | None, fps: float):
    """Return (level, [{id, team, jersey, tracks:[...]}]) — one entry per roster member."""
    if identities is not None and Path(identities).exists():
        idf = pd.read_parquet(identities)
        groups: dict[str, dict] = {}
        for _, r in idf.iterrows():
            sid = str(r["stable_id"])
            g = groups.setdefault(sid, {"id": sid, "team": str(r["team"]),
                                        "jersey": None, "tracks": []})
            g["tracks"].append(int(r["raw_track_id"]))
            if int(r.get("number", -1)) >= 0:
                g["jersey"] = str(int(r["number"]))
        return "identity", list(groups.values())
    # Fallback: raw tracker IDs seen for at least a second, longest 30.
    counts = df.groupby("track_id").size()
    keep = counts[counts >= max(int(fps), 1)].sort_values(ascending=False).head(30)
    return "track", [{"id": f"{team_of.get(int(t), 'O')}-t{int(t)}",
                      "team": team_of.get(int(t), "O"), "jersey": None, "tracks": [int(t)]}
                     for t in keep.index]


def _crop_requests(df: pd.DataFrame, roster: list[dict]) -> dict[int, list]:
    """frame -> [(roster_idx, box)] for the tallest boxes of each roster entry."""
    req: dict[int, list] = {}
    for ri, g in enumerate(roster):
        rows = df[df["track_id"].isin(g["tracks"])]
        if not len(rows):
            continue
        h = rows["y2"] - rows["y1"]
        top = rows.loc[h.sort_values(ascending=False).index[: CROP_CANDIDATES * 4]]
        top = top.iloc[np.linspace(0, len(top) - 1, min(CROP_CANDIDATES, len(top))).astype(int)]
        for _, r in top.iterrows():
            req.setdefault(int(r.frame), []).append((ri, (r.x1, r.y1, r.x2, r.y2)))
    return req


# --------------------------------------------------------------------------- #
# the one decode pass: camera shift + crops
# --------------------------------------------------------------------------- #
def scan_video(video: Path, n_frames: int, crop_req: dict[int, list]):
    """Return (cum_shift Nx2 in full-res px, best_crop per roster idx)."""
    cap = cv2.VideoCapture(str(video))
    shift = np.zeros((n_frames, 2), dtype=float)
    prev = win = None
    best: dict[int, tuple[float, np.ndarray]] = {}
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            shift = shift[:i] if i else shift[:1]
            break
        H, W = frame.shape[:2]
        small = cv2.cvtColor(cv2.resize(frame, (W // SHIFT_SCALE, H // SHIFT_SCALE),
                                        interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is None:
            win = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        else:
            (dx, dy), resp = cv2.phaseCorrelate(prev, small, win)
            dx, dy = dx * SHIFT_SCALE, dy * SHIFT_SCALE
            if resp < 0.02 or abs(dx) > 0.2 * W or abs(dy) > 0.2 * H:
                dx = dy = 0.0                              # unreliable -> assume still
            shift[i] = shift[i - 1] + (dx, dy)
        prev = small
        for ri, (x1, y1, x2, y2) in crop_req.get(i, []):
            pw, ph = (x2 - x1) * 0.12, (y2 - y1) * 0.06
            a, b = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            c, d = int(min(W, x2 + pw)), int(min(H, y2 + ph))
            if c - a < 6 or d - b < 12:
                continue
            crop = frame[b:d, a:c].copy()
            score = _sharpness(crop) * (d - b)              # sharp AND large
            if ri not in best or score > best[ri][0]:
                best[ri] = (score, crop)
    cap.release()
    return shift, {k: v[1] for k, v in best.items()}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def track_distances(df: pd.DataFrame, shift: np.ndarray, W: int, fps: float) -> dict[int, float]:
    """Pan-compensated, smoothed foot-point path length per track (pixels)."""
    n = len(shift)
    max_step = MAX_SPEED_FW_PER_S * W / fps
    out: dict[int, float] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy(int)
        f_c = np.clip(f, 0, n - 1)
        fx = (g["x1"].to_numpy() + g["x2"].to_numpy()) / 2 - shift[f_c, 0]
        fy = g["y2"].to_numpy() - shift[f_c, 1]
        seg = np.concatenate(([0], np.cumsum(np.diff(f) > 2)))   # split on gaps
        total = 0.0
        for s in np.unique(seg):
            m = seg == s
            if m.sum() < 3:
                continue
            xs = pd.Series(fx[m]).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()
            ys = pd.Series(fy[m]).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()
            df_ = np.diff(f[m]).clip(min=1)
            step = np.hypot(np.diff(xs), np.diff(ys))
            ok = step / df_ <= max_step
            total += float(step[ok].sum())
        out[int(tid)] = total
    return out


def heatmap(df: pd.DataFrame, W: int, H: int) -> list[list[float]]:
    cols, rows = GRID
    fx = ((df["x1"] + df["x2"]) / 2).to_numpy()
    fy = df["y2"].to_numpy()
    hm, _, _ = np.histogram2d(fy, fx, bins=[rows, cols], range=[[0, H], [0, W]])
    if hm.max() > 0:
        hm = hm / hm.max()
    return [[round(float(v), 4) for v in row] for row in hm]


def possession(df: pd.DataFrame, team_of: dict, ball: pd.DataFrame | None,
               n_frames: int, fps: float) -> dict:
    counts = {"A": 0, "B": 0, "unresolved": 0}
    ball_by_frame = {}
    if ball is not None and len(ball):
        med_h = float((df["y2"] - df["y1"]).median()) if len(df) else 40.0
        bw = ball["x2"] - ball["x1"]
        ball = ball[bw <= 0.6 * med_h]                     # a "ball" as big as a person is not one
        ball_by_frame = {int(r.frame): ((r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2)
                         for r in ball.itertuples()}
    by_frame = {int(f): g for f, g in df.groupby("frame")}
    hold = int(round(POSSESSION_HOLD_S * fps))
    owner, owner_age = None, 0
    for f in range(n_frames):
        b = ball_by_frame.get(f)
        g = by_frame.get(f)
        if b is not None and g is not None and len(g):
            fx = ((g["x1"] + g["x2"]) / 2).to_numpy()
            fy = g["y2"].to_numpy()
            hgt = (g["y2"] - g["y1"]).to_numpy()
            d = np.hypot(fx - b[0], fy - b[1])
            k = int(d.argmin())
            team = team_of.get(int(g["track_id"].iloc[k]), "O")
            if d[k] <= POSSESSION_REACH_H * hgt[k] and team in ("A", "B"):
                owner, owner_age = team, 0
            else:
                owner = None
        elif owner is not None:
            owner_age += 1
            if owner_age > hold:
                owner = None
        counts[owner if owner else "unresolved"] += 1
    tot = max(n_frames, 1)
    ball_frac = len([f for f in ball_by_frame if f < n_frames]) / tot
    return {
        "A": round(counts["A"] / tot, 4), "B": round(counts["B"] / tot, 4),
        "unresolved": round(counts["unresolved"] / tot, 4),
        "ball_frames_frac": round(ball_frac, 4),
        "method": (
            "Proxy: in each frame with a ball detection (YOLO COCO 'sports ball'), the team "
            "of the nearest player (foot point) within "
            f"{POSSESSION_REACH_H:g} player-heights owns the ball; frames without a ball "
            f"inherit the last owner for up to {POSSESSION_HOLD_S:g} s, otherwise count as "
            "unresolved. Ball nearest a goalkeeper/other-kit player counts as unresolved. "
            f"Ball detected in {ball_frac:.0%} of frames."),
    }


def roster_sheet(crops: list[tuple[dict, np.ndarray | None]], team_rgb: dict, out: Path,
                 tile_h: int = 180, per_row: int = 10):
    """Grid of best crops grouped by team, label strip under each tile."""
    tile_w, strip = tile_h // 2, 22
    rows_img = []
    for team in ("A", "B", "O"):
        items = [(g, c) for g, c in crops if g["team"] == team]
        if not items:
            continue
        rgb = team_rgb.get(team, (150, 150, 150))
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        for start in range(0, len(items), per_row):
            row = np.full((tile_h + strip, per_row * tile_w, 3), 24, np.uint8)
            for j, (g, c) in enumerate(items[start:start + per_row]):
                x0 = j * tile_w
                if c is not None:
                    s = min(tile_w / c.shape[1], tile_h / c.shape[0])
                    rc = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))),
                                    interpolation=cv2.INTER_CUBIC)
                    oy, ox = (tile_h - rc.shape[0]) // 2, (tile_w - rc.shape[1]) // 2
                    row[oy:oy + rc.shape[0], x0 + ox:x0 + ox + rc.shape[1]] = rc
                cv2.rectangle(row, (x0, tile_h), (x0 + tile_w - 1, tile_h + strip - 1), bgr, -1)
                lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
                tc = (20, 20, 20) if lum > 140 else (255, 255, 255)
                cv2.putText(row, g["id"], (x0 + 4, tile_h + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, tc, 1, cv2.LINE_AA)
                cv2.line(row, (x0 + tile_w - 1, 0), (x0 + tile_w - 1, tile_h + strip), (60, 60, 60), 1)
            rows_img.append(row)
    if not rows_img:
        rows_img = [np.full((tile_h + strip, per_row * tile_w, 3), 24, np.uint8)]
    sheet = np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 60, np.uint8)]) for r in rows_img])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def compute(video: Path, tracks: Path, teams: Path, out_dir: Path, *,
            ball: Path | None = None, identities: Path | None = None,
            source: dict | None = None, events: list | None = None,
            pipeline: dict | None = None, file_prefix: str = "",
            assumed_width_m: float = ASSUMED_VISIBLE_WIDTH_M) -> dict:
    """Write stats.json, roster.jpg and crop_<id>.jpg into out_dir; return the stats dict.

    file_prefix is prepended to file references inside stats.json (the API
    passes "/jobs/<id>/files/")."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    df = pd.read_parquet(tracks)
    df = df[df["class_id"] == 0] if "class_id" in df else df
    n_frames = min(total, int(df["frame"].max()) + 1) if len(df) else total

    tdf = pd.read_parquet(teams)
    team_of = dict(zip(tdf["track_id"].astype(int), tdf["team"]))
    team_rgb = {}
    for team in ("A", "B", "O"):
        r = tdf[tdf["team"] == team]
        if len(r):
            team_rgb[team] = (int(r.iloc[0].color_r), int(r.iloc[0].color_g), int(r.iloc[0].color_b))

    level, roster = _roster_groups(df, team_of, identities, fps)
    shift, crops = scan_video(video, n_frames, _crop_requests(df, roster))
    for i, g in enumerate(roster):
        g["crop_img"] = crops.get(i)
    if len(shift) < n_frames:
        shift = np.vstack([shift, np.repeat(shift[-1:], n_frames - len(shift), axis=0)])

    # --- distance ---
    px_per_m = W / assumed_width_m
    dist = track_distances(df, shift, W, fps)
    team_px = {t: 0.0 for t in ("A", "B")}
    for tid, d in dist.items():
        t = team_of.get(tid, "O")
        if t in team_px:
            team_px[t] += d
    frames_of = df.groupby("track_id")["frame"].apply(set).to_dict()
    players = []
    for g in roster:
        px = sum(dist.get(t, 0.0) for t in g["tracks"])
        fr = set().union(*[frames_of.get(t, set()) for t in g["tracks"]])
        g["frames"] = len(fr)
        players.append({"id": g["id"], "team": g["team"], "px": round(px, 1),
                        "m": round(px / px_per_m, 1), "frames": len(fr)})
    players.sort(key=lambda p: -p["m"])

    # --- roster files ---
    roster.sort(key=lambda g: ({"A": 0, "B": 1}.get(g["team"], 2), -g["frames"]))
    roster_out, sheet_items = [], []
    for g in roster:
        crop = g.pop("crop_img")
        name = f"crop_{_safe(g['id'])}.jpg"
        if crop is not None:
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        roster_out.append({"id": g["id"], "team": g["team"], "frames": g["frames"],
                           "crop": file_prefix + name if crop is not None else None,
                           "jersey": g["jersey"]})
        sheet_items.append((g, crop))
    roster_sheet(sheet_items, team_rgb, out_dir / "roster.jpg")

    ball_df = pd.read_parquet(ball) if ball is not None and Path(ball).exists() else None
    team_df = {t: df[df["track_id"].map(lambda x: team_of.get(int(x), "O")) == t] for t in ("A", "B")}

    stats = {
        "schema_version": 1,
        "source": {**(source or {}), "fps": round(float(fps), 3), "width": W, "height": H,
                   "frames": int(n_frames)},
        "teams": {t: {"label": colour_name(team_rgb[t]), "color": _hex(team_rgb[t]),
                      "tracks": int(sum(1 for v in team_of.values() if v == t))}
                  for t in ("A", "B") if t in team_rgb},
        "possession": possession(df, team_of, ball_df, n_frames, fps),
        "distance": {
            "method": (
                "No homography. Player foot-point tracks in image pixels, camera pan removed "
                "by global phase correlation, smoothed "
                f"({SMOOTH_WIN}-frame mean), steps faster than {MAX_SPEED_FW_PER_S:g} "
                f"frame-widths/s dropped. Metres = pixels / (frame width / {assumed_width_m:g} m), "
                f"i.e. assumes the visible pitch width is {assumed_width_m:g} m and ignores "
                "zoom and perspective. Expect ±50%."),
            "assumed_visible_width_m": assumed_width_m,
            "teams": {t: {"px": round(v, 1), "m": round(v / px_per_m, 1)} for t, v in team_px.items()},
            "players": players,
        },
        "heatmaps": {"grid": list(GRID), "space": "image",
                     **{t: heatmap(team_df[t], W, H) for t in ("A", "B")}},
        "roster_level": level,
        "roster": roster_out,
        "identity_note": (IDENTITY_NOTE.format(n=len(roster_out)) if level == "identity" else
                          IDENTITY_NOTE_TRACKS.format(n=len(roster_out))),
        "events": events or [],
        "pipeline": pipeline or {},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute stats.json (+ roster images) from tracks.")
    ap.add_argument("video", type=Path, help="the clip the tracks were computed on")
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--teams", type=Path, required=True)
    ap.add_argument("--ball", type=Path, default=None)
    ap.add_argument("--identities", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/out/stats"))
    ap.add_argument("--assumed-width-m", type=float, default=ASSUMED_VISIBLE_WIDTH_M)
    args = ap.parse_args()
    s = compute(args.video, args.tracks, args.teams, args.out_dir, ball=args.ball,
                identities=args.identities, assumed_width_m=args.assumed_width_m)
    print(json.dumps({k: s[k] for k in ("teams", "possession")}, indent=1))
    print("distance teams:", s["distance"]["teams"], "| roster:", len(s["roster"]), s["roster_level"])


if __name__ == "__main__":
    main()
