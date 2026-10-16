"""Pre-labeller (BENCHMARK.md 3): keyframe boxes + proposed person IDs for the label page.

Not a leaderboard entry (anti-bias rule). Per clip:
  1. boxes   yolo11x @ imgsz 1280, conf 0.10 (fv.detect_track, ByteTrack) on every frame
  2. ids     BoT-SORT + OSNet tracklets over the whole clip (fv.reid_track)
  3. groups  kit-colour teams (fv.teams) + CV identity fusion (fv.identity) -> proposed people
  4. union   RT-DETR-x persons on keyframes, added where no tracked box overlaps (IoU < 0.5)
Keyframes every 90 frames (40 per clip). 3 keyframes per clip are BLIND: exported without
boxes, labelled from scratch for the pre-label bias estimate.

Writes intermediates to ~/footyviz-bench-data/prelabels/<id>/ and the label starting point to
~/footyviz-bench-data/labels/<id>/: frames/kf_<frame>.jpg, crops/t<track>.jpg, prelabel.json.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import manifest as M

STRIDE = 90
N_BLIND = 3
MODELS = M.DATA / "models"
YOLO_X = MODELS / "yolo11x.pt"
RTDETR_X = MODELS / "rtdetr-x.pt"
OSNET = M.REPO / "data" / "models" / "osnet_x1_0_msmt17.pt"
PRELABELLER = "yolo11x1280c0.10+rtdetrx(kf)/botsort-osnet/kmeans+cvfuse"


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def keyframes(n_frames: int) -> list[int]:
    return list(range(0, n_frames, STRIDE))


def blind_keyframes(clip_id: str, kfs: list[int]) -> list[int]:
    return sorted(random.Random(f"fvbench-blind-{clip_id}").sample(kfs[1:-1], N_BLIND))


def run_stages(clip: Path, work: Path, device: str = "cuda:0") -> dict:
    from .. import detect_track, identity, reid_track, teams
    work.mkdir(parents=True, exist_ok=True)
    t = {}
    det, reid = work / "det.parquet", work / "reid.parquet"
    tm, ident = work / "teams.parquet", work / "identities.parquet"
    s = time.time()
    if not det.exists():
        detect_track.run(clip, det, model_path=str(YOLO_X), device=device, conf=0.10,
                         ball=True, ball_conf=0.10, imgsz=1280)
    t["detect"] = time.time() - s
    s = time.time()
    if not reid.exists():
        reid_track.run(clip, det, reid, device=device, reid_weights=str(OSNET))
    t["reid"] = time.time() - s
    s = time.time()
    if not tm.exists():
        teams.run(clip, reid, tm)
    if not ident.exists():
        identity.run(clip, reid, ident, tm, device=device, cache=work / "identity_desc.npz",
                     reid_weights=str(OSNET))
    t["teams+identity"] = time.time() - s
    return {k: round(v, 1) for k, v in t.items()}


def rtdetr_keyframes(clip: Path, kfs: list[int], device: str = "cuda:0") -> dict[int, list]:
    from ultralytics import RTDETR
    model = RTDETR(str(RTDETR_X))
    cap = cv2.VideoCapture(str(clip))
    out = {}
    for f in kfs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        r = model(im, verbose=False, device=device, conf=0.30, classes=[0], imgsz=1280)[0]
        out[f] = [(list(map(float, b)), float(c)) for b, c in
                  zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    cap.release()
    return out


def build(x: dict, force: bool = False) -> Path:
    clip = M.data_path(x["clip_file"])
    if not clip.exists():
        raise SystemExit(f"[{x['id']}] no clip {clip}; run `fv.bench cut --clip {x['id']}` first")
    work = M.DATA / "prelabels" / x["id"]
    out = M.DATA / "labels" / x["id"]
    if force and work.exists():
        for p in work.glob("*"):
            p.unlink()
    timing = run_stages(clip, work)

    reid = pd.read_parquet(work / "reid.parquet")
    det = pd.read_parquet(work / "det.parquet")
    ball = pd.read_parquet(work / "det_ball.parquet")
    tm = pd.read_parquet(work / "teams.parquet")
    ident = pd.read_parquet(work / "identities.parquet")
    team_of = dict(zip(tm["track_id"].astype(int), tm["team"]))
    group_of = dict(zip(ident["raw_track_id"].astype(int), ident["stable_id"]))

    c = x["clip"]
    n = c["n_frames"]
    kfs = keyframes(n)
    blind = set(blind_keyframes(x["id"], kfs))
    s = time.time()
    rt = rtdetr_keyframes(clip, kfs)
    timing["rtdetr_kf"] = round(time.time() - s, 1)

    # keyframe boxes
    kf_rows = []
    by_reid = {f: g for f, g in reid[reid["frame"].isin(kfs)].groupby("frame")}
    by_det = {f: g for f, g in det[det["frame"].isin(kfs)].groupby("frame")}
    nb = 0
    for f in kfs:
        boxes = []
        for _, r in by_reid.get(f, pd.DataFrame()).iterrows():
            tid = int(r["track_id"])
            boxes.append({"box": [r.x1, r.y1, r.x2, r.y2], "conf": float(r.confidence),
                          "src": "track", "track": tid, "group": group_of.get(tid),
                          "team_guess": team_of.get(tid)})
        for _, r in by_det.get(f, pd.DataFrame()).iterrows():
            b = [r.x1, r.y1, r.x2, r.y2]
            if all(iou(b, o["box"]) < 0.5 for o in boxes):
                boxes.append({"box": b, "conf": float(r.confidence), "src": "yolo",
                              "track": None, "group": None, "team_guess": None})
        for b, cf in rt.get(f, []):
            if all(iou(b, o["box"]) < 0.5 for o in boxes):
                boxes.append({"box": b, "conf": cf, "src": "rtdetr", "track": None,
                              "group": None, "team_guess": None})
            else:
                for o in boxes:
                    if iou(b, o["box"]) >= 0.5 and o["src"] != "rtdetr":
                        o["rtdetr"] = True
                        break
        for o in boxes:
            o["box"] = [round(float(v), 1) for v in o["box"]]
            o["conf"] = round(o["conf"], 3)
            o["bid"] = f"b{nb}"
            nb += 1
        kf_rows.append({"frame": f, "src_s": round(c["start_src_s"] + f * eval(c["fps"]) ** -1, 3),
                        "blind": f in blind, "boxes": [] if f in blind else boxes,
                        "_hidden_boxes": len(boxes) if f in blind else 0})

    # tracklets: span + exemplar (largest confident box away from the frame edge)
    W, H = c["width"], c["height"]
    tracklets = {}
    kf_set = set(kfs)
    for tid, g in reid.groupby("track_id"):
        tid = int(tid)
        h = g.y2 - g.y1
        edge = (g.x1 < 5) | (g.y1 < 5) | (g.x2 > W - 5) | (g.y2 > H - 5)
        score = h * g.confidence * np.where(edge, 0.3, 1.0)
        e = g.loc[score.idxmax()]
        tracklets[str(tid)] = {
            "group": group_of.get(tid), "team_guess": team_of.get(tid),
            "first": int(g.frame.min()), "last": int(g.frame.max()), "n_frames": int(len(g)),
            "n_kf": int(g.frame.isin(kf_set).sum()),
            "median_h": round(float(h.median()), 1),
            "exemplar": {"frame": int(e.frame), "box": [round(float(v), 1) for v in (e.x1, e.y1, e.x2, e.y2)]},
        }
    groups: dict[str, dict] = {}
    for tid, t in tracklets.items():
        gid = t["group"] or f"t{tid}"
        g = groups.setdefault(gid, {"team_guess": t["team_guess"], "tracklets": [], "n_kf": 0})
        g["tracklets"].append(tid)
        g["n_kf"] += t["n_kf"]

    # images: keyframes + exemplar crops, one sequential pass
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    want_ex: dict[int, list[str]] = {}
    for tid, t in tracklets.items():
        want_ex.setdefault(t["exemplar"]["frame"], []).append(tid)
    cap = cv2.VideoCapture(str(clip))
    for f in range(n):
        ok = cap.grab()
        if not ok:
            break
        if f not in kf_set and f not in want_ex:
            continue
        ok, im = cap.retrieve()
        if f in kf_set:
            cv2.imwrite(str(out / "frames" / f"kf_{f:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        for tid in want_ex.get(f, []):
            x1, y1, x2, y2 = tracklets[tid]["exemplar"]["box"]
            pw, ph = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            crop = im[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))]
            if crop.size:
                cv2.imwrite(str(out / "crops" / f"t{tid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()

    # ball, 1 Hz (for the later L2 layer)
    fps = eval(c["fps"])
    ball_pts = []
    bb = ball.set_index("frame")
    for sec in range(int(n / fps)):
        f = int(round(sec * fps))
        if f in bb.index:
            r = bb.loc[f]
            ball_pts.append({"frame": f, "xy": [round((r.x1 + r.x2) / 2, 1), round((r.y1 + r.y2) / 2, 1)],
                             "conf": round(float(r.confidence), 3)})
        else:
            ball_pts.append({"frame": f, "xy": None, "conf": None})

    pl = {
        "schema": "fvbench.prelabel/1", "clip_id": x["id"], "video_id": x["video_id"],
        "start_s": x["label_window"]["start_s"], "end_s": x["label_window"]["end_s"],
        "start_src_s": c["start_src_s"], "fps": c["fps"], "n_frames": n, "width": W, "height": H,
        "stride": STRIDE, "blind_frames": sorted(blind), "kits": x.get("kits"),
        "prelabeller": PRELABELLER, "timing_s": timing, "date": time.strftime("%Y-%m-%d"),
        "keyframes": kf_rows, "tracklets": tracklets, "groups": groups, "ball_1hz": ball_pts,
    }
    (out / "prelabel.json").write_text(json.dumps(pl))
    nbox = sum(len(k["boxes"]) for k in kf_rows)
    print(f"[{x['id']}] {len(kfs)} keyframes ({len(blind)} blind {sorted(blind)}), {nbox} boxes, "
          f"{len(tracklets)} tracklets -> {len(groups)} proposed people; timing {timing}")
    print(f"[{x['id']}] -> {out}")
    return out


def main(ids: list[str], force: bool = False) -> int:
    m = M.load()
    for x in M.matches(m, ids=ids):
        build(x, force)
    return 0
