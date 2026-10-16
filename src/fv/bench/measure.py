"""Image-tier measurements per match (BENCHMARK.md 0.1, 8.1 #7).

Samples the labelled 120 s window every 4 s (30 frames), drops the bottom watermark strip,
and measures mean luma, RMS contrast, Laplacian variance (sharpness), HSV saturation and Lab
cast. Writes `quality` + `image_tier` into bench/manifest.json and prints the table.

Image tier bands (fixed here so they are versioned with the code):
    low   luma < 55  or  laplacian_var < 250     (night / dark / soft)
    mid   luma < 90                              (shadow-striped or dim daylight)
    high  otherwise
"""
from __future__ import annotations

import cv2
import numpy as np

from . import manifest as M

STEP_S = 4.0
STRIP = 0.08  # bottom fraction excluded (veo watermark + scoreboard-free footer)
BANDS = {"luma_low": 55.0, "lap_low": 250.0, "luma_mid": 90.0}


def image_tier(luma: float, lap: float) -> str:
    if luma < BANDS["luma_low"] or lap < BANDS["lap_low"]:
        return "low"
    if luma < BANDS["luma_mid"]:
        return "mid"
    return "high"


def frame_stats(bgr: np.ndarray) -> dict:
    h = bgr.shape[0]
    img = bgr[: int(h * (1 - STRIP))]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return {
        "luma": float(g.mean()),
        "rms_contrast": float(g.std()),
        "laplacian_var": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "saturation": float(hsv[..., 1].mean()),
        "lab_a": float(lab[..., 1].mean() - 128),
        "lab_b": float(lab[..., 2].mean() - 128),
    }


def measure(x: dict) -> dict | None:
    src = M.data_path(x["source"]["file"])
    t0 = x["source"].get("src_t0")
    if not src.exists() or t0 is None:
        return None
    cap = cv2.VideoCapture(str(src))
    w = x["label_window"]
    rows = []
    t = w["start_s"]
    while t < w["end_s"] - 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, (t - t0) * 1000.0)
        ok, fr = cap.read()
        if ok:
            rows.append(frame_stats(fr))
        t += STEP_S
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not rows:
        return None
    q = {k: round(float(np.mean([r[k] for r in rows])), 1) for k in rows[0]}
    dur = nfr / fps if fps else None
    q["bitrate_mbps"] = round(src.stat().st_size * 8 / dur / 1e6, 2) if dur else None
    q["n_samples"] = len(rows)
    q["method"] = f"every {STEP_S:g} s over label_window, bottom {STRIP:.0%} excluded, full res"
    return q


def main(ids: list[str] | None = None, write: bool = True) -> int:
    m = M.load()
    hdr = f"{'id':3} {'split':5} {'eye':3} {'kit':5} {'luma':>6} {'rms':>5} {'lap':>6} {'sat':>5} {'a':>5} {'b':>5} {'Mb/s':>5}  image_tier"
    print(hdr)
    for x in M.matches(m, ids=ids):
        q = measure(x)
        if q is None:
            print(f"{x['id']:3} {x['split']:5} {x['eyeball_tier']:3} {str(x['kit_tier']):5}  -- source missing ({x['source']['file']})")
            continue
        x["quality"] = q
        x["image_tier"] = image_tier(q["luma"], q["laplacian_var"])
        print(f"{x['id']:3} {x['split']:5} {x['eyeball_tier']:3} {str(x['kit_tier']):5} {q['luma']:6.1f} {q['rms_contrast']:5.1f} "
              f"{q['laplacian_var']:6.0f} {q['saturation']:5.1f} {q['lab_a']:5.1f} {q['lab_b']:5.1f} {q['bitrate_mbps'] or 0:5.2f}  {x['image_tier']}")
    if write:
        m.setdefault("conventions", {})["image_tier_bands"] = BANDS
        M.save(m)
    return 0
