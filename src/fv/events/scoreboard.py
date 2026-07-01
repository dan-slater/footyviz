"""Physical-scoreboard OCR probe (M3 event detection).

Goal: detect goals by reading the venue's PHYSICAL scoreboard and watching the
score change (change time == goal time).

FEASIBILITY PROBE RESULT (2026-07-01, video hhaWMwLlugE @ 1080p)
----------------------------------------------------------------
NO physical scoreboard is visible anywhere in the camera's field of view.

The footage is a Veo auto-panning single-camera feed of a small caged 5-a-side
pitch (Cape Town, evening/floodlit throughout -- there is no real daylight
segment; even t=2s is dusk). The camera is mounted on one touchline and only
ever pans across:
  * the FAR fence, carrying a fixed row of advertising boards:
    Cape Town City FC, The Greek (Fisherman), Gelato Mania, City Grill
    (Steakhouse), STILES (Tiles & Sanware), Merchant Factors, Aylett (a backlit
    lightbox with two pint glasses), Mike's Sports;
  * the two goal ends (short sides).
The near touchline -- where a manual/electronic scoreboard is most likely
mounted at such venues (by the entrance/dugout) -- is BEHIND the camera and is
never imaged. A Veo pan camera physically cannot show behind itself.

The "small lit fixture near top-center above the goal" flagged in earlier night
frames is the **Aylett advertising lightbox**, not a scoreboard (confirmed by
6x crops at t=90s and t=1200s: reads "Aylett" over two illuminated pint
glasses).

Pan positions sampled (seconds): 2, 20, 45, 78, 90, 156, 234, 300, 390, 468,
546, 600, 702, 780, 858, 900, 1014, 1092, 1170, 1200, 1326, 1404, 1500, 1560,
1638, 1700, 1794, 1872, 1900, 1950, 2028, 2050, 2106, 2184, 2200, 2262, 2300,
2340, 2360.  Evidence frames: data/out/sb_probe_full_*.jpg,
data/out/sb_contact_sheet.jpg, data/out/scoreboard_crop_*.jpg.

CONCLUSION: scoreboard-OCR goal detection is NOT possible from this feed as-is.
Fall back to the M3 audio-cue + ball-near-goal approach (the 1080p re-download
now carries an AAC stereo audio track), or obtain the board's location if one
exists on the near side (it will require a different camera angle).

--------------------------------------------------------------------------------
The code below is a reusable probe: once a board IS located (in this or another
video), pass its pixel ROI and it will dump enhanced crops and optionally run
EasyOCR (GPU). It intentionally writes only to data/out/ and touches no other
module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _ffmpeg() -> str:
    """Path to imageio-ffmpeg's bundled binary (this box has no system ffmpeg)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def grab_frame(video: Path, t_sec: float) -> np.ndarray:
    """Grab a single BGR frame at t_sec via OpenCV (msec seek)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame at {t_sec}s from {video}")
    return frame


def enhance(crop: np.ndarray, scale: int = 4) -> dict[str, np.ndarray]:
    """Return several enhanced variants of a small board crop for OCR/eyeballing."""
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"up": up, "gray": gray, "clahe": clahe, "otsu": otsu, "otsu_inv": 255 - otsu}


def probe(
    video: Path,
    times: list[float],
    roi: tuple[int, int, int, int] | None,
    out_dir: Path,
    scale: int,
    do_ocr: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = None
    if do_ocr:
        import easyocr  # heavy import; only when requested

        reader = easyocr.Reader(["en"], gpu=True)

    for t in times:
        frame = grab_frame(video, t)
        tag = f"{int(round(t)):04d}s"
        cv2.imwrite(str(out_dir / f"sb_probe_full_{tag}.jpg"), frame)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi
        crop = frame[y0:y1, x0:x1]
        variants = enhance(crop, scale)
        for name, img in variants.items():
            cv2.imwrite(str(out_dir / f"scoreboard_crop_{tag}_{name}.jpg"), img)
        if reader is not None:
            # allowlist digits + separators for a 2-digit score board
            res = reader.readtext(
                variants["clahe"], allowlist="0123456789:-", detail=0
            )
            print(f"[{tag}] OCR -> {res!r}")


def read_scoreboard(crop: np.ndarray, config: dict) -> dict | None:
    """Optionally read the score off a board crop via the LLM layer.

    Pure-CV pipelines pass ``llm.enabled: false`` (the default) and get None
    back — the caller then keeps whatever CV behaviour it had (here: nothing,
    since this feed has no readable board). Only when ``llm.enabled`` AND
    ``llm.uses.scoreboard`` are both true does this reach out to a VLM.
    """
    from ..llm import get_llm, uses  # lazy: no import cost when llm is off

    llm = get_llm(config)
    if llm and uses(config, "scoreboard"):
        return llm.read_scoreboard(crop)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Physical-scoreboard OCR probe.")
    ap.add_argument("video", type=Path)
    ap.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[90, 300, 600, 900, 1200, 1500, 1700, 1900, 2050, 2200, 2300, 2360],
        help="timestamps (seconds) to sample",
    )
    ap.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="board pixel box; omit to only dump full frames",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/out"))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--ocr", action="store_true", help="run EasyOCR (GPU) on crops")
    args = ap.parse_args()
    probe(args.video, args.times, args.roi, args.out_dir, args.scale, args.ocr)


if __name__ == "__main__":
    main()
