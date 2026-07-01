"""Stage — jersey-number OCR per track.

Amateur football (Veo panning single-camera) footage. Many players carry a
1-2 digit number on the upper back. When a player faces away from the camera
that number is briefly legible; most of the time it is not. So the strategy is:

  * sample many detection crops spread across each track's lifetime,
  * crop the UPPER-BACK region of each box (where the number sits),
  * upscale + contrast-boost and run digit-only easyocr on each crop,
  * majority-vote a number per track over all its hits.

A track is only reported as "confident" when it has enough agreeing votes and
those votes clearly beat the runner-up, because single OCR reads on this kind
of footage are noisy (ad-board bleed, shirt creases, sponsor text).

Frames are indexed by ABSOLUTE frame number (matching the `frame` column of the
tracks parquet). The video is decoded in a SINGLE sequential pass so crops line
up without per-crop seeking.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLUMNS = [
    "track_id", "number", "votes", "hits", "mean_conf", "n_samples", "confident",
]

# --- confidence policy (tunable) -------------------------------------------
# The winning digit is chosen by CONFIDENCE-WEIGHTED votes (sum of OCR conf per
# distinct read), not raw counts, because a couple of crisp 0.9+ reads of the
# true number are far more trustworthy than several faint 0.4 reads of a
# fragment/hallucination. A track is "confident" only when the winner:
#   * has >= MIN_VOTES agreeing reads,
#   * beats the runner-up's score by the MIN_SCORE_MARGIN factor, AND
#   * accounts for >= MIN_PURITY of the track's total read mass.
# The purity test is what rejects "scattered" tracks (a red-kit player whose
# folds/shadows read as many different digits with no consensus), which was the
# main source of false positives.
MIN_VOTES = 2
MIN_SCORE_MARGIN = 1.5  # winner_score >= MIN_SCORE_MARGIN * runner_up_score
MIN_PURITY = 0.55       # winner_score / total_score across all digits

# Otsu binarisation tends to split a genuine 2-digit number into its leading
# digit ("11" -> "1"). When a 2-digit read has at least one STRONG_CONF read,
# we treat it as the truth and fold matching 1-digit reads into it, so the
# high-quality double-digit evidence is not out-voted by its own fragments.
STRONG_CONF = 0.85

# --- upper-back crop geometry (fraction of the player box) ------------------
# The number sits centred on the upper back: skip the head/shoulders (top),
# stop before the mid-back, and inset the sides so arms/background are dropped.
CROP_TOP = 0.15    # start at 15% down the box
CROP_BOTTOM = 0.45  # end at 45% down the box
CROP_SIDE_INSET = 0.20  # drop 20% off each side

DEBUG_DIR = Path("data/out/jersey_debug")
MAX_DEBUG = 40
DEBUG_PER_KEY = 2  # cap debug crops per (track, digit) to spread across tracks


def _merge_fragments(tv: dict[str, list]) -> dict[str, list]:
    """Fold 1-digit fragments into a strongly-read 2-digit number.

    `tv` maps digit-string -> [count, conf_sum, max_conf]. If a 2-digit string
    D has a max read >= STRONG_CONF and a 1-digit d is one of its digits, d's
    votes are merged into D (and d removed). Returns a new tally dict.
    """
    strong2 = {
        t for t, (_c, _s, mx) in tv.items() if len(t) == 2 and mx >= STRONG_CONF
    }
    if not strong2:
        return tv
    merged = {t: list(rec) for t, rec in tv.items()}
    for d in [t for t in list(merged) if len(t) == 1]:
        # merge into the highest-scoring strong 2-digit number containing d
        targets = [t for t in strong2 if d in t]
        if not targets:
            continue
        target = max(targets, key=lambda t: merged[t][1])
        merged[target][0] += merged[d][0]
        merged[target][1] += merged[d][1]
        merged[target][2] = max(merged[target][2], merged[d][2])
        del merged[d]
    return merged


def _sample_frames(frames: np.ndarray, k: int) -> np.ndarray:
    """Pick up to k frame indices spread evenly across a track's frames."""
    frames = np.unique(frames)
    if len(frames) <= k:
        return frames
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return frames[np.unique(idx)]


def _crop_upper_back(frame: np.ndarray, box: tuple[float, float, float, float]):
    """Crop the upper-back region of a player box. Returns None if degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    cx1 = x1 + CROP_SIDE_INSET * bw
    cx2 = x2 - CROP_SIDE_INSET * bw
    cy1 = y1 + CROP_TOP * bh
    cy2 = y1 + CROP_BOTTOM * bh
    ix1, iy1 = max(0, int(round(cx1))), max(0, int(round(cy1)))
    ix2, iy2 = min(w, int(round(cx2))), min(h, int(round(cy2)))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None
    return frame[iy1:iy2, ix1:ix2]


def _prep_crops(crop: np.ndarray, upscale: int) -> list[np.ndarray]:
    """Upscale + produce a few OCR-friendly variants of a crop.

    Numbers on this footage are small, dark and low-contrast, so we hand
    easyocr several renderings and keep whatever any of them reads. The first
    variant (CLAHE-enhanced colour) is treated as the canonical one for debug
    images.
    """
    up = cv2.resize(
        crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
    )
    # CLAHE on the luminance channel lifts faint numbers off dark night kit.
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    le = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge((le, a, b)), cv2.COLOR_LAB2BGR)

    # Sharpen the enhanced luminance to crisp digit edges.
    blur = cv2.GaussianBlur(le, (0, 0), 2.0)
    sharp = cv2.addWeighted(le, 1.6, blur, -0.6, 0)

    # Otsu binarisation, both polarities (dark-on-light and light-on-dark kit).
    # These recover extra reads on faint crops. They also fragment double digits
    # and hallucinate on texture, but that noise arrives at LOW confidence and is
    # suppressed by the confidence-weighted vote downstream.
    _, th = cv2.threshold(le, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)

    return [clahe_bgr, sharp, th, th_inv]


def run(
    video,
    tracks,
    out,
    start_frame: int | None = None,
    end_frame: int | None = None,
    samples_per_track: int = 25,
    min_conf: float = 0.4,
    upscale: int = 4,
    device: str = "cuda:0",
):
    """OCR jersey numbers per track and write a per-track parquet.

    Args:
        video: path to the source video.
        tracks: path to the tracks parquet [frame, track_id, ..., x1..y2].
        out: output parquet path.
        start_frame/end_frame: inclusive absolute-frame window filter.
        samples_per_track: max detections sampled per track.
        min_conf: min easyocr confidence to keep a digit read.
        upscale: crop upscale factor (INTER_CUBIC).
        device: "cuda:0" enables GPU OCR; anything else falls back to CPU.
    """
    import easyocr  # local import: heavy, and keeps import cost off other modules

    video = Path(video)
    tracks = Path(tracks)
    out = Path(out)

    df = pd.read_parquet(tracks)
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("no detections in the requested frame window")

    # Choose sampled (frame, box) per track. Build a map frame -> list of jobs
    # so a single sequential decode pass can serve every crop.
    jobs: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    track_n_samples: dict[int, int] = {}
    for tid, g in df.groupby("track_id"):
        chosen = _sample_frames(g["frame"].to_numpy(), samples_per_track)
        chosen_set = set(int(f) for f in chosen)
        # one box per chosen frame (first detection of this track on that frame)
        sub = g[g["frame"].isin(chosen_set)].drop_duplicates("frame")
        track_n_samples[int(tid)] = len(sub)
        for row in sub.itertuples(index=False):
            jobs.setdefault(int(row.frame), []).append(
                (int(tid), (row.x1, row.y1, row.x2, row.y2))
            )

    frames_needed = sorted(jobs.keys())
    max_frame = frames_needed[-1]
    print(
        f"{df['track_id'].nunique()} tracks, "
        f"{sum(len(v) for v in jobs.values())} crop jobs across "
        f"{len(frames_needed)} distinct frames (decode 0..{max_frame})"
    )

    gpu = device.startswith("cuda")
    reader = easyocr.Reader(["en"], gpu=gpu)

    # per-track accumulator: {digit_str: [count, conf_sum]}, plus total hits
    votes: dict[int, dict[str, list]] = {}
    hits: dict[int, int] = {}
    conf_sum: dict[int, float] = {}
    for tid in track_n_samples:
        votes[tid] = {}
        hits[tid] = 0
        conf_sum[tid] = 0.0

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for old in DEBUG_DIR.glob("*.jpg"):
        old.unlink()
    # Buffer candidate debug crops (small JPEG-worth of pixels) keyed by
    # (track, digit); we write a curated spread AFTER voting so the reported
    # confident tracks are the ones actually shown.
    debug_buf: dict[tuple[int, str], list[tuple[float, np.ndarray]]] = {}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video}")

    needed_set = set(frames_needed)
    fi = 0
    try:
        while fi <= max_frame:
            grabbed = cap.grab()
            if not grabbed:
                break
            if fi in needed_set:
                ok, frame = cap.retrieve()
                if ok:
                    for tid, box in jobs[fi]:
                        crop = _crop_upper_back(frame, box)
                        if crop is None:
                            continue
                        variants = _prep_crops(crop, upscale)
                        # Aggregate reads across variants: one vote per crop per
                        # distinct digit, taking the best confidence seen.
                        best_by_text: dict[str, float] = {}
                        for var in variants:
                            for _bbox, text, ocr_conf in reader.readtext(
                                var, allowlist="0123456789", detail=1
                            ):
                                text = text.strip()
                                if not (text.isdigit() and 1 <= len(text) <= 2):
                                    continue
                                if ocr_conf < min_conf:
                                    continue
                                if ocr_conf > best_by_text.get(text, 0.0):
                                    best_by_text[text] = float(ocr_conf)
                        for text, ocr_conf in best_by_text.items():
                            rec = votes[tid].setdefault(text, [0, 0.0, 0.0])
                            rec[0] += 1
                            rec[1] += ocr_conf
                            rec[2] = max(rec[2], ocr_conf)
                            hits[tid] += 1
                            conf_sum[tid] += ocr_conf
                            # Buffer up to DEBUG_PER_KEY crops per (track, digit),
                            # keeping the highest-confidence ones.
                            key = (tid, text)
                            buf = debug_buf.setdefault(key, [])
                            buf.append((float(ocr_conf), variants[0].copy()))
                            if len(buf) > DEBUG_PER_KEY:
                                buf.sort(key=lambda t: t[0], reverse=True)
                                del buf[DEBUG_PER_KEY:]
            fi += 1
            if fi % 300 == 0:
                print(f"  decoded {fi}/{max_frame} frames")
    finally:
        cap.release()

    rows = []
    for tid, n_samples in track_n_samples.items():
        tv = votes[tid]
        h = hits[tid]
        mean_conf = (conf_sum[tid] / h) if h else 0.0
        number, win_votes, confident = -1, 0, False
        if tv:
            tally = _merge_fragments(tv)
            # rank by confidence-weighted score (conf_sum), tie-break on count
            ranked = sorted(
                tally.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True
            )
            best_text, (best_votes, best_score, _bmax) = ranked[0]
            runner_score = ranked[1][1][1] if len(ranked) > 1 else 0.0
            total_score = sum(rec[1] for rec in tally.values())
            win_votes = best_votes
            beats_runner = best_score >= MIN_SCORE_MARGIN * runner_score
            pure = total_score > 0 and (best_score / total_score) >= MIN_PURITY
            if best_votes >= MIN_VOTES and beats_runner and pure:
                number = int(best_text)
                confident = True
        rows.append(
            {
                "track_id": tid,
                "number": number,
                "votes": int(win_votes),
                "hits": int(h),
                "mean_conf": float(mean_conf),
                "n_samples": int(n_samples),
                "confident": bool(confident),
            }
        )

    out_df = pd.DataFrame(rows, columns=COLUMNS).sort_values("track_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    # Curate debug crops: confident tracks first (so they're always shown),
    # then everything else, up to MAX_DEBUG.
    confident_tids = set(out_df.loc[out_df["confident"], "track_id"])
    ordered_keys = sorted(
        debug_buf.keys(),
        key=lambda k: (k[0] not in confident_tids, k[0], k[1]),
    )
    n_debug = 0
    for tid, text in ordered_keys:
        for ocr_conf, img in debug_buf[(tid, text)]:
            if n_debug >= MAX_DEBUG:
                break
            p = DEBUG_DIR / f"t{tid}_n{text}_c{ocr_conf:.2f}.jpg"
            cv2.imwrite(str(p), img)
            n_debug += 1
        if n_debug >= MAX_DEBUG:
            break

    n_conf = int(out_df["confident"].sum())
    print(f"wrote {len(out_df)} tracks -> {out} ({n_conf} confident)")
    print(f"saved {n_debug} debug crops -> {DEBUG_DIR}")
    if n_conf:
        conf_rows = out_df[out_df["confident"]]
        for r in conf_rows.itertuples(index=False):
            print(
                f"  track {r.track_id}: number={r.number} "
                f"votes={r.votes} hits={r.hits} mean_conf={r.mean_conf:.2f}"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Jersey-number OCR per track -> parquet.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/tracks/jersey.parquet"))
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--samples-per-track", type=int, default=25)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(
        args.video,
        args.tracks,
        args.out,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        samples_per_track=args.samples_per_track,
        min_conf=args.min_conf,
        upscale=args.upscale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
