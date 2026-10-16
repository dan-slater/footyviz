"""Cut the labelled 120 s clips (BENCHMARK.md 7).

Window sources are stream copies that start on the source keyframe at or before 840 s
(`source.src_t0`, 835.835 s for Rookie's fixed 150-frame GOP at 29.97 fps). Their timestamps
are re-zeroed with no edit list, so we cut by FRAME INDEX: window frame n is source time
src_t0 + n / fps. Clip frame 0 = the first frame with source time >= label_window.start_s;
the clip has exactly round(120 * fps) frames. Full sources (src_t0 = 0) are cut the same way
after a fast seek to just before the start.

Writes clips/<id>.mp4 (libx264 crf 12, audio kept) and records `clip` in the manifest:
first source frame, n_frames, fps and 64-bit pHashes of frames 0 / mid / last so a re-fetched
clip can be checked for frame alignment.
"""
from __future__ import annotations

import math
import subprocess
from fractions import Fraction

import cv2
import numpy as np

from . import manifest as M

PRESEEK_S = 3.0


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def phash(bgr: np.ndarray) -> str:
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g.astype(np.float32))[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def plan(x: dict) -> dict:
    fr = Fraction(x["source"]["fps"])
    t0 = float(x["source"]["src_t0"])
    w = x["label_window"]
    # first window frame with source time >= start_s - half a frame (29.97 fps frame times
    # never land exactly on 900.000: window frame 1923 is source 899.9991 s)
    first = math.ceil((w["start_s"] - t0) * fr - 0.5)
    n = round((w["end_s"] - w["start_s"]) * fr)
    return {"fps": fr, "first_src_frame": first, "n_frames": n,
            "start_src_s": round(t0 + first / float(fr), 4)}


def window_check(path) -> dict:
    """Is frame-index re-timing valid for this stream copy? It needs frame 0 to be the
    keyframe and uniform pts. Validated 2026-10-16 on the reference: a copy WITHOUT an edit
    list can drop open-GOP leading B-frames, which leaves a pts gap after frame 0 and shifts
    every index by the gap (2 frames there). Rookie eval windows pass; the A2 relay starts
    mid-GOP (first keyframe at index 25)."""
    import av
    c = av.open(str(path))
    s = c.streams.video[0]
    pts, keys = [], []
    for i, f in enumerate(c.decode(s)):
        pts.append(float(f.pts * s.time_base))
        if f.key_frame:
            keys.append(i)
        if i >= 200:
            break
    c.close()
    d = np.diff(pts)
    step = float(np.median(d))
    gaps = int(np.sum(d > 1.5 * step))
    first_key = keys[0] if keys else None
    return {"frame0_key": first_key == 0, "first_key_idx": first_key, "pts_gaps": gaps,
            "first_pts": round(pts[0], 4), "ok": first_key == 0 and gaps == 0}


def cut_one(x: dict, force: bool = False) -> dict | None:
    src = M.data_path(x["source"]["file"])
    out = M.data_path(x["clip_file"])
    if not src.exists() or x["source"].get("src_t0") is None or not x["source"].get("fps"):
        print(f"[{x['id']}] source missing or not re-timed, skipped ({x['source']['file']})")
        return None
    if x["source"]["kind"] == "window":
        wc = window_check(src)
        if not wc["ok"]:
            msg = f"[{x['id']}] window fails the re-timing check {wc}"
            if x["split"] == "eval" and not x["source"].get("src_t0_verified"):
                raise SystemExit(msg + ": refusing to cut an eval clip (verify src_t0 by content)")
            print("WARNING " + msg + " (train: cutting anyway)")
    p = plan(x)
    out.parent.mkdir(parents=True, exist_ok=True)
    if force or not out.exists():
        fr, first, n = p["fps"], p["first_src_frame"], p["n_frames"]
        # fast seek to PRESEEK_S before the start (decode is exact after -ss), then select by
        # frame index relative to the seek point
        seek_frame = max(0, first - int(PRESEEK_S * fr))
        seek_s = float(seek_frame / fr)
        skip = first - seek_frame
        cmd = [ffmpeg(), "-y", "-loglevel", "error"]
        if seek_frame:
            cmd += ["-ss", f"{seek_s - 0.25 / float(fr):.4f}"]
        cmd += ["-i", str(src),
                "-vf", f"select=gte(n\\,{skip}),setpts=PTS-STARTPTS",
                "-af", f"atrim=start={skip / float(fr):.4f},asetpts=PTS-STARTPTS",
                "-frames:v", str(n), "-map", "0:v:0", "-map", "0:a:0?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p",
                "-r", f"{fr.numerator}/{fr.denominator}",
                "-c:a", "aac", "-b:a", "128k", str(out)]
        print(f"[{x['id']}] cutting src frames {first}..{first + n - 1} -> {out}")
        subprocess.run(cmd, check=True)
    cap = cv2.VideoCapture(str(out))
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hashes = {}
    for f in (0, p["n_frames"] // 2, p["n_frames"] - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            hashes[str(f)] = phash(im)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if got != p["n_frames"]:
        raise SystemExit(f"[{x['id']}] FRAME COUNT MISMATCH: clip has {got}, expected {p['n_frames']}")
    return {"fps": f"{p['fps'].numerator}/{p['fps'].denominator}", "n_frames": got,
            "width": w, "height": h, "first_src_frame": p["first_src_frame"],
            "start_src_s": p["start_src_s"], "phash": hashes}


def main(ids: list[str] | None = None, force: bool = False) -> int:
    m = M.load()
    for x in M.matches(m, ids=ids):
        c = cut_one(x, force)
        if c:
            x["clip"] = c
            print(f"[{x['id']}] ok: {c['n_frames']} frames @ {c['fps']}, src start {c['start_src_s']} s, pHash {c['phash']}")
            M.save(m)
    return 0
