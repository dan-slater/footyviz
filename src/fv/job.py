"""One footyviz job: URL + time window -> overlay.mp4, stats.json, poster.jpg, roster.jpg.

    python -m fv.job --url "https://www.youtube.com/watch?v=hhaWMwLlugE" \
        --start-s 666 --end-s 726 --out out/my_match

This is the main entry point of footyviz. --url takes a YouTube/Veo URL or a
local video file. The optional local HTTP API (fv.api) just queues runs of this.

Stages (status names are the API contract, see API.md):

    downloading  acquire 720p via yt-dlp (reuses data/raw/<id>_720.mp4) -> slice clip.mp4
    tracking     YOLO11 persons + ball, ByteTrack -> BoT-SORT/OSNet re-track
    teams        kit-colour teams -> CV-only identity fusion (no VLM)
    stats        whistles + fv.stats -> stats.json, roster.jpg, crop_*.jpg
    rendering    FIFA-style overlay, honesty footer -> H.264 overlay.mp4 + poster.jpg

`progress.json` in the job dir is rewritten at every step so the API (fv.api)
can report status without talking to this process. The VLM layer is never
enabled here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw"
MAX_CLIP_S = float(os.environ.get("FOOTYVIZ_MAX_CLIP_S", 180))
# Local weights if present, else ultralytics / boxmot download them on first use.
MODEL = str(REPO / "yolo11m.pt") if (REPO / "yolo11m.pt").exists() else "yolo11m.pt"
OSNET = REPO / "data" / "models" / "osnet_x1_0_msmt17.pt"

# (status, progress at stage start, progress at stage end)
STAGES = {
    "downloading": (0.00, 0.10),
    "tracking":    (0.10, 0.45),
    "teams":       (0.45, 0.60),
    "stats":       (0.60, 0.70),
    "rendering":   (0.70, 0.99),
}
RESULT_FILES = ("overlay.mp4", "stats.json", "poster.jpg", "roster.jpg")


class Progress:
    """Atomic writer for <job>/progress.json, throttled to ~2 writes/s."""

    def __init__(self, job_dir: Path, echo: bool = True):
        self.path = job_dir / "progress.json"
        self.echo = echo
        self._last = 0.0
        self.state = {"status": "queued", "progress": 0.0, "stage_note": "", "error": None}

    def _write(self):
        self.state["updated_at"] = time.time()
        if self.echo:
            st = self.state
            print(f"[{st['progress']:6.1%}] {st['status']:<11} {st['stage_note']}", flush=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)

    def stage(self, status: str, frac: float = 0.0, note: str = "", force: bool = False):
        lo, hi = STAGES[status]
        p = round(lo + (hi - lo) * min(max(frac, 0.0), 1.0), 4)
        self.state.update(status=status, stage_note=note,
                          progress=max(p, self.state["progress"]))
        now = time.time()
        if force or now - self._last > 2.0:
            self._last = now
            self._write()

    def done(self):
        self.state.update(status="done", progress=1.0, stage_note="done")
        self._write()

    def fail(self, msg: str):
        self.state.update(status="error", error=msg)
        self._write()


# --------------------------------------------------------------------------- #
# acquire + slice
# --------------------------------------------------------------------------- #
def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def youtube_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([\w-]{11})", url)
    return m.group(1) if m else None


def _local_source(url: str, restrict: bool) -> Path | None:
    """A local video path. With restrict (the HTTP API) only files under data/raw/ count."""
    p = url[len("file://"):] if url.startswith("file://") else url
    if "://" in p:
        return None
    cand = Path(p) if Path(p).is_absolute() else REPO / p
    try:
        cand = cand.resolve(strict=True)
    except OSError:
        return None
    if not cand.is_file():
        return None
    return cand if not restrict or cand.is_relative_to(RAW_DIR.resolve()) else None


# yt-dlp extras: cookies get past YouTube's "confirm you're not a bot" wall; deno
# solves its JS challenge (picked up from ~/.deno/bin if installed).
YTDLP_EXTRA: dict = {}
if (Path.home() / ".deno" / "bin").is_dir():
    os.environ["PATH"] = f"{Path.home() / '.deno' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"


def _title(vid: str, url: str) -> str:
    """Video title, cached next to the raw file; falls back to the id."""
    cache = RAW_DIR / f"{vid}.title.json"
    if cache.exists():
        return json.loads(cache.read_text())["title"]
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "socket_timeout": 15,
                               **YTDLP_EXTRA}) as y:
            title = y.extract_info(url, download=False).get("title") or vid
        cache.write_text(json.dumps({"title": title}))
        return title
    except Exception:
        return vid


def acquire(url: str, prog: Progress, restrict_local: bool = False) -> tuple[Path, str]:
    """Return (720p source path, title)."""
    local = _local_source(url, restrict_local)
    if local is not None:
        return local, local.stem
    if "://" not in url:
        raise ValueError(f"not a URL and not a readable local file: {url}")
    vid = youtube_id(url)
    if vid and (RAW_DIR / f"{vid}_720.mp4").exists():
        prog.stage("downloading", 0.5, f"using cached source {vid}_720.mp4", force=True)
        return RAW_DIR / f"{vid}_720.mp4", _title(vid, url)

    import yt_dlp

    def hook(d):
        if d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                frac = d.get("downloaded_bytes", 0) / tot
                prog.stage("downloading", 0.9 * frac, f"downloading {frac:.0%}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]",
        "outtmpl": str(RAW_DIR / "%(id)s_720.%(ext)s"),
        "noplaylist": True, "quiet": True, "socket_timeout": 30,
        "progress_hooks": [hook],
        "ffmpeg_location": os.path.dirname(_ffmpeg()),
        **YTDLP_EXTRA,
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
        path = Path(y.prepare_filename(info))
    (RAW_DIR / f"{info['id']}.title.json").write_text(json.dumps({"title": info.get("title") or info["id"]}))
    return path, info.get("title") or info["id"]


def video_meta(path: Path) -> tuple[float, int, float]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n, n / fps


def slice_clip(src: Path, start_s: float, end_s: float, out: Path) -> None:
    """Frame-accurate re-encoded cut (keeps audio for whistle detection)."""
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src),
           "-t", f"{end_s - start_s:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", str(out)]
    subprocess.run(cmd, check=True)


def whistles(src: Path, start_s: float, end_s: float, fps: float, wav: Path) -> list[dict]:
    """Referee-whistle candidates in the clip window's audio.

    Uses the source's own audio, else a sibling `<id>_1080.mp4` (YouTube's 720p
    combined stream sometimes ships without audio). Empty if neither has any."""
    import numpy as np
    from .events import whistle
    cands = [src]
    if src.stem.endswith("_720"):
        cands.append(src.with_name(src.stem[:-4] + "_1080.mp4"))
    for a in cands:
        if not a.exists():
            continue
        r = subprocess.run([_ffmpeg(), "-y", "-loglevel", "quiet", "-ss", f"{start_s:.3f}",
                            "-i", str(a), "-t", f"{end_s - start_s:.3f}", "-vn", "-ac", "1",
                            "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
        if r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000:
            break
    else:
        return []
    try:
        t, eb, ratio = whistle.band_features(wav, 3000.0, 4000.0, 1024, 256)
        blasts = whistle.detect(t, eb, ratio, float(np.percentile(eb, 99.5)),
                                ratio_thr=0.75, merge_gap=0.35, min_dur=0.08)
    finally:
        wav.unlink(missing_ok=True)
    return [{"type": "whistle", "t_s": b.peak_s, "frame": int(round(b.peak_s * fps))}
            for b in blasts]


def _versions() -> dict:
    import importlib.metadata as md

    def v(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "?"
    return {"footyviz": v("footyviz"), "yolo": f"yolo11m.pt (ultralytics {v('ultralytics')})",
            "tracker": f"ByteTrack (supervision {v('supervision')})",
            "reid": f"BoT-SORT + OSNet x1_0 msmt17 (boxmot {v('boxmot')})", "vlm": None}


# --------------------------------------------------------------------------- #
# the job
# --------------------------------------------------------------------------- #
def run(url: str, start_s: float, end_s: float, job_dir: Path, *,
        identity: bool = True, device: str = "cuda:0", file_prefix: str = "",
        restrict_local: bool = False) -> dict:
    from . import detect_track, overlay, stats, teams

    t0 = time.time()
    job_dir.mkdir(parents=True, exist_ok=True)
    prog = Progress(job_dir)

    # downloading -------------------------------------------------------------
    prog.stage("downloading", 0.0, "fetching source", force=True)
    src, title = acquire(url, prog, restrict_local)
    fps, _, dur = video_meta(src)
    start_s = max(0.0, float(start_s))
    end_s = min(float(end_s), dur, start_s + MAX_CLIP_S)
    if end_s - start_s < 2.0:
        raise ValueError(f"clip [{start_s:.1f}, {end_s:.1f}] s is too short (source is {dur:.1f} s)")
    prog.stage("downloading", 0.92, f"cutting {start_s:.1f}-{end_s:.1f} s", force=True)
    clip = job_dir / "clip.mp4"
    slice_clip(src, start_s, end_s, clip)
    fps, n_frames, _ = video_meta(clip)

    # tracking ----------------------------------------------------------------
    tracks = job_dir / "tracks.parquet"
    detect_track.run(clip, tracks, model_path=MODEL, device=device, conf=0.30, ball=True,
                     progress=lambda i, n: prog.stage("tracking", 0.75 * i / max(n, 1),
                                                      f"detect+track {i}/{n} frames"))
    person_tracks = tracks
    reid = job_dir / "reid.parquet"
    if identity:
        prog.stage("tracking", 0.78, "re-identification tracking (BoT-SORT + OSNet)", force=True)
        try:
            from . import reid_track
            reid_track.run(clip, tracks, reid, device=device,
                           reid_weights=str(OSNET))
            person_tracks = reid
        except Exception:
            traceback.print_exc()
            identity = False

    # teams + identity --------------------------------------------------------
    prog.stage("teams", 0.0, "clustering kit colours", force=True)
    team_map = job_dir / "teams.parquet"
    teams.run(clip, person_tracks, team_map)
    identities = None
    if identity:
        prog.stage("teams", 0.4, "fusing tracklets into stable identities (CV only)", force=True)
        try:
            from . import identity as ident
            identities = job_dir / "identities.parquet"
            ident.run(clip, person_tracks, identities, team_map, device=device,
                      cache=job_dir / "identity_desc.npz", reid_weights=str(OSNET))
        except Exception:
            traceback.print_exc()
            identities = None

    # stats -------------------------------------------------------------------
    prog.stage("stats", 0.0, "whistles", force=True)
    events = whistles(src, start_s, end_s, fps, job_dir / "clip.wav")
    prog.stage("stats", 0.2, "possession, distance, heatmaps, roster", force=True)
    pipeline = _versions()
    if identities is None:
        pipeline["reid"] = None
    source = {"url": url, "title": title, "start_s": round(start_s, 3), "end_s": round(end_s, 3)}
    st = stats.compute(clip, person_tracks, team_map, job_dir, ball=detect_track.ball_path(tracks),
                       identities=identities, source=source, events=events,
                       pipeline=pipeline, file_prefix=file_prefix)

    # rendering ---------------------------------------------------------------
    prog.stage("rendering", 0.0, "rendering overlay", force=True)
    overlay.run(clip, person_tracks, job_dir / "overlay.mp4", teams=team_map,
                identities=identities, style="fifa", footer=st["identity_note"],
                poster=job_dir / "poster.jpg",
                progress=lambda i, n: prog.stage("rendering", i / max(n, 1),
                                                 f"rendering {i}/{n} frames"))

    st["pipeline"]["run_time_s"] = round(time.time() - t0, 1)
    (job_dir / "stats.json").write_text(json.dumps(st, indent=1))
    clip.unlink(missing_ok=True)                       # keep the job dir small
    prog.done()
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one footyviz job end to end.")
    ap.add_argument("--url", required=True, help="YouTube/Veo URL, or a local video file")
    ap.add_argument("--start-s", type=float, default=0.0, help="clip start, seconds into the video")
    ap.add_argument("--end-s", type=float, default=None,
                    help=f"clip end (default start + {MAX_CLIP_S:g} s; longer windows are capped)")
    ap.add_argument("--out", type=Path, required=True, help="output dir (created)")
    ap.add_argument("--no-identity", action="store_true", help="skip reid + identity (track-level roster)")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 (default) or cpu (works, but slow)")
    ap.add_argument("--cookies", type=Path, default=None,
                    help="cookies.txt for yt-dlp (if YouTube asks you to sign in)")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                    help="let yt-dlp read cookies from your browser, e.g. firefox or chrome")
    ap.add_argument("--api", action="store_true", help=argparse.SUPPRESS)  # set by fv.api
    args = ap.parse_args()
    if args.cookies:
        YTDLP_EXTRA["cookiefile"] = str(args.cookies)
    if args.cookies_from_browser:
        YTDLP_EXTRA["cookiesfrombrowser"] = (args.cookies_from_browser,)
    end_s = args.end_s if args.end_s is not None else args.start_s + MAX_CLIP_S
    if args.device == "cpu":
        print("warning: --device cpu works but is slow (roughly 2-5 frames/s; a 60 s clip "
              "takes many minutes). A CUDA GPU is ~120 frames/s.", flush=True)
    try:
        st = run(args.url, args.start_s, end_s, args.out, identity=not args.no_identity,
                 device=args.device, restrict_local=args.api,
                 file_prefix=f"/jobs/{args.out.name}/files/" if args.api else "")
    except Exception as e:
        traceback.print_exc()
        p = Progress(args.out)
        if p.path.exists():
            p.state.update(json.loads(p.path.read_text()))
        p.fail(f"{type(e).__name__}: {e}"[:500])
        raise SystemExit(1)
    pos, dist = st["possession"], st["distance"]["teams"]
    print(f"\ndone in {st['pipeline']['run_time_s']} s -> {args.out}/")
    print("  overlay.mp4  poster.jpg  roster.jpg  stats.json")
    for t, info in st["teams"].items():
        print(f"  team {t} ({info['label']}): possession {pos[t]:.0%}, "
              f"distance ~{dist[t]['m']:.0f} m (estimate)")
    print(f"  unresolved possession {pos['unresolved']:.0%} (ball seen in {pos['ball_frames_frac']:.0%} of frames)")
    print(f"  {st['identity_note']}")


if __name__ == "__main__":
    main()
