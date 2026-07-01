"""Stage 1 — pull the match video from YouTube with yt-dlp."""
from __future__ import annotations

import argparse
from pathlib import Path

import yt_dlp


def download(url: str, out_dir: Path, max_height: int = 1080) -> Path:
    """Download a YouTube video as mp4 and return the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={max_height}]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a match video from YouTube.")
    ap.add_argument("url")
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--max-height", type=int, default=1080)
    args = ap.parse_args()
    path = download(args.url, args.out_dir, args.max_height)
    print(path)


if __name__ == "__main__":
    main()
