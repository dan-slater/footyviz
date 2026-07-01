"""Stage 7 — burn boxes, motion traces and "Player N" labels onto the video.

Reads the persisted tracks parquet and re-draws over the original frames.
Note: supervision's VideoSink writes video only. Audio (needed for M3 event
detection anyway) is muxed back later with ffmpeg if required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv


# Fallback colour for tracks with no team assignment ("other"/unknown).
_OTHER_RGB = (170, 170, 170)
_TEAM_ORDER = ("A", "B", "O")


def _load_team_map(teams: Path):
    """Return (track_id -> team_index, ColorPalette, team->label prefix).

    Team index is 0/1/2 for A/B/O so supervision's CLASS colour-lookup can pick
    the right kit colour per box. Palette colours come from the actual clustered
    kit colour so the box matches what's on screen.
    """
    tdf = pd.read_parquet(teams)
    tid_to_team = dict(zip(tdf["track_id"].astype(int), tdf["team"]))
    # One representative colour per team label (rows repeat it per track).
    colours = {}
    for team in _TEAM_ORDER:
        rows = tdf[tdf["team"] == team]
        if len(rows):
            r, g, b = int(rows.iloc[0].color_r), int(rows.iloc[0].color_g), int(rows.iloc[0].color_b)
            colours[team] = (r, g, b)
    palette = sv.ColorPalette([
        sv.Color(*colours.get(team, _OTHER_RGB)) for team in _TEAM_ORDER
    ])
    team_to_idx = {t: i for i, t in enumerate(_TEAM_ORDER)}
    return tid_to_team, team_to_idx, palette


def run(
    video: Path,
    tracks: Path,
    out: Path,
    trace_length: int = 30,
    draw_traces: bool = True,
    teams: Path | None = None,
) -> Path:
    df = pd.read_parquet(tracks)
    by_frame = {int(f): g for f, g in df.groupby("frame")}
    # Only render up to the last tracked frame. Detection may have been capped
    # (--max-frames) or strided, so rendering the untracked tail of the source
    # video just produces a huge un-annotated clip and wastes time.
    last_frame = int(df["frame"].max()) if len(df) else -1

    tid_to_team = team_to_idx = None
    info = sv.VideoInfo.from_video_path(str(video))
    if teams is not None:
        # Colour boxes/traces/labels by TEAM (durable anchor across ID swaps).
        tid_to_team, team_to_idx, palette = _load_team_map(teams)
        box = sv.BoxAnnotator(thickness=2, color=palette, color_lookup=sv.ColorLookup.CLASS)
        label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3,
                                  color=palette, color_lookup=sv.ColorLookup.CLASS)
        trace = sv.TraceAnnotator(trace_length=trace_length, color=palette,
                                  color_lookup=sv.ColorLookup.CLASS)
    else:
        box = sv.BoxAnnotator(thickness=2)
        label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
        trace = sv.TraceAnnotator(trace_length=trace_length)

    out.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out), info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(str(video))):
            if i > last_frame:
                break
            group = by_frame.get(i)
            if group is not None and len(group):
                tracker_id = group["track_id"].to_numpy(dtype=int)
                if tid_to_team is not None:
                    teams_of = [tid_to_team.get(int(t), "O") for t in tracker_id]
                    class_id = np.array([team_to_idx[t] for t in teams_of], dtype=int)
                    # ASCII separator: cv2's Hershey font can't render "·".
                    labels = [f"{t} | {tid}" for t, tid in zip(teams_of, tracker_id)]
                else:
                    class_id = group["class_id"].to_numpy(dtype=int)
                    labels = [f"Player {tid}" for tid in tracker_id]
                detections = sv.Detections(
                    xyxy=group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
                    confidence=group["confidence"].to_numpy(dtype=float),
                    class_id=class_id,
                    tracker_id=tracker_id,
                )
                if draw_traces:
                    frame = trace.annotate(frame, detections)
                frame = box.annotate(frame, detections)
                frame = label.annotate(frame, detections, labels=labels)
            sink.write_frame(frame)
    print(f"wrote annotated video -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Burn tracking overlay onto the video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/out/annotated.mp4"))
    ap.add_argument("--trace-length", type=int, default=30)
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--teams", type=Path, default=None,
                    help="team-map parquet (from fv.teams): colour boxes by team")
    args = ap.parse_args()
    run(args.video, args.tracks, args.out, args.trace_length, not args.no_traces,
        teams=args.teams)


if __name__ == "__main__":
    main()
