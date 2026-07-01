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


def _load_identity_map(identities: Path):
    """raw_track_id -> (team, stable_id label) from fv.identity output."""
    idf = pd.read_parquet(identities)
    tid_to_team = dict(zip(idf["raw_track_id"].astype(int), idf["team"]))
    tid_to_label = dict(zip(idf["raw_track_id"].astype(int), idf["stable_id"]))
    return tid_to_team, tid_to_label


def _load_marks_map(marks: Path):
    """raw_track_id -> (label, RGB) from a generic marks parquet.

    Columns: track_id, label, color_r, color_g, color_b. Returns the maps plus a
    ColorPalette of the distinct colours and a colour->index lookup so the CLASS
    colour-lookup can paint arbitrary per-track colours (e.g. team / SPECTATOR /
    REF)."""
    mdf = pd.read_parquet(marks)
    tid_to_label = dict(zip(mdf["track_id"].astype(int), mdf["label"].astype(str)))
    tid_to_rgb = {int(r["track_id"]): (int(r["color_r"]), int(r["color_g"]), int(r["color_b"]))
                  for _, r in mdf.iterrows()}
    uniq = sorted(set(tid_to_rgb.values()))
    color_idx = {c: i for i, c in enumerate(uniq)}
    palette = sv.ColorPalette([sv.Color(*c) for c in uniq]) if uniq else sv.ColorPalette([sv.Color(170, 170, 170)])
    return tid_to_label, tid_to_rgb, color_idx, palette


def run(
    video: Path,
    tracks: Path,
    out: Path,
    trace_length: int = 30,
    draw_traces: bool = True,
    teams: Path | None = None,
    identities: Path | None = None,
    marks: Path | None = None,
) -> Path:
    df = pd.read_parquet(tracks)
    by_frame = {int(f): g for f, g in df.groupby("frame")}
    # Only render up to the last tracked frame. Detection may have been capped
    # (--max-frames) or strided, so rendering the untracked tail of the source
    # video just produces a huge un-annotated clip and wastes time.
    last_frame = int(df["frame"].max()) if len(df) else -1

    tid_to_team = team_to_idx = tid_to_label = None
    marks_map = None
    info = sv.VideoInfo.from_video_path(str(video))
    if marks is not None:
        # Generic per-track colour + label (participant filter: players/ref/spectator).
        tid_to_label, tid_to_rgb, color_idx, palette = _load_marks_map(marks)
        marks_map = (tid_to_rgb, color_idx)
        box = sv.BoxAnnotator(thickness=2, color=palette, color_lookup=sv.ColorLookup.CLASS)
        label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3,
                                  color=palette, color_lookup=sv.ColorLookup.CLASS)
        trace = sv.TraceAnnotator(trace_length=trace_length, color=palette,
                                  color_lookup=sv.ColorLookup.CLASS)
    elif identities is not None:
        # Label by STABLE identity ("A#7"/"A-3"), colour by team. Needs --teams
        # for the per-team kit palette.
        if teams is None:
            raise SystemExit("--identities also needs --teams (for the team colour palette)")
        _, team_to_idx, palette = _load_team_map(teams)
        tid_to_team, tid_to_label = _load_identity_map(identities)
        box = sv.BoxAnnotator(thickness=2, color=palette, color_lookup=sv.ColorLookup.CLASS)
        label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3,
                                  color=palette, color_lookup=sv.ColorLookup.CLASS)
        trace = sv.TraceAnnotator(trace_length=trace_length, color=palette,
                                  color_lookup=sv.ColorLookup.CLASS)
    elif teams is not None:
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
                if marks_map is not None:
                    tid_to_rgb, color_idx = marks_map
                    grey = (170, 170, 170)
                    class_id = np.array([color_idx.get(tid_to_rgb.get(int(t), grey), 0)
                                         for t in tracker_id], dtype=int)
                    labels = [tid_to_label.get(int(t), f"?{t}") for t in tracker_id]
                elif tid_to_label is not None:
                    # Stable identity: colour by team, label with the stable id.
                    teams_of = [tid_to_team.get(int(t), "O") for t in tracker_id]
                    class_id = np.array([team_to_idx[t] for t in teams_of], dtype=int)
                    labels = [tid_to_label.get(int(t), f"?{t}") for t in tracker_id]
                elif tid_to_team is not None:
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
    ap.add_argument("--identities", type=Path, default=None,
                    help="stable-id parquet (from fv.identity): label boxes by stable id "
                         "(also pass --teams for the colour palette)")
    ap.add_argument("--marks", type=Path, default=None,
                    help="generic marks parquet (track_id,label,color_r,color_g,color_b)")
    args = ap.parse_args()
    run(args.video, args.tracks, args.out, args.trace_length, not args.no_traces,
        teams=args.teams, identities=args.identities, marks=args.marks)


if __name__ == "__main__":
    main()
