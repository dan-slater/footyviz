"""Referee-whistle detection from the match audio track (M3 event cue).

A referee whistle is a loud, *narrowband tonal* burst: most of its energy sits
in a ~3-4 kHz band and it starts abruptly. We exploit both facts:

  * band energy  E_band  = STFT power summed over [f_lo, f_hi] (default 3-4 kHz)
  * tonality     ratio   = E_band / E_total  (whistles ~>0.5; crowd/shouts/
                            ball-thumps are broadband, ratio well below)

A frame is "whistle-active" when E_band exceeds a high absolute floor AND ratio
exceeds a tonality floor. Adjacent active frames are merged into a single blast
(gap <= merge_gap), and blasts shorter than min_dur are dropped as clicks.

Extract the audio first (mono 16 kHz is plenty; Nyquist 8 kHz > band):
    ffmpeg -i match.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le match_16k.wav

Then:
    python -m fv.events.whistle match_16k.wav --out data/out/whistles.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


@dataclass
class Blast:
    start_s: float
    end_s: float
    dur_s: float
    peak_s: float
    peak_eb: float
    peak_ratio: float
    n_frames: int


def band_features(
    wav_path: Path, f_lo: float, f_hi: float, nperseg: int, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, band_energy, tonality_ratio) per STFT frame."""
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float32)
    x /= np.abs(x).max() + 1e-9
    f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None)
    P = np.abs(Z) ** 2
    band = (f >= f_lo) & (f <= f_hi)
    eb = P[band].sum(0)
    ratio = eb / (P.sum(0) + 1e-12)
    return t, eb, ratio


def detect(
    t: np.ndarray,
    eb: np.ndarray,
    ratio: np.ndarray,
    eb_thr: float,
    ratio_thr: float,
    merge_gap: float,
    min_dur: float,
) -> list[Blast]:
    active = (eb > eb_thr) & (ratio > ratio_thr)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    # split active-frame indices into runs separated by more than merge_gap
    gaps = np.flatnonzero(np.diff(t[idx]) > merge_gap)
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [idx.size - 1]))
    blasts: list[Blast] = []
    for s, e in zip(starts, ends):
        run = idx[s : e + 1]
        dur = float(t[run[-1]] - t[run[0]])
        if dur < min_dur:
            continue
        peak = run[int(np.argmax(eb[run]))]
        blasts.append(
            Blast(
                start_s=round(float(t[run[0]]), 2),
                end_s=round(float(t[run[-1]]), 2),
                dur_s=round(dur, 3),
                peak_s=round(float(t[peak]), 2),
                peak_eb=float(eb[peak]),
                peak_ratio=round(float(ratio[peak]), 3),
                n_frames=int(run.size),
            )
        )
    return blasts


def classify_blast_event(
    frames: list, config: dict, context: str = ""
) -> dict | None:
    """Optionally classify a whistle-candidate clip as a goal via the LLM layer.

    With ``llm.enabled: false`` (default) this returns None and the pipeline
    keeps its pure-CV behaviour: the whistle timestamps in whistles.csv stand on
    their own as goal *candidates*. Only when ``llm.enabled`` AND
    ``llm.uses.events`` are true is a VLM asked to judge goal/no-goal.
    """
    from ..llm import get_llm, uses  # lazy: no import cost when llm is off

    llm = get_llm(config)
    if llm and uses(config, "events"):
        return llm.classify_event(frames, context)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect referee whistles in a wav track.")
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/out/whistles.csv"))
    ap.add_argument("--f-lo", type=float, default=3000.0)
    ap.add_argument("--f-hi", type=float, default=4000.0)
    ap.add_argument("--nperseg", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    # thresholds: eb-pct picks the absolute floor as a percentile of band energy
    ap.add_argument("--eb-pct", type=float, default=99.5)
    ap.add_argument("--ratio-thr", type=float, default=0.6)
    ap.add_argument("--merge-gap", type=float, default=0.35)
    ap.add_argument("--min-dur", type=float, default=0.04)
    args = ap.parse_args()

    t, eb, ratio = band_features(args.wav, args.f_lo, args.f_hi, args.nperseg, args.hop)
    eb_thr = float(np.percentile(eb, args.eb_pct))
    blasts = detect(t, eb, ratio, eb_thr, args.ratio_thr, args.merge_gap, args.min_dur)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(Blast(0, 0, 0, 0, 0, 0, 0)).keys()))
        w.writeheader()
        for b in blasts:
            w.writerow(asdict(b))

    print(f"eb_thr(p{args.eb_pct})={eb_thr:.3g}  ratio_thr={args.ratio_thr}")
    print(f"{len(blasts)} whistle blast(s) -> {args.out}")
    for b in blasts:
        mm, ss = divmod(b.peak_s, 60)
        print(f"  {int(mm):02d}:{ss:05.2f}  dur={b.dur_s:.2f}s  ratio={b.peak_ratio}")


if __name__ == "__main__":
    main()
