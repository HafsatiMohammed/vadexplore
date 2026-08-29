"""Inspection figures for a deliberately extreme set of clips: shortest,
longest, either end of the speech fraction range, plus two seeded random picks.

    python scripts/plot_examples.py [dataset_dir] [--out DIR] [--seed N]
                                    [--max-seconds S] [--n-random K]
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to disk, never open a window

from vadexplore.labels import make_labels, normalize_segments
from vadexplore.loader import load_dataset
from vadexplore.viz import plot_clip

DEFAULT_DIR = "path/to/vad_data"
DEFAULT_OUT = "explore_out/examples"


def speech_fraction(clip) -> float:
    """Share of the clip covered by normalized speech segments."""
    if clip.duration_s <= 0:
        return 0.0
    speech = sum(end - start for start, end in normalize_segments(clip.segments))
    return speech / clip.duration_s


def select(clips, n_random: int = 2, seed: int = 0) -> list[tuple[str, object]]:
    """Pick the representative set, keeping the first reason for a repeat pick."""
    by_speech = sorted(clips, key=speech_fraction)
    candidates = [
        ("shortest", min(clips, key=lambda c: c.duration_s)),
        ("longest", max(clips, key=lambda c: c.duration_s)),
        ("highest speech fraction", by_speech[-1]),
        ("lowest speech fraction", by_speech[0]),
    ]

    rng = random.Random(seed)
    already = {c.stem for _, c in candidates}
    pool = [c for c in clips if c.stem not in already]
    for clip in rng.sample(pool, min(n_random, len(pool))):
        candidates.append((f"random (seed {seed})", clip))

    chosen: list[tuple[str, object]] = []
    seen: set[str] = set()
    for reason, clip in candidates:
        if clip.stem not in seen:
            seen.add(clip.stem)
            chosen.append((reason, clip))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="crop each figure to the first N seconds")
    args = parser.parse_args()

    clips = load_dataset(args.directory)
    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = select(clips, n_random=args.n_random, seed=args.seed)
    print(f"{len(clips)} clips loaded from {os.path.expanduser(args.directory)}")
    print(f"writing {len(chosen)} figures to {out_dir}\n")

    for reason, clip in chosen:
        labels = make_labels(clip)
        path = out_dir / f"{clip.stem}.png"
        plot_clip(clip, labels=labels, max_seconds=args.max_seconds, save=path)

        print(
            f"  {reason:24s} {clip.stem:20s} "
            f"{clip.duration_s:6.2f}s  speech {speech_fraction(clip) * 100:5.1f}%  "
            f"{len(labels['segments_literal']):3d} seg  -> {path.name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
