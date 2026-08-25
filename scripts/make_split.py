"""Freeze the speaker-disjoint train/val/test split into splits/split.json.

The split is keyed by speaker: every clip of a speaker lands in one partition,
so no evaluation number can be inflated by speaker memorization.

This file is committed and read by every experiment. Regenerating it changes
what train and test mean, so the script refuses to overwrite an existing split
unless --force is given.

    python scripts/make_split.py [dataset_dir] [--stats explore_out/corpus_stats.json]
                                 [--out splits/split.json] [--seed 0] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from vadexplore.loader import load_dataset

DEFAULT_DIR = "~/Downloads/vad_data"
DEFAULT_STATS = "explore_out/corpus_stats.json"
DEFAULT_OUT = "splits/split.json"
PARTITIONS = ("train", "val", "test")


def derive_split(clips_per_speaker: dict, val_frac: float, test_frac: float, seed: int) -> dict:
    """Greedy largest-first speaker assignment, deterministic given the seed.

    A speaker is indivisible, so an exact ratio is unreachable. Walking from
    the most clips to the fewest and giving each speaker to whichever
    partition is furthest below target bounds the error by the largest
    remaining speaker instead of letting it accumulate. The seed only breaks
    ties between speakers with identical clip counts, so the result is stable.
    """
    rng = random.Random(seed)
    targets = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    total = sum(clips_per_speaker.values())

    speakers = sorted(clips_per_speaker, key=lambda s: (-clips_per_speaker[s], s))
    jitter = {s: rng.random() for s in speakers}
    speakers.sort(key=lambda s: (-clips_per_speaker[s], jitter[s]))

    assigned = {p: [] for p in PARTITIONS}
    counts = {p: 0 for p in PARTITIONS}
    for speaker in speakers:
        deficit = {p: targets[p] - counts[p] / max(total, 1) for p in PARTITIONS}
        pick = max(PARTITIONS, key=lambda p: deficit[p])
        assigned[pick].append(speaker)
        counts[pick] += clips_per_speaker[speaker]
    return assigned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing split file")
    args = parser.parse_args()

    out_path = Path(os.path.expanduser(args.out))
    if out_path.exists() and not args.force:
        print(f"error: {out_path} already exists.\n"
              "  The split is committed and every experiment reads it. Rerun with "
              "--force only if you intend to invalidate previous results.",
              file=sys.stderr)
        return 2

    clips = load_dataset(args.directory)
    stems_by_speaker = defaultdict(list)
    seconds_by_speaker = defaultdict(float)
    for clip in clips:
        stems_by_speaker[clip.speaker_id].append(clip.stem)
        seconds_by_speaker[clip.speaker_id] += clip.duration_s
    clips_per_speaker = {s: len(v) for s, v in stems_by_speaker.items()}

    # Prefer the proposal already recorded by the exploration run, so the frozen
    # split is the same one the data report describes.
    stats_path = Path(os.path.expanduser(args.stats))
    source = None
    assigned = None
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        proposal = stats.get("proposed_split", {}).get("partitions")
        if proposal and all(p in proposal for p in PARTITIONS):
            candidate = {p: list(proposal[p]["speakers"]) for p in PARTITIONS}
            covered = {s for v in candidate.values() for s in v}
            if covered == set(clips_per_speaker):
                assigned, source = candidate, f"proposal in {stats_path}"
            else:
                print(f"note: proposal in {stats_path} covers {len(covered)} speakers but the "
                      f"dataset has {len(clips_per_speaker)}, deriving instead")

    if assigned is None:
        assigned = derive_split(clips_per_speaker, args.val_frac, args.test_frac, args.seed)
        source = f"derived deterministically, seed {args.seed}"

    # Primary invariant. A speaker in two partitions is a leak, not a warning.
    seen = {}
    for partition, speakers in assigned.items():
        for speaker in speakers:
            assert speaker not in seen, (
                f"speaker {speaker} appears in both {seen[speaker]} and {partition}")
            seen[speaker] = partition
    assert set(seen) == set(clips_per_speaker), (
        f"split covers {len(seen)} speakers, dataset has {len(clips_per_speaker)}")

    total_clips = len(clips)
    payload = {
        "dataset_dir": os.path.expanduser(args.directory),
        "source": source,
        "seed": args.seed,
        "targets": {"train": 1.0 - args.val_frac - args.test_frac,
                    "val": args.val_frac, "test": args.test_frac},
        "n_speakers": len(clips_per_speaker),
        "n_clips": total_clips,
        "partitions": {},
    }
    for partition in PARTITIONS:
        speakers = sorted(assigned[partition])
        stems = sorted(stem for s in speakers for stem in stems_by_speaker[s])
        seconds = sum(seconds_by_speaker[s] for s in speakers)
        payload["partitions"][partition] = {
            "speakers": speakers,
            "n_speakers": len(speakers),
            "stems": stems,
            "n_clips": len(stems),
            "clip_fraction": len(stems) / total_clips if total_clips else 0.0,
            "hours": seconds / 3600.0,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"wrote {out_path}")
    print(f"  source: {source}")
    print(f"  {payload['n_speakers']} speakers, {total_clips} clips\n")
    print(f"  {'partition':<10} {'speakers':>9} {'clips':>7} {'percent':>8} {'hours':>7}")
    for partition in PARTITIONS:
        v = payload["partitions"][partition]
        print(f"  {partition:<10} {v['n_speakers']:9d} {v['n_clips']:7d} "
              f"{v['clip_fraction'] * 100:7.1f}% {v['hours']:7.2f}")
    print("\n  speaker disjointness asserted: no speaker id appears in more than one partition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
