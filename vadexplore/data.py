"""Training data layer: split-aware dataset, normalization, and collation.

Everything is built on the existing pieces. Audio comes from
`loader.read_audio`, conditioning from `preprocess.highpass`, features from
`features.logmel`, and labels from `labels.make_labels`. Nothing here
reimplements a spectrogram or a rasterizer.

The frame grid is fixed project-wide: 16 kHz, 10 ms hop, 25 ms window, 40
log-mel bins, 100 frames per second.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vadexplore.config import DEFAULT_CONFIG, DataConfig
from vadexplore.features import logmel
from vadexplore.labels import make_labels, n_frames_for
from vadexplore.loader import load_clip, read_audio
from vadexplore.preprocess import highpass

DEFAULT_SPLIT = "splits/split.json"
DEFAULT_STATS = "splits/feature_stats.json"
CONVENTIONS = ("literal", "bridged")


def _resolve(path) -> Path:
    return Path(os.path.expanduser(str(path)))


def load_split(path=DEFAULT_SPLIT) -> dict:
    """Read the frozen split, failing clearly if it has not been made yet."""
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen split at {path}\n"
            "  create it once with: python scripts/make_split.py"
        )
    return json.loads(path.read_text())


def partition_stems(split: dict, partition: str) -> list[str]:
    if partition not in split["partitions"]:
        raise KeyError(f"unknown partition {partition!r}, "
                       f"expected one of {sorted(split['partitions'])}")
    return list(split["partitions"][partition]["stems"])


# --- per-clip feature and label construction ------------------------------


def clip_features(clip, config: DataConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Log-mel of the conditioned audio, shape (n_frames, n_mels).

    Transposed relative to `logmel`, because the model consumes time-major
    sequences.
    """
    audio = read_audio(clip, target_sr=config.sample_rate)
    audio = highpass(audio, config.sample_rate, config.highpass_hz, config.highpass_order)
    mel = logmel(
        audio,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        win_ms=config.win_ms,
        hop_ms=config.hop_ms,
        n_frames=n_frames_for(clip.duration_s, config.fps),
    )
    return mel.T.astype(np.float32)


def clip_example(clip, convention: str, config: DataConfig = DEFAULT_CONFIG):
    """Features and labels for one clip, guaranteed to share a frame count.

    `logmel` is asked for exactly `n_frames_for(duration)` frames and
    `make_labels` derives its grid from the same call, so in practice they
    already agree. The truncation below is a guard, not routine behavior: if
    the two ever disagree, both are cut to the shorter length rather than one
    being padded, because padding would invent a labeled frame with no audio
    behind it. Cutting loses at most 10 ms from the end of a clip.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

    features = clip_features(clip, config)
    labels = make_labels(clip, fps=config.fps, bridge_gap_s=config.bridge_gap_s)[convention]

    n = min(len(features), len(labels))
    features, labels = features[:n], labels[:n]
    assert len(features) == len(labels) == n
    return features, np.asarray(labels, dtype=np.int64)


# --- training statistics --------------------------------------------------


def training_feature_stats(
    split=DEFAULT_SPLIT,
    config: DataConfig = DEFAULT_CONFIG,
    dataset_dir=None,
    save_to=DEFAULT_STATS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-mel-bin mean and std over the training partition only.

    Train statistics only, applied unchanged to val and test. Recomputing
    statistics per partition would let the evaluation sets set their own scale,
    which leaks information about the test distribution into the input the
    model sees and quietly flatters any result measured on it. Per-clip
    normalization has the same problem in a subtler form: it would remove
    absolute level, which is part of what separates speech from silence.

    Accumulated in one pass with running sums, so the training partition never
    has to be held in memory at once.
    """
    split_data = split if isinstance(split, dict) else load_split(split)
    directory = _resolve(dataset_dir or split_data["dataset_dir"])
    stems = partition_stems(split_data, "train")

    total = np.zeros(config.n_mels, dtype=np.float64)
    total_sq = np.zeros(config.n_mels, dtype=np.float64)
    n_frames = 0
    for stem in stems:
        features = clip_features(load_clip(directory / stem), config).astype(np.float64)
        total += features.sum(axis=0)
        total_sq += (features ** 2).sum(axis=0)
        n_frames += len(features)

    if n_frames == 0:
        raise ValueError("training partition produced no frames")

    mean = total / n_frames
    var = np.maximum(total_sq / n_frames - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0  # a constant bin must not blow up the normalizer

    if save_to is not None:
        path = _resolve(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "computed_on": "train partition only",
            "why": "train-only statistics prevent test-scale leakage",
            "n_clips": len(stems),
            "n_frames": int(n_frames),
            "config": config.to_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }, indent=2))

    return mean.astype(np.float32), std.astype(np.float32)


def load_feature_stats(path=DEFAULT_STATS) -> tuple[np.ndarray, np.ndarray]:
    """Read saved training statistics."""
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no feature statistics at {path}\n"
            "  create them once with: python -c \"from vadexplore.data import "
            "training_feature_stats; training_feature_stats()\""
        )
    data = json.loads(path.read_text())
    return (np.asarray(data["mean"], dtype=np.float32),
            np.asarray(data["std"], dtype=np.float32))


# --- dataset --------------------------------------------------------------


class VADDataset(Dataset):
    """One frozen split partition under one label convention.

    Returns per clip a dict with `features` of shape (n_frames, n_mels),
    `labels` of shape (n_frames,), and `stem`. Features are normalized with
    training-partition statistics regardless of which partition this is.
    """

    def __init__(
        self,
        partition: str,
        convention: str = "bridged",
        split=DEFAULT_SPLIT,
        config: DataConfig = DEFAULT_CONFIG,
        stats: tuple[np.ndarray, np.ndarray] | None = None,
        dataset_dir=None,
    ):
        if convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

        self.split = split if isinstance(split, dict) else load_split(split)
        self.partition = partition
        self.convention = convention
        self.config = config
        self.stems = partition_stems(self.split, partition)
        self.directory = _resolve(dataset_dir or self.split["dataset_dir"])

        if stats is None:
            stats = load_feature_stats()
        self.mean, self.std = np.asarray(stats[0], np.float32), np.asarray(stats[1], np.float32)
        if self.mean.shape != (config.n_mels,) or self.std.shape != (config.n_mels,):
            raise ValueError(
                f"stats shape {self.mean.shape} does not match n_mels {config.n_mels}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict:
        stem = self.stems[index]
        clip = load_clip(self.directory / stem)
        features, labels = clip_example(clip, self.convention, self.config)
        features = (features - self.mean) / self.std
        return {
            "features": torch.from_numpy(np.ascontiguousarray(features)),
            "labels": torch.from_numpy(labels),
            "stem": stem,
        }


def collate(batch: list[dict], ignore_index: int | None = None) -> dict:
    """Pad a batch to its longest clip and mark the real frames.

    Returns `features` (B, T, n_mels), `labels` (B, T), `mask` (B, T) with True
    on real frames, `lengths` (B,), and the list of stems. Padded feature rows
    are zeros and padded label positions carry `ignore_index`, so a masked loss
    and an ignore-index loss both work without the caller reshaping anything.
    """
    if ignore_index is None:
        ignore_index = DEFAULT_CONFIG.ignore_index

    lengths = torch.tensor([len(item["features"]) for item in batch], dtype=torch.long)
    batch_size, max_len = len(batch), int(lengths.max())
    n_mels = batch[0]["features"].shape[1]

    features = torch.zeros(batch_size, max_len, n_mels, dtype=torch.float32)
    labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = int(lengths[i])
        features[i, :n] = item["features"]
        labels[i, :n] = item["labels"]
        mask[i, :n] = True

    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "lengths": lengths,
        "stems": [item["stem"] for item in batch],
    }


# --- sanity accessor ------------------------------------------------------


def describe_split(
    split=DEFAULT_SPLIT,
    config: DataConfig = DEFAULT_CONFIG,
    dataset_dir=None,
    printout: bool = True,
) -> dict:
    """Clip count, hours, and speech fraction per partition and convention.

    Reads labels only, no audio, so it is cheap. The speech fractions are the
    check that the committed labels still match the exploration numbers.
    """
    split_data = split if isinstance(split, dict) else load_split(split)
    directory = _resolve(dataset_dir or split_data["dataset_dir"])

    out = {"bridge_gap_s": config.bridge_gap_s, "partitions": {}}
    for partition, info in split_data["partitions"].items():
        speech = {c: 0 for c in CONVENTIONS}
        frames = 0
        seconds = 0.0
        for stem in info["stems"]:
            clip = load_clip(directory / stem)
            labels = make_labels(clip, fps=config.fps, bridge_gap_s=config.bridge_gap_s)
            frames += labels["n_frames"]
            seconds += clip.duration_s
            for convention in CONVENTIONS:
                speech[convention] += int(labels[convention].sum())
        out["partitions"][partition] = {
            "n_speakers": info["n_speakers"],
            "n_clips": len(info["stems"]),
            "hours": seconds / 3600.0,
            "n_frames": frames,
            "speech_fraction": {c: speech[c] / frames if frames else 0.0 for c in CONVENTIONS},
        }

    total_frames = sum(v["n_frames"] for v in out["partitions"].values())
    out["overall_speech_fraction"] = {
        c: sum(v["speech_fraction"][c] * v["n_frames"] for v in out["partitions"].values())
        / total_frames if total_frames else 0.0
        for c in CONVENTIONS
    }

    if printout:
        print(f"split summary (bridge_gap_s = {config.bridge_gap_s:g})")
        print(f"  {'partition':<10} {'spk':>4} {'clips':>6} {'hours':>7} {'frames':>9} "
              f"{'literal':>9} {'bridged':>9}")
        for partition, v in out["partitions"].items():
            print(f"  {partition:<10} {v['n_speakers']:4d} {v['n_clips']:6d} {v['hours']:7.2f} "
                  f"{v['n_frames']:9d} {v['speech_fraction']['literal'] * 100:8.1f}% "
                  f"{v['speech_fraction']['bridged'] * 100:8.1f}%")
        print(f"  {'overall':<10} {'':>4} {'':>6} {'':>7} {total_frames:9d} "
              f"{out['overall_speech_fraction']['literal'] * 100:8.1f}% "
              f"{out['overall_speech_fraction']['bridged'] * 100:8.1f}%")

    return out
