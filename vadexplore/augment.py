"""Reverb and noise augmentation on the waveform. Labels are never touched."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vadexplore.config import DataConfig
from vadexplore.features import logmel
from vadexplore.labels import make_labels, n_frames_for
from vadexplore.loader import read_audio
from vadexplore.preprocess import highpass

TARGET_SR = 16000
RIR_CATEGORIES = ("target", "noise")   # echo is a near-field loudspeaker path, not ours
SPLITS = ("train", "hard")
MUSAN_TRAIN_PERCENT = 80

# The direct arrival is rendered as a fractional-delay sinc centred on
# direct_path_sample, so the direct path occupies a window around it rather
# than a single sample.
DIRECT_PATH_HALF_WIDTH = 40


def _resolve(path) -> Path:
    return Path(os.path.expanduser(str(path)))


class RIRBank:
    """Room impulse responses for one category and one split. echo is excluded."""

    def __init__(self, rir_dir, category: str = "target", split: str = "train"):
        if category not in RIR_CATEGORIES:
            raise ValueError(f"category must be one of {RIR_CATEGORIES} "
                             f"(echo is deliberately excluded), got {category!r}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.root = _resolve(rir_dir)
        self.category = category
        self.split = split

        metadata = pd.read_csv(self.root / "metadata.csv")
        rows = metadata[(metadata.category == category) & (metadata.split == split)]
        if rows.empty:
            raise ValueError(f"no {category} RIRs in split {split} under {self.root}")
        self.table = rows.reset_index(drop=True)
        self.ids = self.table.id.tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def path(self, rir_id: int) -> Path:
        return self.root / self.category / f"{int(rir_id):04d}.npy"

    @lru_cache(maxsize=256)
    def _load(self, rir_id: int, direct_path: int) -> np.ndarray:
        """Load a RIR normalized so its direct path has unit energy."""
        impulse = np.load(self.path(rir_id)).astype(np.float32).reshape(-1)
        lo = max(0, direct_path - DIRECT_PATH_HALF_WIDTH)
        hi = min(len(impulse), direct_path + DIRECT_PATH_HALF_WIDTH + 1)
        direct_energy = float(np.sqrt(np.sum(impulse[lo:hi] ** 2)))
        if direct_energy <= 0:
            raise ValueError(f"RIR {rir_id} has no direct path energy")
        return impulse / direct_energy

    def get(self, rir_id: int) -> tuple[np.ndarray, int]:
        row = self.table[self.table.id == rir_id].iloc[0]
        direct_path = int(row.direct_path_sample)
        return self._load(int(rir_id), direct_path), direct_path

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
        rir_id = int(rng.choice(self.ids))
        impulse, direct_path = self.get(rir_id)
        return impulse, direct_path, rir_id


def musan_pool(musan_dir, split: str = "train",
               train_percent: int = MUSAN_TRAIN_PERCENT) -> list[Path]:
    """MUSAN files for one pool, split by md5 of the path relative to the root.

    Independent of directory order, file count, and any seed.
    """
    if split not in ("train", "test"):
        raise ValueError(f"MUSAN split must be train or test, got {split!r}")

    root = _resolve(musan_dir)
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise ValueError(f"no wav files under {root}")

    chosen = []
    for path in files:
        key = str(path.relative_to(root))
        bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
        if (bucket < train_percent) == (split == "train"):
            chosen.append(path)
    if not chosen:
        raise ValueError(f"MUSAN {split} pool is empty under {root}")
    return chosen


@lru_cache(maxsize=64)
def _read_musan(path_str: str) -> np.ndarray:
    audio, sr = sf.read(path_str, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        from vadexplore.loader import _resample
        audio = _resample(audio.astype(np.float64), sr, TARGET_SR).astype(np.float32)
    return audio


def apply_rir(audio: np.ndarray, impulse: np.ndarray, direct_path: int) -> np.ndarray:
    """Convolve and realign on the direct path, preserving length.

    Slicing from direct_path undoes the propagation delay; without it the whole
    utterance slides later by tens of milliseconds and the labels stop matching.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    wet = np.convolve(audio, np.asarray(impulse, dtype=np.float32))
    aligned = wet[direct_path:direct_path + len(audio)]
    if len(aligned) < len(audio):
        aligned = np.pad(aligned, (0, len(audio) - len(aligned)))
    return aligned.astype(np.float32)


def match_level(reverberant: np.ndarray, dry: np.ndarray,
                speech_mask: np.ndarray) -> np.ndarray:
    """Rescale a reverberated signal to the dry speech-active RMS.

    Direct-path normalization fixes the direct arrival, not the total: overall
    gain runs +0.7 to +17.5 dB across this bank, a second uncontrolled variable.
    """
    where = speech_mask if speech_mask.any() else np.ones(len(dry), bool)
    dry_rms = float(np.sqrt(np.mean(dry[where] ** 2)))
    wet_rms = float(np.sqrt(np.mean(reverberant[where] ** 2)))
    if wet_rms <= 0 or dry_rms <= 0:
        return reverberant
    return (reverberant * (dry_rms / wet_rms)).astype(np.float32)


def speech_mask_samples(labels: np.ndarray, n_samples: int,
                        fps: int = 100, sr: int = TARGET_SR) -> np.ndarray:
    """Expand a frame-level label array to a sample-level boolean mask."""
    hop = sr // fps
    mask = np.repeat(np.asarray(labels).astype(bool), hop)
    if len(mask) < n_samples:
        mask = np.pad(mask, (0, n_samples - len(mask)))
    return mask[:n_samples]


def fit_noise(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Crop or loop a noise recording to the required length."""
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(noise) >= length:
        start = int(rng.integers(0, len(noise) - length + 1))
        return noise[start:start + length].copy()
    repeats = int(np.ceil(length / len(noise)))
    return np.tile(noise, repeats)[:length].copy()


def add_noise_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
                     speech_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Mix noise into speech at an SNR measured over speech-active samples.

    Whole-clip power would make the realized SNR depend on how talkative the clip
    is; clips here run 49 to 93 percent speech.
    """
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    where = speech_mask if speech_mask.any() else np.ones(len(speech), bool)
    speech_power = float(np.mean(speech[where] ** 2))
    # the noise level is measured over the same frames, otherwise the realized
    # SNR drifts by however much the noise differs between the speech-active
    # region and the silences
    noise_power = float(np.mean(noise[where] ** 2))
    if speech_power <= 0 or noise_power <= 0:
        return speech, float("nan")

    scale = float(np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0))))
    return (speech + scale * noise).astype(np.float32), scale


@dataclass
class AugmentConfig:
    """One augmentation setting. `enabled=False` is a pass-through."""

    enabled: bool = False
    rir_dir: str = "path/to/rirs"
    musan_dir: str = "path/to/musan"
    rir_split: str = "train"        # train for training, hard for test conditions
    musan_split: str = "train"

    preserve_level: bool = True     # see match_level, and the note in Augmenter
    reverb_prob: float = 0.5
    noise_prob: float = 0.8
    noise_rir_prob: float = 0.5     # put the interferer in the room too
    snr_db_range: tuple = (0.0, 20.0)
    seed: int = 0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["snr_db_range"] = list(self.snr_db_range)
        return out


@dataclass
class AugmentRecord:
    """What was actually applied, for reproducibility and for my report."""

    reverb: bool = False
    rir_id: int | None = None
    noise: bool = False
    noise_file: str | None = None
    noise_rir_id: int | None = None
    snr_db: float | None = None


class Augmenter:
    """Applies reverb then noise to a waveform, leaving labels untouched."""

    def __init__(self, config: AugmentConfig):
        self.config = config
        self.target_bank = RIRBank(config.rir_dir, "target", config.rir_split)
        self.noise_bank = RIRBank(config.rir_dir, "noise", config.rir_split)
        self.musan = musan_pool(config.musan_dir, config.musan_split)

    def __call__(self, audio: np.ndarray, labels: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, AugmentRecord]:
        record = AugmentRecord()
        out = np.asarray(audio, dtype=np.float32).copy()
        mask = speech_mask_samples(labels, len(out))

        if rng.random() < self.config.reverb_prob:
            impulse, direct_path, rir_id = self.target_bank.sample(rng)
            wet = apply_rir(out, impulse, direct_path)
            if self.config.preserve_level:
                wet = match_level(wet, out, mask)
            out = wet
            record.reverb, record.rir_id = True, rir_id

        if rng.random() < self.config.noise_prob:
            path = self.musan[int(rng.integers(0, len(self.musan)))]
            noise = fit_noise(_read_musan(str(path)), len(out), rng)
            if rng.random() < self.config.noise_rir_prob:
                impulse, direct_path, noise_rir_id = self.noise_bank.sample(rng)
                noise = apply_rir(noise, impulse, direct_path)
                record.noise_rir_id = noise_rir_id
            low, high = self.config.snr_db_range
            snr_db = float(rng.uniform(low, high))
            out, _ = add_noise_at_snr(out, noise, snr_db, mask)
            record.noise, record.noise_file = True, path.name
            record.snr_db = snr_db

        return out, record


def features_from_audio(audio: np.ndarray, n_frames: int,
                        config: DataConfig, stats) -> np.ndarray:
    """The front-end applied to an already-augmented waveform."""
    conditioned = highpass(audio, config.sample_rate, config.highpass_hz,
                           config.highpass_order)
    mel = logmel(conditioned, sr=config.sample_rate, n_mels=config.n_mels,
                 win_ms=config.win_ms, hop_ms=config.hop_ms, n_frames=n_frames).T
    mean, std = stats
    return ((mel - mean) / std).astype(np.float32)


class AugmentedDataset:
    """Wraps a VADDataset, augmenting the waveform before the front-end.

    (seed, epoch, index) reproduces a draw exactly. Labels come from the clean
    source timestamps and are returned unchanged.
    """

    def __init__(self, base, augmenter: Augmenter, config: DataConfig,
                 epoch: int = 0):
        self.base = base
        self.augmenter = augmenter
        self.config = config
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(
            [self.augmenter.config.seed, self.epoch, index])

    def __getitem__(self, index: int) -> dict:
        import torch

        from vadexplore.loader import load_clip

        stem = self.base.stems[index]
        clip = load_clip(self.base.directory / stem)
        labels = make_labels(clip, fps=self.config.fps,
                             bridge_gap_s=self.config.bridge_gap_s)[self.base.convention]

        audio = read_audio(clip, target_sr=self.config.sample_rate)
        augmented, _ = self.augmenter(audio, labels, self._rng(index))

        n_frames = n_frames_for(clip.duration_s, self.config.fps)
        features = features_from_audio(augmented, n_frames, self.config,
                                       (self.base.mean, self.base.std))
        n = min(len(features), len(labels))
        return {
            "features": torch.from_numpy(np.ascontiguousarray(features[:n])),
            "labels": torch.from_numpy(np.asarray(labels[:n], dtype=np.int64)),
            "stem": stem,
        }


def augmented_audio(clip, labels: np.ndarray, augmenter: Augmenter,
                    config: DataConfig, index: int) -> tuple[np.ndarray, AugmentRecord]:
    """Deterministic augmented waveform, for fixed test conditions.

    Seeded by (seed, index) with no epoch term, so a condition is reproducible.
    """
    audio = read_audio(clip, target_sr=config.sample_rate)
    rng = np.random.default_rng([augmenter.config.seed, index])
    return augmenter(audio, labels, rng)
