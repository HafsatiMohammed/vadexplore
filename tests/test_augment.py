"""Tests for noise and reverberation augmentation.

The properties that matter are the ones that could silently corrupt an
experiment: a realized SNR that is not what was asked for, a convolution that
slides speech away from its labels, and train and test resources that overlap.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vadexplore.augment import (
    DIRECT_PATH_HALF_WIDTH,
    AugmentConfig,
    Augmenter,
    RIRBank,
    add_noise_at_snr,
    apply_rir,
    augmented_audio,
    fit_noise,
    match_level,
    musan_pool,
    speech_mask_samples,
)
from vadexplore.config import DataConfig
from vadexplore.labels import make_labels
from vadexplore.loader import load_clip, read_audio

RIR_DIR = Path(os.path.expanduser(
    "~/Documents/research_training/kws-augmentation-kit/rirs"))
MUSAN_DIR = Path(os.path.expanduser(
    "~/Documents/research_training/kws-augmentation-kit/musan"))

needs_rirs = pytest.mark.skipif(not (RIR_DIR / "metadata.csv").exists(),
                                reason="RIR bank not present")
needs_musan = pytest.mark.skipif(not MUSAN_DIR.exists(), reason="MUSAN not present")

SR = 16000


def synthetic_speech(seconds: float = 3.0, seed: int = 0):
    """A speech-like signal with known active regions, plus its frame labels."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    audio = rng.normal(0, 0.002, n).astype(np.float32)
    labels = np.zeros(int(seconds * 100), dtype=bool)
    for start, end in ((0.4, 1.2), (1.8, 2.6)):
        lo, hi = int(start * SR), int(end * SR)
        t = np.arange(hi - lo) / SR
        audio[lo:hi] += (0.2 * np.sin(2 * np.pi * 180 * t)
                         * (1 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
        labels[int(start * 100):int(end * 100)] = True
    return audio, labels


# --- additive noise hits the requested SNR --------------------------------


@pytest.mark.parametrize("target_snr", [-5.0, 0.0, 5.0, 10.0, 20.0])
def test_noise_hits_the_requested_snr_over_speech_active_frames(target_snr):
    audio, labels = synthetic_speech()
    mask = speech_mask_samples(labels, len(audio))
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, len(audio)).astype(np.float32)

    mixed, _ = add_noise_at_snr(audio, noise, target_snr, mask)
    added = mixed - audio

    realized = 10 * np.log10(np.mean(audio[mask] ** 2) / np.mean(added[mask] ** 2))
    assert realized == pytest.approx(target_snr, abs=0.05)


def test_snr_is_measured_over_speech_not_the_whole_clip():
    """Measuring over the whole clip would let the silence fraction set the level.

    Two clips with the same speech and different amounts of silence must get
    the same realized SNR over their speech.
    """
    audio, labels = synthetic_speech(seconds=3.0)
    padded = np.concatenate([audio, np.zeros(3 * SR, np.float32)])
    padded_labels = np.concatenate([labels, np.zeros(300, bool)])

    rng = np.random.default_rng(2)
    noise_short = rng.normal(0, 0.05, len(audio)).astype(np.float32)
    noise_long = np.concatenate([noise_short, rng.normal(0, 0.05, 3 * SR).astype(np.float32)])

    a, _ = add_noise_at_snr(audio, noise_short, 10.0,
                            speech_mask_samples(labels, len(audio)))
    b, _ = add_noise_at_snr(padded, noise_long, 10.0,
                            speech_mask_samples(padded_labels, len(padded)))

    mask = speech_mask_samples(labels, len(audio))
    snr_a = 10 * np.log10(np.mean(audio[mask] ** 2) / np.mean((a - audio)[mask] ** 2))
    snr_b = 10 * np.log10(np.mean(audio[mask] ** 2)
                          / np.mean((b[:len(audio)] - audio)[mask] ** 2))
    assert snr_a == pytest.approx(snr_b, abs=0.1)


def test_fit_noise_crops_and_loops_to_length():
    rng = np.random.default_rng(3)
    long_noise = rng.normal(0, 1, 5000).astype(np.float32)
    short_noise = rng.normal(0, 1, 300).astype(np.float32)
    assert len(fit_noise(long_noise, 1000, rng)) == 1000
    assert len(fit_noise(short_noise, 1000, rng)) == 1000
    assert len(fit_noise(short_noise, 100, rng)) == 100


# --- reverberation preserves level and alignment --------------------------


@needs_rirs
def test_rir_is_normalized_to_unit_direct_path_energy():
    bank = RIRBank(RIR_DIR, "target", "train")
    for rir_id in bank.ids[:10]:
        impulse, direct_path = bank.get(rir_id)
        lo = max(0, direct_path - DIRECT_PATH_HALF_WIDTH)
        hi = min(len(impulse), direct_path + DIRECT_PATH_HALF_WIDTH + 1)
        assert float(np.sum(impulse[lo:hi] ** 2)) == pytest.approx(1.0, abs=1e-5)


@needs_rirs
def test_rir_convolution_aligns_onset_to_the_direct_path():
    """An impulse must come out where it went in, not delayed by time of flight."""
    bank = RIRBank(RIR_DIR, "target", "train")
    for rir_id in bank.ids[:10]:
        impulse, direct_path = bank.get(rir_id)
        probe = np.zeros(8000, dtype=np.float32)
        probe[4000] = 1.0
        out = apply_rir(probe, impulse, direct_path)
        assert len(out) == len(probe)
        assert int(np.argmax(np.abs(out))) == 4000, f"RIR {rir_id} moved the onset"


@needs_rirs
def test_rir_convolution_does_not_move_speech_onset():
    audio, labels = synthetic_speech()
    bank = RIRBank(RIR_DIR, "target", "train")
    impulse, direct_path = bank.get(bank.ids[0])
    wet = apply_rir(audio, impulse, direct_path)

    # the first speech onset is at 0.4 s; find it by energy in both signals
    def onset(signal):
        envelope = np.convolve(signal ** 2, np.ones(160) / 160, mode="same")
        return int(np.argmax(envelope > envelope.max() * 0.05))

    assert abs(onset(wet) - onset(audio)) <= 160   # within 10 ms, one frame


@needs_rirs
def test_level_matching_preserves_speech_level_exactly():
    """Direct-path normalization alone does not, in this bank."""
    audio, labels = synthetic_speech()
    mask = speech_mask_samples(labels, len(audio))
    bank = RIRBank(RIR_DIR, "target", "train")

    raw_gains, matched_gains = [], []
    for rir_id in bank.ids[:8]:
        impulse, direct_path = bank.get(rir_id)
        wet = apply_rir(audio, impulse, direct_path)
        matched = match_level(wet, audio, mask)
        reference = float(np.sqrt(np.mean(audio[mask] ** 2)))
        raw_gains.append(float(np.sqrt(np.mean(wet[mask] ** 2))) / reference)
        matched_gains.append(float(np.sqrt(np.mean(matched[mask] ** 2))) / reference)

    assert all(g == pytest.approx(1.0, abs=1e-4) for g in matched_gains)
    # and the correction is doing real work: these rooms are reverberant
    assert max(raw_gains) > 1.5


@needs_rirs
def test_echo_category_is_refused():
    """Echo is the device loudspeaker path and must never enter a VAD pipeline."""
    with pytest.raises(ValueError, match="echo is deliberately excluded"):
        RIRBank(RIR_DIR, "echo", "train")


# --- labels are untouched -------------------------------------------------


@needs_rirs
@needs_musan
def test_augmentation_leaves_labels_identical():
    config = DataConfig()
    clip = load_clip(Path(os.path.expanduser("~/Downloads/vad_data")) /
                     "1447-130552-0010") if Path(os.path.expanduser(
                         "~/Downloads/vad_data/1447-130552-0010.wav")).exists() else None
    if clip is None:
        pytest.skip("dataset not present")

    before = make_labels(clip, fps=config.fps, bridge_gap_s=config.bridge_gap_s)
    augmenter = Augmenter(AugmentConfig(enabled=True, rir_dir=str(RIR_DIR),
                                        musan_dir=str(MUSAN_DIR)))
    audio, record = augmented_audio(clip, before["bridged"], augmenter, config, 0)
    after = make_labels(clip, fps=config.fps, bridge_gap_s=config.bridge_gap_s)

    assert np.array_equal(before["bridged"], after["bridged"])
    assert np.array_equal(before["literal"], after["literal"])
    assert before["segments_bridged"] == after["segments_bridged"]
    assert len(audio) == len(read_audio(clip))


# --- resource disjointness ------------------------------------------------


@needs_rirs
def test_rir_train_and_hard_splits_are_disjoint():
    for category in ("target", "noise"):
        train = set(RIRBank(RIR_DIR, category, "train").ids)
        hard = set(RIRBank(RIR_DIR, category, "hard").ids)
        assert train and hard
        assert not (train & hard), f"{category} RIRs leak between splits"


@needs_musan
def test_musan_train_and_test_pools_are_disjoint_and_cover_everything():
    train = set(musan_pool(MUSAN_DIR, "train"))
    test = set(musan_pool(MUSAN_DIR, "test"))
    everything = set(MUSAN_DIR.rglob("*.wav"))

    assert train and test
    assert not (train & test), "a MUSAN file is in both pools"
    assert train | test == everything, "the split does not cover every file"
    assert 0.7 < len(train) / len(everything) < 0.9


@needs_musan
def test_musan_split_is_stable_across_calls():
    """The partition must not depend on ordering, count, or any seed."""
    first = musan_pool(MUSAN_DIR, "train")
    second = musan_pool(MUSAN_DIR, "train")
    assert first == second


# --- reproducibility ------------------------------------------------------


@needs_rirs
@needs_musan
def test_augmentation_is_reproducible_under_a_fixed_seed():
    audio, labels = synthetic_speech()
    augmenter = Augmenter(AugmentConfig(enabled=True, rir_dir=str(RIR_DIR),
                                        musan_dir=str(MUSAN_DIR)))

    first, record_a = augmenter(audio, labels, np.random.default_rng([0, 0, 7]))
    second, record_b = augmenter(audio, labels, np.random.default_rng([0, 0, 7]))
    assert np.array_equal(first, second)
    assert record_a == record_b

    other, record_c = augmenter(audio, labels, np.random.default_rng([0, 1, 7]))
    assert not np.array_equal(first, other), "different seeds gave the same draw"


@needs_rirs
@needs_musan
def test_augmentation_preserves_length_and_stays_finite():
    audio, labels = synthetic_speech()
    augmenter = Augmenter(AugmentConfig(enabled=True, rir_dir=str(RIR_DIR),
                                        musan_dir=str(MUSAN_DIR),
                                        reverb_prob=1.0, noise_prob=1.0))
    for index in range(6):
        out, record = augmenter(audio, labels, np.random.default_rng([0, 0, index]))
        assert len(out) == len(audio)
        assert np.all(np.isfinite(out))
        assert record.reverb and record.noise
        assert record.rir_id is not None and record.snr_db is not None


@needs_rirs
@needs_musan
def test_training_and_test_augmenters_share_no_resources():
    """The methodological claim, asserted end to end."""
    train = Augmenter(AugmentConfig(enabled=True, rir_dir=str(RIR_DIR),
                                    musan_dir=str(MUSAN_DIR),
                                    rir_split="train", musan_split="train"))
    test = Augmenter(AugmentConfig(enabled=True, rir_dir=str(RIR_DIR),
                                   musan_dir=str(MUSAN_DIR),
                                   rir_split="hard", musan_split="test"))

    assert not set(train.target_bank.ids) & set(test.target_bank.ids)
    assert not set(train.noise_bank.ids) & set(test.noise_bank.ids)
    assert not set(train.musan) & set(test.musan)
