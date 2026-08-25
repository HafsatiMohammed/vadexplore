"""Tests for the modeling data layer.

Most tests run on a synthetic corpus built in tmp_path so they stay fast and
do not depend on the real dataset. The leakage test deliberately reads the
committed splits/split.json, because that specific file is what every
experiment loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from vadexplore import data as D
from vadexplore.config import DataConfig
from vadexplore.labels import n_frames_for
from vadexplore.loader import load_clip
from vadexplore.preprocess import butter_highpass_sos, highpass

REPO_SPLIT = Path(__file__).resolve().parents[1] / "splits" / "split.json"
CONFIG = DataConfig()


# --- synthetic corpus -----------------------------------------------------


def write_clip(directory, stem, duration_s, segments, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sr))
    audio = rng.normal(0, 0.01, n).astype(np.float32)
    t = np.arange(n) / sr
    for start, end in segments:
        lo, hi = int(start * sr), min(n, int(end * sr))
        audio[lo:hi] += 0.3 * np.sin(2 * np.pi * 180 * t[lo:hi]).astype(np.float32)
    sf.write(str(directory / f"{stem}.wav"), audio, sr)
    (directory / f"{stem}.json").write_text(json.dumps(
        {"speech_segments": [{"start_time": s, "end_time": e} for s, e in segments]}))


@pytest.fixture
def corpus(tmp_path):
    """Three speakers, seven clips, varied lengths, plus a frozen split."""
    directory = tmp_path / "vad_data"
    directory.mkdir()
    spec = {
        "1000-1-0000": (2.00, [(0.2, 0.8), (0.86, 1.7)]),   # 60 ms gap, bridged at 100 ms
        "1000-1-0001": (1.50, [(0.1, 0.6), (1.0, 1.4)]),    # 400 ms gap, kept
        "1000-2-0001": (1.80, [(0.15, 0.7), (0.95, 1.6)]),  # 250 ms gap, kept
        "1000-2-0000": (3.017, [(0.5, 2.5)]),
        "2000-1-0000": (1.20, [(0.1, 1.0)]),
        "2000-1-0001": (2.40, [(0.3, 1.1), (1.16, 2.2)]),   # 60 ms gap, bridged at 100 ms
        "3000-1-0000": (1.00, [(0.2, 0.7)]),
        "3000-1-0001": (2.75, [(0.4, 1.5), (1.9, 2.6)]),
    }
    for i, (stem, (dur, segs)) in enumerate(spec.items()):
        write_clip(directory, stem, dur, segs, seed=i)

    split = {
        "dataset_dir": str(directory),
        "n_speakers": 3,
        "n_clips": len(spec),
        "partitions": {
            "train": {"speakers": ["1000"], "n_speakers": 1,
                      "stems": [s for s in spec if s.startswith("1000")]},
            "val": {"speakers": ["2000"], "n_speakers": 1,
                    "stems": [s for s in spec if s.startswith("2000")]},
            "test": {"speakers": ["3000"], "n_speakers": 1,
                     "stems": [s for s in spec if s.startswith("3000")]},
        },
    }
    for part in split["partitions"].values():
        part["n_clips"] = len(part["stems"])
    return {"dir": directory, "split": split, "spec": spec}


@pytest.fixture
def stats(corpus):
    return D.training_feature_stats(corpus["split"], CONFIG, save_to=None)


# --- 1. speaker-disjoint leakage, the primary test ------------------------


@pytest.mark.skipif(not REPO_SPLIT.exists(),
                    reason="run scripts/make_split.py to create splits/split.json")
def test_committed_split_has_no_speaker_leakage():
    split = json.loads(REPO_SPLIT.read_text())
    partitions = split["partitions"]

    seen = {}
    for name, info in partitions.items():
        for speaker in info["speakers"]:
            assert speaker not in seen, (
                f"speaker {speaker} leaks between {seen[speaker]} and {name}")
            seen[speaker] = name
    assert len(seen) == split["n_speakers"]

    # the same must hold at clip level: a stem's speaker prefix decides its partition
    for name, info in partitions.items():
        for stem in info["stems"]:
            assert seen[stem.split("-")[0]] == name, f"{stem} sits in the wrong partition"

    stems = [s for info in partitions.values() for s in info["stems"]]
    assert len(stems) == len(set(stems)) == split["n_clips"]


def test_synthetic_split_partitions_are_disjoint(corpus):
    partitions = corpus["split"]["partitions"]
    speakers = [set(info["speakers"]) for info in partitions.values()]
    for i, a in enumerate(speakers):
        for b in speakers[i + 1:]:
            assert not (a & b)


# --- 2. training statistics come from the train partition only ------------


def test_feature_stats_read_only_training_clips(corpus, monkeypatch):
    read = []
    original = D.load_clip

    def spy(path):
        read.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(D, "load_clip", spy)
    D.training_feature_stats(corpus["split"], CONFIG, save_to=None)

    train = set(corpus["split"]["partitions"]["train"]["stems"])
    held_out = (set(corpus["split"]["partitions"]["val"]["stems"])
                | set(corpus["split"]["partitions"]["test"]["stems"]))
    assert set(read) == train
    assert not (set(read) & held_out), f"held-out clips were read: {set(read) & held_out}"


def test_feature_stats_are_saved_with_provenance(corpus, tmp_path):
    target = tmp_path / "nested" / "feature_stats.json"
    mean, std = D.training_feature_stats(corpus["split"], CONFIG, save_to=target)
    saved = json.loads(target.read_text())
    assert saved["computed_on"] == "train partition only"
    assert saved["n_clips"] == len(corpus["split"]["partitions"]["train"]["stems"])
    assert np.allclose(saved["mean"], mean, atol=1e-5)
    assert np.allclose(saved["std"], std, atol=1e-5)


# --- 3. shapes ------------------------------------------------------------


@pytest.mark.parametrize("convention", ["literal", "bridged"])
def test_getitem_shapes(corpus, stats, convention):
    dataset = D.VADDataset("train", convention, corpus["split"], CONFIG, stats=stats)
    assert len(dataset) == 4

    for i in range(len(dataset)):
        item = dataset[i]
        features, labels = item["features"], item["labels"]
        assert features.ndim == 2 and features.shape[1] == CONFIG.n_mels
        assert labels.ndim == 1
        assert features.shape[0] == labels.shape[0]

        expected = n_frames_for(corpus["spec"][item["stem"]][0], CONFIG.fps)
        assert abs(features.shape[0] - expected) <= 1
        assert features.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert set(labels.tolist()) <= {0, 1}


def test_frame_count_matches_duration_exactly(corpus, stats):
    dataset = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    item = next(d for d in (dataset[i] for i in range(len(dataset)))
                if d["stem"] == "1000-2-0000")
    assert item["features"].shape[0] == round(3.017 * 100) == 302


# --- 4. collation ---------------------------------------------------------


def test_collate_pads_and_masks(corpus, stats):
    dataset = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    items = [dataset[i] for i in range(len(dataset))]
    true_lengths = [len(item["features"]) for item in items]
    assert len(set(true_lengths)) > 1, "fixture must have varied lengths"

    batch = D.collate(items)
    max_len = max(true_lengths)

    assert batch["features"].shape == (len(items), max_len, CONFIG.n_mels)
    assert batch["labels"].shape == (len(items), max_len)
    assert batch["mask"].shape == (len(items), max_len)
    assert batch["lengths"].tolist() == true_lengths
    assert batch["mask"].sum(dim=1).tolist() == true_lengths
    assert batch["stems"] == [item["stem"] for item in items]

    for i, n in enumerate(true_lengths):
        assert batch["mask"][i, :n].all() and not batch["mask"][i, n:].any()
        assert torch.equal(batch["labels"][i, :n], items[i]["labels"])
        assert (batch["labels"][i, n:] == CONFIG.ignore_index).all()
        assert (batch["features"][i, n:] == 0).all()

    # no real frame may carry the ignore value
    assert (batch["labels"][batch["mask"]] != CONFIG.ignore_index).all()


def test_collate_single_item_needs_no_padding(corpus, stats):
    dataset = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    batch = D.collate([dataset[0]])
    assert batch["mask"].all()
    assert (batch["labels"] != CONFIG.ignore_index).all()


def test_collate_honours_a_custom_ignore_index(corpus, stats):
    dataset = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    batch = D.collate([dataset[i] for i in range(3)], ignore_index=-1)
    assert (batch["labels"][~batch["mask"]] == -1).all()


def test_collate_works_as_a_dataloader_collate_fn(corpus, stats):
    from torch.utils.data import DataLoader
    dataset = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    loader = DataLoader(dataset, batch_size=2, collate_fn=D.collate)
    seen = 0
    for batch in loader:
        seen += batch["features"].shape[0]
        assert batch["features"].shape[1] == int(batch["lengths"].max())
    assert seen == len(dataset)


# --- 5. label convention --------------------------------------------------


def test_bridged_never_has_fewer_speech_frames_than_literal(corpus, stats):
    literal = D.VADDataset("train", "literal", corpus["split"], CONFIG, stats=stats)
    bridged = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)

    strictly_greater = 0
    for i in range(len(literal)):
        a, b = literal[i], bridged[i]
        assert a["stem"] == b["stem"]
        assert int(b["labels"].sum()) >= int(a["labels"].sum())
        # features are identical, only the labels differ
        assert torch.allclose(a["features"], b["features"])
        strictly_greater += int(b["labels"].sum()) > int(a["labels"].sum())
    assert strictly_greater > 0, "fixture must contain a gap short enough to bridge"


def test_unknown_convention_is_rejected(corpus, stats):
    with pytest.raises(ValueError, match="convention must be one of"):
        D.VADDataset("train", "smoothed", corpus["split"], CONFIG, stats=stats)


def test_unknown_partition_is_rejected(corpus, stats):
    with pytest.raises(KeyError, match="unknown partition"):
        D.VADDataset("dev", "bridged", corpus["split"], CONFIG, stats=stats)


# --- 6. normalization -----------------------------------------------------


def test_training_features_are_standardized(corpus, stats):
    dataset = D.VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    stacked = torch.cat([dataset[i]["features"] for i in range(len(dataset))]).numpy()

    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(stacked.std(axis=0), 1.0, atol=1e-4)


def test_val_and_test_reuse_training_stats_without_recomputing(corpus, stats):
    mean, std = stats
    for partition in ("val", "test"):
        dataset = D.VADDataset(partition, "bridged", corpus["split"], CONFIG, stats=stats)
        assert np.array_equal(dataset.mean, mean)
        assert np.array_equal(dataset.std, std)

        stacked = torch.cat([dataset[i]["features"] for i in range(len(dataset))]).numpy()
        # if stats had been recomputed per partition this would be exactly zero
        assert not np.allclose(stacked.mean(axis=0), 0.0, atol=1e-3)

        # the transform really is the frozen affine one
        raw = D.clip_features(load_clip(corpus["dir"] / dataset.stems[0]), CONFIG)
        assert np.allclose(dataset[0]["features"].numpy(), (raw - mean) / std, atol=1e-5)


def test_stats_shape_is_validated(corpus):
    with pytest.raises(ValueError, match="does not match n_mels"):
        D.VADDataset("train", "bridged", corpus["split"], CONFIG,
                     stats=(np.zeros(13, np.float32), np.ones(13, np.float32)))


# --- high-pass ------------------------------------------------------------


def test_highpass_matches_scipy_butterworth():
    scipy_signal = pytest.importorskip("scipy.signal")
    for order in (2, 4, 6):
        mine = butter_highpass_sos(80.0, 16000, order)
        reference = scipy_signal.butter(order, 80 / 8000, btype="high", output="sos")
        _, h_mine = scipy_signal.sosfreqz(mine, worN=2048, fs=16000)
        _, h_ref = scipy_signal.sosfreqz(reference, worN=2048, fs=16000)
        assert np.allclose(h_mine, h_ref, atol=1e-8)


def test_highpass_is_zero_phase():
    sr = 16000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = highpass(x, sr, 80.0, 2)
    mid = slice(sr // 4, 3 * sr // 4)
    lag = int(np.argmax(np.correlate(y[mid], x[mid], "full"))) - (len(x[mid]) - 1)
    assert lag == 0


def test_highpass_attenuates_below_cutoff_and_passes_above():
    sr = 16000
    t = np.arange(sr) / sr
    mid = slice(sr // 4, 3 * sr // 4)

    def gain_db(freq):
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)
        y = highpass(x, sr, 80.0, 2)
        return 20 * np.log10(np.abs(y[mid]).max() / np.abs(x[mid]).max())

    assert gain_db(30) < -25
    assert gain_db(80) == pytest.approx(-6.0, abs=0.5)  # squared by filtfilt
    assert gain_db(1000) == pytest.approx(0.0, abs=0.1)


def test_highpass_none_is_a_passthrough():
    x = np.random.default_rng(0).normal(0, 0.1, 8000).astype(np.float32)
    assert np.array_equal(highpass(x, 16000, None), x)


def test_highpass_config_disables_filtering(corpus, stats):
    off = DataConfig(highpass_hz=None)
    clip = load_clip(corpus["dir"] / "1000-2-0000")
    assert not np.allclose(D.clip_features(clip, CONFIG), D.clip_features(clip, off))


def test_odd_order_is_rejected():
    with pytest.raises(ValueError, match="even number"):
        butter_highpass_sos(80.0, 16000, 3)


# --- describe_split -------------------------------------------------------


def test_describe_split_reports_both_conventions(corpus):
    out = D.describe_split(corpus["split"], CONFIG, printout=False)
    assert set(out["partitions"]) == {"train", "val", "test"}
    for v in out["partitions"].values():
        assert v["speech_fraction"]["bridged"] >= v["speech_fraction"]["literal"]
        assert 0.0 < v["speech_fraction"]["literal"] < 1.0
    assert out["overall_speech_fraction"]["bridged"] >= out["overall_speech_fraction"]["literal"]
