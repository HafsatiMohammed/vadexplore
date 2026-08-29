"""Tests for the training path.

Runs on a synthetic corpus in tmp_path so nothing depends on the real dataset,
except one test that checks the committed split's pos_weight value.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from vadexplore import train as T
from vadexplore.config import DataConfig, ModelConfig
from vadexplore.data import collate, training_feature_stats
from vadexplore.model import VADModel, masked_bce_loss

REPO_SPLIT = Path(__file__).resolve().parents[1] / "splits" / "split.json"
CONFIG = DataConfig()


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
    directory = tmp_path / "vad_data"
    directory.mkdir()
    spec = {
        "1000-1-0000": (2.00, [(0.2, 0.8), (0.86, 1.7)]),
        "1000-1-0001": (1.50, [(0.1, 0.6), (1.0, 1.4)]),
        "1000-2-0000": (2.20, [(0.5, 1.9)]),
        "1000-2-0001": (1.80, [(0.15, 0.7), (0.95, 1.6)]),
        "2000-1-0000": (1.60, [(0.1, 1.2)]),
        "2000-1-0001": (2.00, [(0.3, 1.1), (1.16, 1.9)]),
        "3000-1-0000": (1.40, [(0.2, 1.1)]),
        "3000-1-0001": (1.90, [(0.4, 1.5)]),
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

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    return {"dir": directory, "split": split, "split_path": split_path, "tmp": tmp_path}


@pytest.fixture
def run_config(corpus):
    base = yaml.safe_load((Path(__file__).resolve().parents[1] /
                           "configs" / "train.yaml").read_text())
    config = copy.deepcopy(base)
    config["name"] = "unit"
    config["device"] = "cpu"
    config["out_root"] = str(corpus["tmp"] / "runs")
    config["data"]["split"] = str(corpus["split_path"])
    config["data"]["feature_stats"] = str(corpus["tmp"] / "feature_stats.json")
    config["optim"]["batch_size"] = 2
    config["optim"]["epochs"] = 2
    config["eval"]["early_stopping_patience"] = None
    config["model"]["dropout"] = 0.0
    config["model"]["attn_dropout"] = 0.0
    # These tests exercise the loop's bookkeeping on a synthetic corpus. The
    # shipped config turns augmentation on, which would reach for the external
    # RIR and MUSAN banks; augmentation has its own tests in test_augment.py.
    config["augment"]["enabled"] = False
    return config


# --- 1. pos_weight comes from the training split only ---------------------


def test_pos_weight_reads_only_training_clips(corpus, monkeypatch):
    read = []
    original = T.load_clip

    def spy(path):
        read.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(T, "load_clip", spy)
    T.compute_pos_weight(corpus["split"], "bridged", CONFIG)

    partitions = corpus["split"]["partitions"]
    train = set(partitions["train"]["stems"])
    held_out = set(partitions["val"]["stems"]) | set(partitions["test"]["stems"])
    assert set(read) == train
    assert not (set(read) & held_out), f"held-out clips were read: {set(read) & held_out}"


def test_pos_weight_matches_the_counted_frequencies(corpus):
    speech, non_speech = T.training_class_counts(corpus["split"], "bridged", CONFIG)
    assert T.compute_pos_weight(corpus["split"], "bridged", CONFIG) == pytest.approx(
        non_speech / speech)


@pytest.mark.skipif(not REPO_SPLIT.exists(), reason="run scripts/make_split.py first")
@pytest.mark.parametrize("convention, expected", [("literal", 0.2493), ("bridged", 0.2326)])
def test_pos_weight_on_the_committed_split(convention, expected):
    """The real training partition is about 80 percent speech, so pos_weight < 1.

    BCEWithLogitsLoss multiplies the positive term by pos_weight, and speech is
    the positive and majority class here, so inverse frequency damps it.
    """
    value = T.compute_pos_weight(REPO_SPLIT, convention, CONFIG)
    assert value == pytest.approx(expected, abs=0.005)
    assert value < 1.0


def test_unweighted_option_gives_pos_weight_one(corpus, run_config):
    run_config["loss"]["weighting"] = "none"
    run_config["optim"]["epochs"] = 1
    assert T.train(run_config, verbose=False)["pos_weight"] == 1.0


def test_unknown_weighting_is_rejected(run_config):
    run_config["loss"]["weighting"] = "focal"
    with pytest.raises(ValueError, match="loss.weighting"):
        T.train(run_config, verbose=False)


# --- 2. the loss actually goes down ---------------------------------------


@pytest.mark.parametrize("core", ["bigru", "causal_attn"])
def test_overfits_a_tiny_fixed_batch(corpus, core):
    """Many steps on a few clips must drive the loss down for both cores."""
    stats = training_feature_stats(corpus["split"], CONFIG, save_to=None)
    from vadexplore.data import VADDataset

    dataset = VADDataset("train", "bridged", corpus["split"], CONFIG, stats=stats)
    batch = collate([dataset[i] for i in range(len(dataset))])

    torch.manual_seed(0)
    model = VADModel(ModelConfig(temporal=core, dropout=0.0, attn_dropout=0.0))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(60):
        logits = model.forward_batch(batch)
        loss = masked_bce_loss(logits, batch["labels"], batch["mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.5, f"{core}: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(np.isfinite(losses))


# --- 3. a full short run produces reloadable artifacts --------------------


@pytest.mark.parametrize("core", ["bigru", "causal_attn"])
def test_short_run_writes_and_reloads(corpus, run_config, core):
    run_config["model"]["core"] = core
    run_config["name"] = f"unit_{core}"
    result = T.train(run_config, verbose=False)

    out_dir = Path(result["out_dir"])
    for filename in ("best.pt", "history.json", "config.resolved.yaml"):
        assert (out_dir / filename).exists(), f"missing {filename}"

    history = json.loads((out_dir / "history.json").read_text())
    assert len(history["epochs"]) == run_config["optim"]["epochs"]
    assert history["selection_metric"] == run_config["eval"]["select_on"]
    for record in history["epochs"]:
        for key in ("train_loss", "val_loss", "eer", "auc", "frr_at_fa"):
            assert key in record and np.isfinite(record[key])

    resolved = yaml.safe_load((out_dir / "config.resolved.yaml").read_text())
    assert resolved["model"]["core"] == core
    assert resolved["_resolved"]["pos_weight"] == pytest.approx(result["pos_weight"])

    model, payload = T.load_checkpoint(out_dir / "best.pt")
    assert payload["label_convention"] == run_config["data"]["label"]
    assert payload["model_config"]["temporal"] == core
    assert "feature_stats" in payload and len(payload["feature_stats"]["mean"]) == 40
    with torch.no_grad():
        assert model(torch.randn(1, 33, 40)).shape == (1, 33)


def test_checkpoint_tensors_are_on_cpu(corpus, run_config):
    run_config["optim"]["epochs"] = 1
    result = T.train(run_config, verbose=False)
    payload = torch.load(result["best_path"], map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in payload["state_dict"].values())


def test_run_requires_a_name(run_config):
    run_config["name"] = None
    with pytest.raises(ValueError, match="run name is required"):
        T.train(run_config, verbose=False)


# --- 4. reproducibility ---------------------------------------------------


def test_same_seed_gives_the_same_first_epoch_loss(corpus, run_config):
    run_config["optim"]["epochs"] = 1

    first = copy.deepcopy(run_config); first["name"] = "seed_a"
    second = copy.deepcopy(run_config); second["name"] = "seed_b"
    a = T.train(first, verbose=False)["history"][0]["train_loss"]
    b = T.train(second, verbose=False)["history"][0]["train_loss"]
    assert a == pytest.approx(b, rel=1e-6), f"{a} vs {b}"


def test_different_seeds_diverge(corpus, run_config):
    run_config["optim"]["epochs"] = 1

    first = copy.deepcopy(run_config); first["name"] = "seed_c"; first["seed"] = 0
    second = copy.deepcopy(run_config); second["name"] = "seed_d"; second["seed"] = 1
    a = T.train(first, verbose=False)["history"][0]["train_loss"]
    b = T.train(second, verbose=False)["history"][0]["train_loss"]
    assert a != pytest.approx(b, rel=1e-6)


# --- 5. masking: padding contributes nothing to the loss ------------------


@pytest.mark.parametrize("pos_weight", [None, 0.2326])
def test_loss_ignores_padding(pos_weight):
    """The padded loss must equal the loss over the unpadded clips."""
    torch.manual_seed(0)
    lengths = [23, 11, 17]
    items = [{"features": torch.randn(n, 40),
              "labels": torch.randint(0, 2, (n,)),
              "stem": f"clip-{n}"} for n in lengths]
    batch = collate(items)

    weight = None if pos_weight is None else torch.tensor(pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    logits = torch.randn(len(lengths), max(lengths))
    valid = batch["mask"].reshape(-1)
    padded_loss = criterion(logits.reshape(-1)[valid],
                            batch["labels"].reshape(-1)[valid].float())

    # the same frames gathered without ever building a padded tensor
    flat_logits = torch.cat([logits[i, :n] for i, n in enumerate(lengths)])
    flat_labels = torch.cat([items[i]["labels"] for i in range(len(lengths))]).float()
    unpadded_loss = criterion(flat_logits, flat_labels)

    assert torch.allclose(padded_loss, unpadded_loss, atol=1e-6)


def test_garbage_in_padding_does_not_change_the_loss():
    torch.manual_seed(1)
    items = [{"features": torch.randn(n, 40),
              "labels": torch.randint(0, 2, (n,)),
              "stem": f"clip-{n}"} for n in (20, 9)]
    batch = collate(items)
    logits = torch.randn(2, 20)

    clean = masked_bce_loss(logits, batch["labels"], batch["mask"])
    poisoned = logits.clone()
    poisoned[~batch["mask"]] = 1e3
    assert torch.allclose(clean, masked_bce_loss(poisoned, batch["labels"], batch["mask"]))


# --- 6. metrics -----------------------------------------------------------


def test_auc_and_eer_on_a_separable_case():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert T.roc_auc(scores, labels) == pytest.approx(1.0)
    assert T.equal_error_rate(scores, labels)[0] == pytest.approx(0.0, abs=1e-9)


def test_auc_on_a_random_case_is_near_half():
    rng = np.random.default_rng(0)
    scores = rng.random(20000)
    labels = rng.random(20000) < 0.5
    assert T.roc_auc(scores, labels) == pytest.approx(0.5, abs=0.02)


def test_false_alarm_events_counts_runs_not_frames():
    labels = np.zeros(10, dtype=bool)
    prediction = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 1], dtype=bool)
    starts = np.zeros(10, dtype=bool); starts[0] = True
    assert T.false_alarm_events(prediction, labels, starts) == 3


def test_false_alarm_events_do_not_merge_across_clips():
    labels = np.zeros(6, dtype=bool)
    prediction = np.ones(6, dtype=bool)
    starts = np.zeros(6, dtype=bool); starts[0] = True; starts[3] = True
    assert T.false_alarm_events(prediction, labels, starts) == 2


def test_short_false_alarms_are_ignored_below_the_minimum():
    """A 10 ms blip is not an audible false alarm."""
    labels = np.zeros(12, dtype=bool)
    prediction = np.array([1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    starts = np.zeros(12, dtype=bool); starts[0] = True

    assert T.false_alarm_events(prediction, labels, starts, min_frames=1) == 3
    assert T.false_alarm_events(prediction, labels, starts, min_frames=3) == 1
    assert T.false_alarm_events(prediction, labels, starts, min_frames=10) == 0


def test_frr_at_fa_reports_when_the_budget_cannot_be_met():
    """Unreachable only when even the strictest threshold fires on non-speech.

    A zero false-alarm budget is otherwise satisfiable by predicting nothing,
    which is feasible and useless rather than infeasible, so the two cases are
    tested apart.
    """
    starts = np.array([True, False, False])

    # top-scoring frame is non-speech, so every threshold produces a false alarm
    infeasible = T.frr_at_fa_per_hour(
        np.array([0.9, 0.1, 0.2]), np.array([False, True, True]), starts,
        hours=0.001, target_fa_per_hour=0.0, min_fa_frames=1)
    assert infeasible["budget_met"] is False
    assert infeasible["frr"] == 1.0

    # top-scoring frame is speech, so the budget is met at the top threshold
    feasible = T.frr_at_fa_per_hour(
        np.array([0.9, 0.1, 0.2]), np.array([True, False, False]), starts,
        hours=0.001, target_fa_per_hour=0.0, min_fa_frames=1)
    assert feasible["budget_met"] is True
    assert feasible["frr"] == pytest.approx(0.0)


def test_frr_at_fa_prefers_the_lowest_miss_rate_within_budget():
    """Among thresholds meeting the budget, the best miss rate wins."""
    rng = np.random.default_rng(3)
    n = 4000
    labels = rng.random(n) < 0.8
    scores = np.clip(rng.normal(np.where(labels, 0.8, 0.2), 0.1), 0, 1)
    starts = np.zeros(n, dtype=bool); starts[0] = True

    loose = T.frr_at_fa_per_hour(scores, labels, starts, hours=n / 100 / 3600,
                                 target_fa_per_hour=500.0, min_fa_frames=1)
    tight = T.frr_at_fa_per_hour(scores, labels, starts, hours=n / 100 / 3600,
                                 target_fa_per_hour=50.0, min_fa_frames=1)
    assert loose["frr"] <= tight["frr"]


# --- 7. config plumbing ---------------------------------------------------


def test_cli_flags_override_the_config_file():
    parser = T.build_parser()
    args = parser.parse_args([
        "--name", "attn_bridged", "--core", "causal_attn", "--label", "bridged",
        "--past-window-frames", "100", "--lookahead-frames", "5",
    ])
    base = yaml.safe_load((Path(__file__).resolve().parents[1] /
                           "configs" / "train.yaml").read_text())
    merged = T.apply_overrides(base, args)

    assert merged["name"] == "attn_bridged"
    assert merged["model"]["core"] == "causal_attn"
    assert merged["data"]["label"] == "bridged"
    assert merged["model"]["past_window_frames"] == 100
    assert merged["model"]["lookahead_frames"] == 5
    # untouched keys keep their file values
    assert merged["optim"]["batch_size"] == base["optim"]["batch_size"]

    _, model_config = T.build_configs(merged)
    assert model_config.temporal == "causal_attn"
    assert model_config.past_window_frames == 100


def test_absent_flags_do_not_override():
    args = T.build_parser().parse_args(["--name", "x"])
    base = yaml.safe_load((Path(__file__).resolve().parents[1] /
                           "configs" / "train.yaml").read_text())
    merged = T.apply_overrides(base, args)
    assert merged["model"]["core"] == base["model"]["core"]
    assert merged["data"]["label"] == base["data"]["label"]


def test_unknown_selection_metric_is_rejected(run_config):
    run_config["eval"]["select_on"] = "accuracy"
    with pytest.raises(ValueError, match="eval.select_on"):
        T.train(run_config, verbose=False)


# --- 8. device selection --------------------------------------------------


def test_auto_device_prefers_the_fastest_available():
    device = T.resolve_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_cpu_is_always_selectable():
    assert T.resolve_device("cpu").type == "cpu"


@pytest.mark.parametrize("backend, available, hint", [
    ("cuda", torch.cuda.is_available(), "set device: mps or auto"),
    ("mps", torch.backends.mps.is_available(), "set device: cuda or auto"),
])
def test_explicit_device_fails_loudly_when_missing(backend, available, hint):
    """A misconfigured GPU run must not silently become a slow cpu run."""
    if available:
        assert T.resolve_device(backend).type == backend
        return
    with pytest.raises(ValueError, match=f"{backend.upper()} requested but not available"):
        T.resolve_device(backend)
    with pytest.raises(ValueError, match=hint.split(";")[0].strip()[:10]):
        T.resolve_device(backend)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device must be cuda, mps, cpu, or auto"):
        T.resolve_device("tpu")


def test_describe_device_names_the_backend():
    text = T.describe_device(torch.device("cpu"))
    assert text.startswith("cpu:")
    if torch.backends.mps.is_available():
        assert "Apple Silicon" in T.describe_device(torch.device("mps"))
    if torch.cuda.is_available():
        assert T.describe_device(torch.device("cuda")).startswith("cuda:")


def test_lengths_reach_pack_padded_on_cpu_from_any_device():
    """pack_padded_sequence needs cpu lengths whatever device the data is on."""
    from vadexplore.model import VADModel
    device = T.resolve_device("auto")
    model = VADModel(ModelConfig(temporal="bigru")).to(device).eval()
    items = [{"features": torch.randn(n, 40), "labels": torch.randint(0, 2, (n,)),
              "stem": f"c{n}"} for n in (17, 9)]
    batch = collate(items)
    with torch.no_grad():
        logits = model(batch["features"].to(device), batch["mask"].to(device),
                       batch["lengths"].to(device))
    assert logits.shape == batch["labels"].shape


# --- 9. progress bars -----------------------------------------------------


def test_progress_follows_the_tty_and_the_flag(monkeypatch):
    monkeypatch.setattr(T.sys.stderr, "isatty", lambda: True, raising=False)
    assert T.progress_enabled("auto") is True
    assert T.progress_enabled("auto", no_progress_flag=True) is False
    monkeypatch.setattr(T.sys.stderr, "isatty", lambda: False, raising=False)
    assert T.progress_enabled("auto") is False
    # explicit settings win over the tty
    assert T.progress_enabled(True) is True
    assert T.progress_enabled(False) is False
    assert T.progress_enabled(True, no_progress_flag=True) is False


def test_no_progress_run_completes(corpus, run_config):
    run_config["optim"]["epochs"] = 1
    run_config["progress"] = True   # would enable bars, the flag must override
    result = T.train(run_config, verbose=False, no_progress=True)
    assert Path(result["best_path"]).exists()


def test_progress_does_not_change_the_result(corpus, run_config):
    """Observation only: bars must not move a single decimal place."""
    run_config["optim"]["epochs"] = 2

    quiet = copy.deepcopy(run_config); quiet["name"] = "p_off"; quiet["progress"] = False
    loud = copy.deepcopy(run_config); loud["name"] = "p_on"; loud["progress"] = True

    a = T.train(quiet, verbose=False)["history"]
    b = T.train(loud, verbose=False)["history"]
    for x, y in zip(a, b):
        assert x["train_loss"] == pytest.approx(y["train_loss"], rel=1e-12)
        assert x["val_loss"] == pytest.approx(y["val_loss"], rel=1e-12)
        assert x["eer"] == pytest.approx(y["eer"], rel=1e-12)


def test_no_progress_flag_reaches_train_from_the_cli():
    args = T.build_parser().parse_args(["--name", "x", "--no-progress"])
    assert args.no_progress is True
    assert T.build_parser().parse_args(["--name", "x"]).no_progress is False


# --- 10. timing -----------------------------------------------------------


def test_timing_json_is_written_with_totals_and_throughput(corpus, run_config):
    run_config["optim"]["epochs"] = 2
    result = T.train(run_config, verbose=False, no_progress=True)
    timing_path = Path(result["out_dir"]) / "timing.json"
    assert timing_path.exists()

    timing = json.loads(timing_path.read_text())
    for key in ("device", "device_name", "total_seconds", "total_human",
                "epochs_run", "train", "val", "overhead_seconds", "per_epoch"):
        assert key in timing, f"missing {key}"

    for section in ("train", "val"):
        for key in ("total_seconds", "mean_epoch_seconds",
                    "clips_per_second", "frames_per_second"):
            value = timing[section][key]
            assert isinstance(value, float) and np.isfinite(value) and value > 0, \
                f"{section}.{key} = {value}"

    assert timing["epochs_run"] == 2
    assert len(timing["per_epoch"]) == 2
    assert timing["total_seconds"] >= timing["train"]["total_seconds"]
    assert timing["total_seconds"] >= (timing["train"]["total_seconds"]
                                       + timing["val"]["total_seconds"] - 1e-6)


def test_history_carries_per_epoch_timing(corpus, run_config):
    run_config["optim"]["epochs"] = 2
    result = T.train(run_config, verbose=False, no_progress=True)
    history = json.loads((Path(result["out_dir"]) / "history.json").read_text())

    assert "device" in history and "device_name" in history
    for record in history["epochs"]:
        for key in ("train_seconds", "val_seconds", "seconds",
                    "train_clips_per_second", "train_frames_per_second",
                    "val_clips_per_second", "val_frames_per_second"):
            assert key in record, f"missing {key}"
            assert np.isfinite(record[key]) and record[key] > 0
        assert record["seconds"] == pytest.approx(
            record["train_seconds"] + record["val_seconds"], rel=1e-9)


def test_format_duration_reads_sensibly():
    assert T.format_duration(0.0) == "0.0s"
    assert T.format_duration(42.5) == "42.5s"
    assert T.format_duration(125) == "2m 05.0s"
    assert T.format_duration(7200) == "2h 00m"


def test_timing_result_is_returned_to_the_caller(corpus, run_config):
    run_config["optim"]["epochs"] = 1
    timing = T.train(run_config, verbose=False, no_progress=True)["timing"]
    assert timing["epochs_run"] == 1
    assert timing["train"]["frames_per_second"] > 0


# --- 11. the shipped config must be valid ---------------------------------


def test_shipped_config_passes_every_validator():
    """Guards against a corrupted or mistyped configs/train.yaml.

    Without this, an invalid field only surfaces when someone starts a run,
    which on a long job means finding out minutes in.
    """
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             "configs" / "train.yaml").read_text())

    assert config["eval"]["select_on"] in T.SELECTION
    assert config["model"]["core"] in ("bigru", "causal_attn")
    assert config["data"]["label"] in ("literal", "bridged")
    assert config["loss"]["weighting"] in ("none", "inverse_freq")
    assert config["device"] in ("auto", "cuda", "mps", "cpu")
    assert config["optim"]["scheduler"] in ("cosine", "plateau", "none")
    assert config["eval"]["min_fa_frames"] >= 1
    assert config["optim"]["batch_size"] >= 1 and config["optim"]["epochs"] >= 1

    # the dataclasses must build from it without a missing or extra key
    data_config, model_config = T.build_configs(config)
    assert data_config.fps == 100
    assert model_config.n_mels == data_config.n_mels
