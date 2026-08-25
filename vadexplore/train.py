"""Train one VAD model. Lean single-run path, no sweeps.

Everything upstream is reused unchanged: the frozen split and feature
statistics from `vadexplore.data`, the model from `vadexplore.model`, and the
fixed frame grid. This module only adds the loop, the loss weighting, and
enough detection metrics to select a checkpoint on something meaningful.

    python vadexplore/train.py --config configs/train.yaml --name bigru_bridged \\
        --core bigru --label bridged

    python vadexplore/train.py --config configs/train.yaml --name attn_bridged \\
        --core causal_attn --label bridged --past-window-frames 100 \\
        --lookahead-frames 5

torch, numpy, pyyaml, and tqdm.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from vadexplore.config import DataConfig, ModelConfig
from vadexplore.data import (
    VADDataset,
    collate,
    load_feature_stats,
    load_split,
    partition_stems,
    training_feature_stats,
)
from vadexplore.labels import make_labels
from vadexplore.loader import load_clip
from vadexplore.model import VADModel

PARTITIONS = ("train", "val", "test")


# --- plumbing -------------------------------------------------------------


def _resolve(path) -> Path:
    return Path(os.path.expanduser(str(path)))


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch for one run.

    This fixes initialization, shuffling, and dropout for a given seed, so two
    runs of the same config agree. It is per-run reproducibility, not a claim
    that results are seed-independent: comparing two cores or two label
    conventions still needs several seeds before a small gap means anything.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve the config `device` field.

    `auto` prefers cuda, then mps, then cpu. Any explicit value is honored and
    fails loudly when that backend is missing, rather than silently dropping to
    cpu and turning a misconfigured GPU run into a very slow one that still
    reports success.
    """
    spec = (spec or "auto").lower()

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available; "
                             "set device: mps or auto")
        return torch.device("cuda")

    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available; "
                             "set device: cuda or auto")
        return torch.device("mps")

    if spec == "cpu":
        return torch.device("cpu")

    raise ValueError(f"device must be cuda, mps, cpu, or auto; got {spec!r}")


def describe_device(device: torch.device) -> str:
    """Human-readable name for the run header."""
    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
        return f"cuda: {name} ({total:.1f} GiB)"
    if device.type == "mps":
        return f"mps: Apple Silicon, {platform.machine()}"
    return f"cpu: {platform.processor() or platform.machine()}"


def synchronize(device: torch.device) -> None:
    """Flush queued work so a wall-clock reading measures finished compute.

    Both cuda and mps dispatch asynchronously, so timing without this measures
    how fast the queue filled rather than how long the epoch took.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 90:
        return f"{int(minutes)}m {rest:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m"


def progress_enabled(setting, no_progress_flag: bool = False) -> bool:
    """Bars on for an interactive terminal, off for a redirected log.

    A headless run on a remote box writes to a file or a job log, and bar
    control characters make those unreadable, so the default follows the TTY
    and the printed epoch summary carries the information either way.
    """
    if no_progress_flag:
        return False
    if setting in (True, False):
        return bool(setting)
    return bool(sys.stderr.isatty())


class CachedDataset(Dataset):
    """Memoizes a dataset's items so features are built once, not once per epoch.

    Wraps rather than modifies `VADDataset`. The whole training partition is
    roughly 130 MB of float32 log-mel, which is worth trading for the repeated
    decode, filter, and spectrogram work.
    """

    def __init__(self, dataset: Dataset, enabled: bool = True):
        self.dataset = dataset
        self.enabled = enabled
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        if not self.enabled:
            return self.dataset[index]
        if index not in self._cache:
            self._cache[index] = self.dataset[index]
        return self._cache[index]


# --- loss weighting -------------------------------------------------------


def training_class_counts(split, convention: str, config: DataConfig) -> tuple[int, int]:
    """(speech frames, non-speech frames) over the TRAINING partition only.

    Reads labels, never audio, and never touches val or test. Class priors are
    a property of the training distribution; taking them from a held-out set
    would leak its balance into the objective.
    """
    split_data = split if isinstance(split, dict) else load_split(split)
    directory = _resolve(split_data["dataset_dir"])

    speech = frames = 0
    for stem in partition_stems(split_data, "train"):
        labels = make_labels(load_clip(directory / stem), fps=config.fps,
                             bridge_gap_s=config.bridge_gap_s)[convention]
        speech += int(labels.sum())
        frames += int(len(labels))
    return speech, frames - speech


def compute_pos_weight(split, convention: str, config: DataConfig) -> float:
    """`pos_weight` for BCEWithLogitsLoss, as non-speech over speech.

    `BCEWithLogitsLoss` multiplies the positive term of the loss by
    `pos_weight`, so the standard inverse-frequency setting is
    negatives / positives. Here the positive class is speech and it is the
    majority at roughly 81 percent of frames, so the weight comes out near
    0.23: it damps the majority class rather than boosting it, which is the
    opposite of the usual voice-activity intuition and is correct for this
    corpus.
    """
    speech, non_speech = training_class_counts(split, convention, config)
    if speech == 0:
        raise ValueError("training partition has no speech frames")
    return non_speech / speech


# --- detection metrics ----------------------------------------------------


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, exact and tie-aware."""
    positive = labels.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties so the estimate is unbiased
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    summed = np.zeros(len(unique))
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Frame-level EER and the threshold that attains it."""
    positive = labels.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), 0.5

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = positive[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(~sorted_labels)

    frr = 1.0 - true_positive / n_pos   # missed speech
    far = false_positive / n_neg        # non-speech called speech
    crossing = int(np.argmin(np.abs(frr - far)))
    return float((frr[crossing] + far[crossing]) / 2.0), float(scores[order][crossing])


def false_alarm_events(prediction: np.ndarray, labels: np.ndarray,
                       clip_start: np.ndarray, min_frames: int = 1) -> int:
    """Count runs of false-positive frames, never merging across clips.

    An operator hears one interruption per burst, not one per frame, so a
    per-hour rate has to count events. `clip_start` marks the first frame of
    each clip in the concatenated array and forces a run boundary there.

    `min_frames` drops runs shorter than that. A one or two frame blip is
    10 to 20 ms of spurious speech, below what a listener notices and below
    what survives any output smoothing, so counting it as a false alarm makes
    the metric measure jitter rather than audible errors. Measured on this
    corpus, roughly half of all false-positive runs at threshold 0.5 are
    3 frames or shorter.
    """
    false_positive = prediction & ~labels
    if not false_positive.any():
        return 0

    boundary = clip_start | ~np.concatenate(([False], false_positive[:-1]))
    starts = np.flatnonzero(false_positive & boundary)
    if min_frames <= 1:
        return int(len(starts))

    ends = np.flatnonzero(
        false_positive & (np.concatenate((clip_start[1:], [True])) |
                          ~np.concatenate((false_positive[1:], [False]))))
    return int(((ends - starts + 1) >= min_frames).sum())


def frr_at_fa_per_hour(
    scores: np.ndarray,
    labels: np.ndarray,
    clip_start: np.ndarray,
    hours: float,
    target_fa_per_hour: float,
    n_thresholds: int = 101,
    min_fa_frames: int = 3,
) -> dict:
    """Miss rate at the strictest operating point meeting the false-alarm budget.

    The false-alarm rate is scanned on a threshold grid rather than searched,
    because event counts are not monotonic in the threshold: raising it can
    split one long false alarm into two shorter ones and increase the count.

    This metric needs a validation set large enough to resolve the budget. At
    10 false alarms per hour a half-hour validation split permits about five
    events in total, so the measurement is coarse and jumpy; `budget_met`
    records whether any threshold satisfied it at all.
    """
    labels = labels.astype(bool)
    grid = np.quantile(scores, np.linspace(0.0, 1.0, int(n_thresholds)))
    grid = np.unique(np.concatenate([grid, [0.5]]))

    best = None
    for threshold in grid:
        prediction = scores >= threshold
        events = false_alarm_events(prediction, labels, clip_start, min_fa_frames)
        fa_rate = events / hours if hours > 0 else float("inf")
        if fa_rate <= target_fa_per_hour:
            misses = int((labels & ~prediction).sum())
            frr = misses / max(int(labels.sum()), 1)
            if best is None or frr < best["frr"]:
                best = {"frr": float(frr), "threshold": float(threshold),
                        "fa_per_hour": float(fa_rate)}

    if best is None:
        # the budget is unreachable even at the most conservative threshold
        return {"frr": 1.0, "threshold": float(grid[-1]),
                "fa_per_hour": float("nan"), "budget_met": False}
    best["budget_met"] = True
    return best


# --- evaluation -----------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, criterion, device, settings: dict,
             show_progress: bool = False, epoch_label: str = "") -> dict:
    """Validation loss plus the detection metrics used for selection."""
    model.eval()
    total_loss, total_frames = 0.0, 0
    scores, labels, starts = [], [], []

    bar = tqdm(loader, desc=f"{epoch_label} val".strip(), leave=False,
               unit="batch", disable=not show_progress)
    for batch in bar:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        target = batch["labels"].to(device)

        logits = model(features, mask, lengths)
        valid = mask.reshape(-1)
        flat_logits = logits.reshape(-1)[valid]
        flat_target = target.reshape(-1)[valid].to(flat_logits.dtype)

        loss = criterion(flat_logits, flat_target)
        total_loss += float(loss.detach()) * int(valid.sum())
        total_frames += int(valid.sum())
        bar.set_postfix(loss=f"{total_loss / max(total_frames, 1):.4f}", refresh=False)

        for i, length in enumerate(batch["lengths"].tolist()):
            scores.append(torch.sigmoid(logits[i, :length]).float().cpu().numpy())
            labels.append(target[i, :length].cpu().numpy().astype(bool))
            flag = np.zeros(length, dtype=bool)
            flag[0] = True
            starts.append(flag)

    bar.close()
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    starts = np.concatenate(starts)
    hours = len(scores) / settings["fps"] / 3600.0

    eer, eer_threshold = equal_error_rate(scores, labels)
    operating = frr_at_fa_per_hour(scores, labels, starts, hours,
                                   settings["target_fa_per_hour"],
                                   settings["n_thresholds"],
                                   settings["min_fa_frames"])
    return {
        "val_loss": total_loss / max(total_frames, 1),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": roc_auc(scores, labels),
        "frr_at_fa": operating["frr"],
        "frr_threshold": operating["threshold"],
        "fa_per_hour": operating["fa_per_hour"],
        "fa_budget_met": operating["budget_met"],
        "n_frames": int(len(scores)),
        "hours": hours,
    }


SELECTION = {  # metric name -> True when lower is better
    "frr_at_fa": True,
    "eer": True,
    "val_loss": True,
    "auc": False,
}


# --- configuration --------------------------------------------------------


def load_config(path) -> dict:
    return yaml.safe_load(_resolve(path).read_text())


def apply_overrides(config: dict, args) -> dict:
    """Command-line flags win over the file."""
    config = copy.deepcopy(config)
    mapping = {
        "name": ("name",), "seed": ("seed",), "device": ("device",),
        "core": ("model", "core"), "label": ("data", "label"),
        "past_window_frames": ("model", "past_window_frames"),
        "lookahead_frames": ("model", "lookahead_frames"),
        "batch_size": ("optim", "batch_size"), "epochs": ("optim", "epochs"),
        "lr": ("optim", "lr"), "weight_decay": ("optim", "weight_decay"),
        "loss_weighting": ("loss", "weighting"),
        "limit_clips": ("data", "limit_clips"),
        "augment_enabled": ("augment", "enabled"),
        "highpass_hz": ("data", "highpass_hz"),
    }
    for flag, path in mapping.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        node = config
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return config


def build_configs(config: dict) -> tuple[DataConfig, ModelConfig]:
    data, model = config["data"], config["model"]
    data_config = DataConfig(
        bridge_gap_s=data["bridge_gap_s"],
        highpass_hz=data["highpass_hz"],
        highpass_order=data["highpass_order"],
    )
    model_config = ModelConfig(
        n_mels=data_config.n_mels,
        conv_channels=tuple(model["conv_channels"]),
        d_model=model["d_model"],
        causal_frontend=model["causal_frontend"],
        temporal=model["core"],
        gru_hidden=model["gru_hidden"],
        gru_layers=model["gru_layers"],
        causal=model["causal"],
        attn_layers=model["attn_layers"],
        attn_heads=model["attn_heads"],
        attn_ff_ratio=model["attn_ff_ratio"],
        attn_dropout=model["attn_dropout"],
        lookahead_frames=model["lookahead_frames"],
        past_window_frames=model["past_window_frames"],
        dropout=model["dropout"],
    )
    return data_config, model_config


# --- training -------------------------------------------------------------


def train(config: dict, verbose: bool = True, no_progress: bool = False) -> dict:
    if not config.get("name"):
        raise ValueError("a run name is required, pass --name or set it in the config")

    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    data_config, model_config = build_configs(config)

    out_dir = _resolve(config["out_root"]) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(config["data"]["split"])
    convention = config["data"]["label"]

    stats_path = _resolve(config["data"]["feature_stats"])
    if stats_path.exists():
        stats = load_feature_stats(stats_path)
    else:
        stats = training_feature_stats(split, data_config, save_to=stats_path)

    limit = config["data"].get("limit_clips")

    augment_settings = config.get("augment", {}) or {}
    augmenter = None
    if augment_settings.get("enabled"):
        from vadexplore.augment import AugmentConfig, Augmenter
        augment_config = AugmentConfig(**{k: (tuple(v) if k == "snr_db_range" else v)
                                          for k, v in augment_settings.items()})
        augmenter = Augmenter(augment_config)

    datasets, loaders = {}, {}
    for partition in ("train", "val"):
        dataset = VADDataset(partition, convention, split, data_config, stats=stats)
        if limit:
            dataset.stems = dataset.stems[:limit]
        if augmenter is not None and partition == "train":
            # Augmentation is training only. Validation stays clean so the
            # selection metric measures the model, not the draw of rooms and
            # noise it happened to get that epoch. Caching is disabled here on
            # purpose: the point is a fresh room and a fresh interferer every
            # epoch, and a cache would freeze the first draw forever.
            from vadexplore.augment import AugmentedDataset
            datasets[partition] = AugmentedDataset(dataset, augmenter, data_config)
        else:
            datasets[partition] = CachedDataset(dataset, config["data"]["cache_features"])
        loaders[partition] = DataLoader(
            datasets[partition],
            batch_size=config["optim"]["batch_size"],
            shuffle=(partition == "train"),
            collate_fn=collate,
            generator=torch.Generator().manual_seed(int(config["seed"])),
        )

    # class prior from the training partition only
    if config["loss"]["weighting"] == "inverse_freq":
        pos_weight_value = compute_pos_weight(split, convention, data_config)
    elif config["loss"]["weighting"] == "none":
        pos_weight_value = 1.0
    else:
        raise ValueError(f"loss.weighting must be none or inverse_freq, "
                         f"got {config['loss']['weighting']!r}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device))

    model = VADModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optim"]["lr"],
                                  weight_decay=config["optim"]["weight_decay"])
    epochs = int(config["optim"]["epochs"])
    if config["optim"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config["optim"]["min_lr"])
    elif config["optim"]["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    else:
        scheduler = None

    select_on = config["eval"]["select_on"]
    if select_on not in SELECTION:
        raise ValueError(f"eval.select_on must be one of {sorted(SELECTION)}")
    lower_is_better = SELECTION[select_on]
    eval_settings = {"fps": data_config.fps,
                     "target_fa_per_hour": config["eval"]["target_fa_per_hour"],
                     "n_thresholds": config["eval"]["n_thresholds"],
                     "min_fa_frames": config["eval"]["min_fa_frames"]}

    resolved = copy.deepcopy(config)
    resolved["_resolved"] = {
        "device": str(device),
        "device_name": describe_device(device),
        "pos_weight": pos_weight_value,
        "selection_metric": select_on,
        "selection_direction": "lower is better" if lower_is_better else "higher is better",
        "n_train_clips": len(datasets["train"]),
        "n_val_clips": len(datasets["val"]),
        "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "augmentation": (augmenter.config.to_dict() if augmenter is not None
                         else {"enabled": False}),
    }
    (out_dir / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    show_progress = verbose and progress_enabled(config.get("progress", "auto"), no_progress)

    if verbose:
        print(f"run {config['name']}  core {model_config.temporal}  label {convention}")
        print(f"  device {describe_device(device)}")
        print(f"  {resolved['_resolved']['n_parameters']:,} parameters, "
              f"{len(datasets['train'])} train clips, {len(datasets['val'])} val clips")
        print(f"  pos_weight {pos_weight_value:.4f} "
              f"({config['loss']['weighting']}), selecting on {select_on} "
              f"({'lower' if lower_is_better else 'higher'} is better)")

    history, best, best_epoch, since_improved = [], None, -1, 0
    patience = config["eval"].get("early_stopping_patience")
    best_path = out_dir / "best.pt"

    run_started = time.time()
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="epoch",
                     disable=not show_progress)

    for epoch in epoch_bar:
        if augmenter is not None:
            datasets["train"].set_epoch(epoch)
        model.train()
        synchronize(device)
        train_started = time.time()
        running, seen = 0.0, 0

        batch_bar = tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs} train",
                         leave=False, unit="batch", disable=not show_progress)
        for batch in batch_bar:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            target = batch["labels"].to(device)

            logits = model(features, mask, lengths)
            valid = mask.reshape(-1)
            loss = criterion(logits.reshape(-1)[valid],
                             target.reshape(-1)[valid].to(logits.dtype))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["optim"]["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), config["optim"]["grad_clip"])
            optimizer.step()

            running += float(loss.detach()) * int(valid.sum())
            seen += int(valid.sum())
            batch_bar.set_postfix(loss=f"{running / max(seen, 1):.4f}", refresh=False)

        batch_bar.close()
        synchronize(device)
        train_seconds = time.time() - train_started

        val_started = time.time()
        metrics = evaluate(model, loaders["val"], criterion, device, eval_settings,
                           show_progress=show_progress,
                           epoch_label=f"epoch {epoch}/{epochs}")
        synchronize(device)
        val_seconds = time.time() - val_started

        train_loss = running / max(seen, 1)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics[select_on])
        elif scheduler is not None:
            scheduler.step()

        timing = {
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "seconds": train_seconds + val_seconds,
            "train_clips_per_second": len(datasets["train"]) / max(train_seconds, 1e-9),
            "train_frames_per_second": seen / max(train_seconds, 1e-9),
            "val_clips_per_second": len(datasets["val"]) / max(val_seconds, 1e-9),
            "val_frames_per_second": metrics["n_frames"] / max(val_seconds, 1e-9),
        }
        record = {"epoch": epoch, "train_loss": train_loss,
                  "lr": optimizer.param_groups[0]["lr"], **timing, **metrics}
        history.append(record)

        score = metrics[select_on]
        improved = (best is None
                    or (score < best if lower_is_better else score > best))
        if improved:
            best, best_epoch, since_improved = score, epoch, 0
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config.to_dict(),
                "data_config": data_config.to_dict(),
                "label_convention": convention,
                "feature_stats_path": str(stats_path),
                "feature_stats": {"mean": stats[0].tolist(), "std": stats[1].tolist()},
                "epoch": epoch,
                "selection_metric": select_on,
                "selection_value": float(score),
                "metrics": metrics,
                "pos_weight": pos_weight_value,
                "seed": int(config["seed"]),
                "run_name": config["name"],
                "augmentation": (augmenter.config.to_dict() if augmenter is not None
                                 else {"enabled": False}),
            }, best_path)
        else:
            since_improved += 1

        if verbose and epoch == 1:
            per_epoch = record["seconds"]
            print(f"  first epoch took {format_duration(per_epoch)} "
                  f"(train {format_duration(train_seconds)}, "
                  f"val {format_duration(val_seconds)}); "
                  f"estimated total for {epochs} epochs "
                  f"{format_duration(per_epoch * epochs)}")

        if verbose and epoch == 1 and metrics["frr_at_fa"] > 0.5:
            print(f"  note: the {config['eval']['target_fa_per_hour']:.0f} fa/h budget "
                  f"forces frr above 50 percent on {metrics['hours']:.2f} h of validation "
                  f"({metrics['hours'] * config['eval']['target_fa_per_hour']:.1f} events "
                  f"allowed in total). Selection will be coarse; treat eer and auc as "
                  f"the readable numbers and set the target from a product requirement.")
        if verbose:
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {metrics['val_loss']:.4f}  "
                  f"eer {metrics['eer']*100:5.2f}%  auc {metrics['auc']:.4f}  "
                  f"frr@fa {metrics['frr_at_fa']*100:5.2f}%  "
                  f"({metrics['fa_per_hour']:.1f} fa/h)  "
                  f"{train_seconds:.1f}s+{val_seconds:.1f}s"
                  f"{'  *' if improved else ''}")
        epoch_bar.set_postfix(eer=f"{metrics['eer']*100:.2f}%",
                              best=f"{best:.4f}", refresh=False)

        (out_dir / "history.json").write_text(json.dumps({
            "run": config["name"],
            "selection_metric": select_on,
            "best_epoch": best_epoch,
            "best_value": best,
            "pos_weight": pos_weight_value,
            "device": str(device),
            "device_name": describe_device(device),
            "epochs": history,
        }, indent=2))

        epoch_bar.set_description(f"epochs (best {select_on} {best:.4f})")

        if patience and since_improved >= patience:
            if verbose:
                print(f"  early stop: no improvement in {select_on} for {patience} epochs")
            break
    epoch_bar.close()

    total_seconds = time.time() - run_started
    train_total = sum(r["train_seconds"] for r in history)
    val_total = sum(r["val_seconds"] for r in history)
    train_frames = sum(r["train_frames_per_second"] * r["train_seconds"] for r in history)
    val_frames = sum(r["val_frames_per_second"] * r["val_seconds"] for r in history)

    timing_summary = {
        "run": config["name"],
        "device": str(device),
        "device_name": describe_device(device),
        "core": model_config.temporal,
        "batch_size": config["optim"]["batch_size"],
        "epochs_run": len(history),
        "epochs_configured": epochs,
        "total_seconds": total_seconds,
        "total_human": format_duration(total_seconds),
        "train": {
            "total_seconds": train_total,
            "mean_epoch_seconds": train_total / max(len(history), 1),
            "clips_per_second": len(datasets["train"]) * len(history) / max(train_total, 1e-9),
            "frames_per_second": train_frames / max(train_total, 1e-9),
        },
        "val": {
            "total_seconds": val_total,
            "mean_epoch_seconds": val_total / max(len(history), 1),
            "clips_per_second": len(datasets["val"]) * len(history) / max(val_total, 1e-9),
            "frames_per_second": val_frames / max(val_total, 1e-9),
        },
        "overhead_seconds": total_seconds - train_total - val_total,
        "per_epoch": [
            {k: r[k] for k in ("epoch", "train_seconds", "val_seconds", "seconds",
                               "train_clips_per_second", "train_frames_per_second",
                               "val_clips_per_second", "val_frames_per_second")}
            for r in history
        ],
    }
    (out_dir / "timing.json").write_text(json.dumps(timing_summary, indent=2))

    if verbose:
        print(f"  best {select_on} {best:.4f} at epoch {best_epoch}")
        print(f"  checkpoint {best_path}")
        print(f"\n  timing on {describe_device(device)}")
        print(f"    total          {format_duration(total_seconds)} "
              f"over {len(history)} epochs")
        print(f"    train          {format_duration(train_total)} "
              f"({format_duration(timing_summary['train']['mean_epoch_seconds'])} per epoch, "
              f"{timing_summary['train']['clips_per_second']:.1f} clips/s, "
              f"{timing_summary['train']['frames_per_second']:,.0f} frames/s)")
        print(f"    validation     {format_duration(val_total)} "
              f"({format_duration(timing_summary['val']['mean_epoch_seconds'])} per epoch, "
              f"{timing_summary['val']['clips_per_second']:.1f} clips/s, "
              f"{timing_summary['val']['frames_per_second']:,.0f} frames/s)")
        print(f"    other          {format_duration(timing_summary['overhead_seconds'])} "
              f"(setup, feature caching, checkpoint writes)")
        print(f"    timing written to {out_dir / 'timing.json'}")

    return {"out_dir": out_dir, "best_path": best_path, "history": history,
            "best_epoch": best_epoch, "best_value": best,
            "pos_weight": pos_weight_value, "timing": timing_summary}


def load_checkpoint(path, device=None) -> tuple[VADModel, dict]:
    """Rebuild a model from a checkpoint without needing the original config."""
    payload = torch.load(_resolve(path), map_location="cpu", weights_only=False)
    model = VADModel(ModelConfig(**{**payload["model_config"],
                                    "conv_channels": tuple(payload["model_config"]["conv_channels"])}))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model, payload


# --- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--name")
    parser.add_argument("--core", choices=["bigru", "causal_attn"])
    parser.add_argument("--label", choices=["literal", "bridged"])
    parser.add_argument("--past-window-frames", type=int, dest="past_window_frames")
    parser.add_argument("--lookahead-frames", type=int, dest="lookahead_frames")
    parser.add_argument("--loss-weighting", choices=["none", "inverse_freq"],
                        dest="loss_weighting")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--limit-clips", type=int, dest="limit_clips")
    parser.add_argument("--highpass-hz", type=float, dest="highpass_hz")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--augment", dest="augment_enabled", action="store_true",
                        default=None, help="enable training-time augmentation")
    parser.add_argument("--no-progress", action="store_true", dest="no_progress",
                        help="suppress tqdm bars, for redirected logs and headless runs")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    try:
        train(config, no_progress=args.no_progress)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
