"""Evaluate a trained checkpoint, and score any other system the same way.

Thresholds are chosen on validation and applied unchanged to test. EER and the
areas under the curves need no threshold and are computed on test directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from vadexplore.config import DataConfig
from vadexplore.data import VADDataset, collate, load_split
from vadexplore.labels import segments_from_frames
from vadexplore.train import (
    equal_error_rate,
    false_alarm_events,
    load_checkpoint,
    resolve_device,
    roc_auc,
)

DEFAULT_FA_TARGETS = (10.0, 100.0)
DEFAULT_COLLAR_S = 0.050
DEFAULT_MIN_FA_FRAMES = 3
N_CURVE_POINTS = 200
# Boundary errors are float subtractions, and abs(0.15 - 0.10) is
# 0.05000000000000000277, so a bare <= on a round collar rejects an offset
# the caller meant to accept. Same guard as the label geometry uses.
_EPS = 1e-12
CONVENTIONS = ("literal", "bridged")


def _resolve(path) -> Path:
    return Path(os.path.expanduser(str(path)))


class Predictions:
    """Per-frame scores and labels for one split, kept clip by clip.

    Clip boundaries matter: false-alarm events must not merge across clips.
    """

    def __init__(self, scores: list[np.ndarray], labels: list[np.ndarray],
                 stems: list[str], fps: int = 100):
        self.per_clip_scores = scores
        self.per_clip_labels = labels
        self.stems = stems
        self.fps = fps

        self.scores = np.concatenate(scores) if scores else np.zeros(0)
        self.labels = np.concatenate(labels).astype(bool) if labels else np.zeros(0, bool)
        starts = []
        for array in scores:
            flag = np.zeros(len(array), dtype=bool)
            if len(flag):
                flag[0] = True
            starts.append(flag)
        self.clip_start = np.concatenate(starts) if starts else np.zeros(0, bool)

    @property
    def n_frames(self) -> int:
        return int(len(self.scores))

    @property
    def hours(self) -> float:
        return self.n_frames / self.fps / 3600.0


@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size: int = 8) -> Predictions:
    """Run a model over a dataset and gather masked per-frame probabilities."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval().to(device)

    scores, labels, stems = [], [], []
    for batch in loader:
        logits = model(batch["features"].to(device), batch["mask"].to(device),
                       batch["lengths"].to(device))
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
        for i, length in enumerate(batch["lengths"].tolist()):
            scores.append(probabilities[i, :length])
            labels.append(batch["labels"][i, :length].numpy().astype(bool))
            stems.append(batch["stems"][i])
    return Predictions(scores, labels, stems)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise definition."""
    positive = labels.astype(bool)
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = positive[order]
    true_positive = np.cumsum(ranked)
    precision = true_positive / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positive.sum())


def curve_points(predictions: Predictions, n_points: int = N_CURVE_POINTS,
                 min_fa_frames: int = DEFAULT_MIN_FA_FRAMES) -> dict:
    """ROC and DET points on a threshold grid.

    The DET axis is false alarms per hour, the unit a deployment budget uses.
    """
    scores, labels = predictions.scores, predictions.labels
    grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_points)))

    rows = []
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((~labels).sum()), 1)
    for threshold in grid:
        prediction = scores >= threshold
        misses = int((labels & ~prediction).sum())
        false_positive_frames = int((~labels & prediction).sum())
        events = false_alarm_events(prediction, labels, predictions.clip_start,
                                    min_fa_frames)
        rows.append({
            "threshold": float(threshold),
            "frr": misses / n_pos,
            "far": false_positive_frames / n_neg,
            "tpr": 1.0 - misses / n_pos,
            "fa_per_hour": events / predictions.hours if predictions.hours else float("nan"),
        })
    return {"points": rows, "n_thresholds": len(grid)}


def threshold_free_metrics(predictions: Predictions) -> dict:
    eer, eer_threshold = equal_error_rate(predictions.scores, predictions.labels)
    return {
        "roc_auc": roc_auc(predictions.scores, predictions.labels),
        "pr_auc": average_precision(predictions.scores, predictions.labels),
        "eer": eer,
        "eer_threshold_on_this_split": eer_threshold,
        "n_frames": predictions.n_frames,
        "hours": predictions.hours,
        "speech_fraction": float(predictions.labels.mean()) if predictions.n_frames else None,
    }


def threshold_for_fa_budget(predictions: Predictions, target_fa_per_hour: float,
                            n_points: int = N_CURVE_POINTS,
                            min_fa_frames: int = DEFAULT_MIN_FA_FRAMES) -> dict:
    """Threshold meeting a false-alarm budget at the lowest miss rate.

    Call it on validation; the result is frozen and handed to score_at_threshold.
    """
    best = None
    for row in curve_points(predictions, n_points, min_fa_frames)["points"]:
        if row["fa_per_hour"] <= target_fa_per_hour:
            if best is None or row["frr"] < best["frr"]:
                best = row
    if best is None:
        return {"budget_met": False, "threshold": 1.0,
                "target_fa_per_hour": target_fa_per_hour}
    return {"budget_met": True, "threshold": best["threshold"],
            "target_fa_per_hour": target_fa_per_hour,
            "frr_on_selection_split": best["frr"],
            "fa_per_hour_on_selection_split": best["fa_per_hour"]}


def score_at_threshold(predictions: Predictions, threshold: float,
                       min_fa_frames: int = DEFAULT_MIN_FA_FRAMES) -> dict:
    """Apply a fixed threshold. No search, so this cannot tune on the split."""
    prediction = predictions.scores >= threshold
    labels = predictions.labels
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((~labels).sum()), 1)

    true_positive = int((labels & prediction).sum())
    false_positive = int((~labels & prediction).sum())
    false_negative = int((labels & ~prediction).sum())
    events = false_alarm_events(prediction, labels, predictions.clip_start, min_fa_frames)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "threshold": float(threshold),
        "frr": false_negative / n_pos,
        "far": false_positive / n_neg,
        "accuracy": float((prediction == labels).mean()) if len(labels) else None,
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "fa_events": int(events),
        "fa_per_hour": events / predictions.hours if predictions.hours else float("nan"),
    }


def match_segments(reference: list, hypothesis: list, collar_s: float) -> dict:
    """One-to-one segment matching with a boundary collar.

    Both start and end must land within collar_s. Greedy on the smallest total
    boundary error, each segment used at most once.
    """
    pairs = []
    for i, (ref_start, ref_end) in enumerate(reference):
        for j, (hyp_start, hyp_end) in enumerate(hypothesis):
            start_error = abs(hyp_start - ref_start)
            end_error = abs(hyp_end - ref_end)
            if start_error <= collar_s + _EPS and end_error <= collar_s + _EPS:
                pairs.append((start_error + end_error, i, j))

    pairs.sort()
    used_reference, used_hypothesis, matched = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_reference and j not in used_hypothesis:
            used_reference.add(i)
            used_hypothesis.add(j)
            matched += 1

    precision = matched / len(hypothesis) if hypothesis else 0.0
    recall = matched / len(reference) if reference else 0.0
    return {
        "matched": matched,
        "n_reference": len(reference),
        "n_hypothesis": len(hypothesis),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def segment_metrics(predictions: Predictions, threshold: float,
                    collar_s: float = DEFAULT_COLLAR_S) -> dict:
    """Segment precision, recall, and F1 over the split, pooled across clips."""
    totals = {"matched": 0, "n_reference": 0, "n_hypothesis": 0}
    for scores, labels in zip(predictions.per_clip_scores, predictions.per_clip_labels):
        hypothesis = segments_from_frames(scores >= threshold, predictions.fps)
        reference = segments_from_frames(labels, predictions.fps)
        result = match_segments(reference, hypothesis, collar_s)
        for key in totals:
            totals[key] += result[key]

    precision = totals["matched"] / max(totals["n_hypothesis"], 1)
    recall = totals["matched"] / max(totals["n_reference"], 1)
    return {
        **totals,
        "collar_s": collar_s,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_det(curves: dict, operating_points: list, path: Path, title: str) -> None:
    """FRR against false alarms per hour, log axes, operating points marked."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, (points, color, style) in curves.items():
        fa = np.array([p["fa_per_hour"] for p in points])
        frr = np.array([p["frr"] for p in points])
        keep = (fa > 0) & (frr > 0)
        ax.plot(fa[keep], frr[keep] * 100, style, color=color, linewidth=1.7, label=name)

    for point in operating_points:
        ax.plot(point["fa_per_hour"], point["frr"] * 100, point.get("marker", "o"),
                color=point.get("color", "#c53030"), markersize=8,
                markeredgecolor="white", markeredgewidth=1.0, label=point["label"], zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, title, "false alarms per hour (log)", "false rejection rate, percent (log)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_roc(curves: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for name, (points, color, style) in curves.items():
        far = np.array([p["far"] for p in points])
        tpr = np.array([p["tpr"] for p in points])
        order = np.argsort(far)
        ax.plot(far[order], tpr[order], style, color=color, linewidth=1.8, label=name)
    ax.plot([0, 1], [0, 1], ":", color="0.6", linewidth=1.0, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _style(ax, title, "false alarm rate (non-speech frames called speech)",
           "true positive rate (speech frames found)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def evaluate_run(
    run_dir,
    split: str = "test",
    fa_targets=DEFAULT_FA_TARGETS,
    collar_s: float = DEFAULT_COLLAR_S,
    device=None,
    batch_size: int = 8,
    split_file=None,
    write: bool = True,
    convention: str | None = None,
) -> dict:
    """Evaluate one checkpoint, every threshold chosen on validation.

    Both conventions are always scored; convention only picks which is primary.
    """
    run_dir = _resolve(run_dir)
    device = resolve_device(device) if isinstance(device, str) or device is None else device

    model, payload = load_checkpoint(run_dir / "best.pt", device=device)
    data_config = DataConfig(**payload["data_config"])
    trained_on = payload["label_convention"]
    primary_convention = convention or trained_on
    if primary_convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, "
                         f"got {primary_convention!r}")
    stats = (np.asarray(payload["feature_stats"]["mean"], dtype=np.float32),
             np.asarray(payload["feature_stats"]["std"], dtype=np.float32))

    split_data = load_split(split_file) if split_file else load_split()
    results = {
        "run": str(run_dir),
        "run_name": payload.get("run_name"),
        "split": split,
        "device": str(device),
        "trained_on_convention": trained_on,
        "primary_convention": primary_convention,
        "primary_is_trained_on": primary_convention == trained_on,
        "checkpoint": {"epoch": payload["epoch"],
                       "selection_metric": payload["selection_metric"],
                       "selection_value": payload["selection_value"]},
        "discipline": (
            "Every threshold is chosen on the validation split and applied unchanged "
            "to the evaluation split. EER and the areas under the curves need no "
            "threshold and are computed on the evaluation split directly. No number "
            "here was tuned on test."
        ),
        # Evaluating the val split against itself is a legitimate thing to do
        # while developing, but the operating point is then selected on the
        # same data it is scored on and reads optimistically. Say so in the
        # output rather than letting the discipline note above imply otherwise.
        "threshold_split_equals_eval_split": split == "val",
        "conventions": {},
    }

    for name in CONVENTIONS:
        validation = collect_predictions(
            model, VADDataset("val", name, split_data, data_config, stats=stats),
            device, batch_size)
        evaluation = collect_predictions(
            model, VADDataset(split, name, split_data, data_config, stats=stats),
            device, batch_size)

        entry = {
            "role": "primary" if name == primary_convention else "cross-reference",
            "threshold_free_on_" + split: threshold_free_metrics(evaluation),
            "threshold_free_on_val": threshold_free_metrics(validation),
            "operating_points": [],
        }

        eval_curve = curve_points(evaluation)
        entry["curve"] = eval_curve

        for target in fa_targets:
            chosen = threshold_for_fa_budget(validation, target)
            applied = score_at_threshold(evaluation, chosen["threshold"])
            segments = segment_metrics(evaluation, chosen["threshold"], collar_s)
            entry["operating_points"].append({
                "target_fa_per_hour": target,
                "target_is_placeholder": True,
                "threshold": chosen["threshold"],
                "threshold_chosen_on": "val",
                "threshold_applied_to": split,
                "budget_met_on_val": chosen["budget_met"],
                "val_frr": chosen.get("frr_on_selection_split"),
                "val_fa_per_hour": chosen.get("fa_per_hour_on_selection_split"),
                f"{split}_frame": applied,
                f"{split}_segment": segments,
                "resolution_caveat": (
                    f"the {split} split is {evaluation.hours:.2f} h, so a budget of "
                    f"{target:g} per hour allows about {target * evaluation.hours:.1f} "
                    f"events in total and the measurement is coarse"
                ),
            })

        results["conventions"][name] = entry

    if write:
        primary = results["conventions"][primary_convention]
        plot_det({f"model, {primary_convention} labels": (primary["curve"]["points"], "#2b6cb0", "-")},
                 [{"label": f"FA budget {p['target_fa_per_hour']:g}/h "
                            f"(threshold from val)",
                   "fa_per_hour": p[f"{split}_frame"]["fa_per_hour"],
                   "frr": p[f"{split}_frame"]["frr"],
                   "color": ["#c53030", "#2f855a"][i % 2],
                   "marker": ["o", "s"][i % 2]}
                  for i, p in enumerate(primary["operating_points"])
                  if p[f"{split}_frame"]["fa_per_hour"] > 0],
                 run_dir / f"det_{split}.png",
                 f"DET on {split}: {payload.get('run_name')} ({primary_convention} labels)")
        plot_roc({f"model, {primary_convention} labels": (primary["curve"]["points"], "#2b6cb0", "-")},
                 run_dir / f"roc_{split}.png",
                 f"ROC on {split}: {payload.get('run_name')} ({primary_convention} labels)")
        results["figures"] = {"det": str(run_dir / f"det_{split}.png"),
                              "roc": str(run_dir / f"roc_{split}.png")}
        (run_dir / f"eval_{split}.json").write_text(json.dumps(results, indent=2))

    return results


def summarize(results: dict) -> None:
    """Print the primary convention's headline numbers."""
    split = results["split"]
    trained_on = results["trained_on_convention"]
    primary = results.get("primary_convention", trained_on)
    header = f"\n{results['run_name']} on {split}  (trained on {trained_on} labels"
    if primary != trained_on:
        header += f", reported primary against {primary}"
    print(header + ")")
    if results.get("threshold_split_equals_eval_split"):
        print("  WARNING: thresholds come from val and are scored on val")
    for convention, entry in results["conventions"].items():
        free = entry[f"threshold_free_on_{split}"]
        print(f"\n  {convention} labels ({entry['role']})")
        print(f"    EER {free['eer']*100:6.2f}%   ROC-AUC {free['roc_auc']:.4f}   "
              f"PR-AUC {free['pr_auc']:.4f}   ({free['hours']:.2f} h, "
              f"{free['speech_fraction']*100:.1f}% speech)")
        for point in entry["operating_points"]:
            frame = point[f"{split}_frame"]
            segment = point[f"{split}_segment"]
            print(f"    FA target {point['target_fa_per_hour']:>5g}/h  "
                  f"threshold {point['threshold']:.4f} (from val)  ->  "
                  f"{split} FRR {frame['frr']*100:5.2f}%  "
                  f"realized {frame['fa_per_hour']:6.1f} fa/h  "
                  f"segment F1 {segment['f1']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VAD checkpoint on one split. Every "
                    "threshold is chosen on validation and applied unchanged.")
    parser.add_argument("--run", required=True,
                        help="run directory containing best.pt")
    parser.add_argument("--split", default="test",
                        help="split to evaluate, default test")
    parser.add_argument("--convention", default=None, choices=list(CONVENTIONS),
                        help="which convention to report as primary; "
                             "defaults to the one the checkpoint was trained on")
    parser.add_argument("--device", default=None,
                        help="cuda, mps, cpu, or auto; defaults to auto")
    parser.add_argument("--collar-s", type=float, default=DEFAULT_COLLAR_S,
                        dest="collar_s")
    parser.add_argument("--fa-targets", type=float, nargs="+",
                        default=list(DEFAULT_FA_TARGETS), dest="fa_targets")
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = _resolve(args.run)
    if not (run_dir / "best.pt").exists():
        print(f"error: no checkpoint at {run_dir / 'best.pt'}\n"
              f"  train one first: python vadexplore/train.py --name {run_dir.name}",
              file=sys.stderr)
        return 2

    try:
        results = evaluate_run(
            run_dir,
            split=args.split,
            fa_targets=tuple(args.fa_targets),
            collar_s=args.collar_s,
            device=args.device,
            batch_size=args.batch_size,
            convention=args.convention,
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summarize(results)
    print(f"\n  wrote {run_dir / f'eval_{args.split}.json'}")
    for path in results.get("figures", {}).values():
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
