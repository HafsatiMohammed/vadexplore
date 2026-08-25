"""Robustness matrix: clean-trained against augmented-trained, over test conditions.

Fixed evaluation conditions are built from the clean test split using hard-split
RIRs and the MUSAN test pool, so no room and no noise recording seen in training
appears here. Every system is scored on byte-identical audio for a given
condition, and every threshold still comes from validation.

    python scripts/robustness_eval.py --clean runs/bigru_clean \\
        --augmented runs/bigru_augmented

torch, numpy, matplotlib, and the existing evaluation code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from vadexplore.augment import (
    AugmentConfig,
    Augmenter,
    augmented_audio,
    features_from_audio,
)
from vadexplore.config import DataConfig
from vadexplore.data import load_split, partition_stems
from vadexplore.evaluate import (
    DEFAULT_COLLAR_S,
    Predictions,
    _resolve,
    match_segments,
    score_at_threshold,
    threshold_for_fa_budget,
    threshold_free_metrics,
)
from vadexplore.labels import make_labels, n_frames_for, segments_from_frames
from vadexplore.loader import load_clip
from vadexplore.postprocess import PostprocessConfig, apply_pipeline
from vadexplore.preprocess import highpass
from vadexplore.silero import silero_speech_probs
from vadexplore.train import load_checkpoint, resolve_device

OUT_DIR = "runs/robustness"
FIGURE = "explore_out/figures/robustness_matrix.png"
WEBRTC_MODE = 2
HANGOVER_MS = 100.0

# Test conditions. Reverb uses target rooms only; noise sits in a room half the
# time, matching how the training augmenter places it.
CONDITIONS = {
    "clean":        dict(reverb_prob=0.0, noise_prob=0.0),
    "noise":        dict(reverb_prob=0.0, noise_prob=1.0, noise_rir_prob=0.0,
                         snr_db_range=(0.0, 10.0)),
    "reverb":       dict(reverb_prob=1.0, noise_prob=0.0),
    "noise+reverb": dict(reverb_prob=1.0, noise_prob=1.0, noise_rir_prob=1.0,
                         snr_db_range=(0.0, 10.0)),
}


def build_condition(name: str, stems, directory, convention, config, rir_dir, musan_dir):
    """Deterministic augmented audio and labels for every clip in one condition."""
    settings = dict(CONDITIONS[name])
    augmenter = Augmenter(AugmentConfig(
        enabled=True, rir_dir=rir_dir, musan_dir=musan_dir,
        rir_split="hard", musan_split="test", seed=1234, **settings))

    items = []
    for index, stem in enumerate(stems):
        clip = load_clip(directory / stem)
        labels = make_labels(clip, fps=config.fps,
                             bridge_gap_s=config.bridge_gap_s)[convention]
        if name == "clean":
            from vadexplore.loader import read_audio
            audio = read_audio(clip, target_sr=config.sample_rate)
        else:
            audio, _ = augmented_audio(clip, labels, augmenter, config, index)
        items.append({"stem": stem, "audio": audio, "labels": labels,
                      "n_frames": n_frames_for(clip.duration_s, config.fps)})
    return items


def model_predictions(model, items, config, stats, device) -> Predictions:
    scores, labels, stems = [], [], []
    with torch.no_grad():
        for item in items:
            features = features_from_audio(item["audio"], item["n_frames"], config, stats)
            logits = model(torch.from_numpy(features)[None].to(device))[0]
            n = min(len(logits), len(item["labels"]))
            scores.append(torch.sigmoid(logits[:n]).float().cpu().numpy())
            labels.append(np.asarray(item["labels"][:n], dtype=bool))
            stems.append(item["stem"])
    return Predictions(scores, labels, stems, fps=config.fps)


def silero_predictions(items, config) -> Predictions:
    scores, labels, stems = [], [], []
    for item in items:
        conditioned = highpass(item["audio"], config.sample_rate,
                               config.highpass_hz, config.highpass_order)
        probs = silero_speech_probs(conditioned, sr=config.sample_rate, fps=config.fps,
                                    n_frames=item["n_frames"], apply_highpass=False)
        n = min(len(probs), len(item["labels"]))
        scores.append(probs[:n])
        labels.append(np.asarray(item["labels"][:n], dtype=bool))
        stems.append(item["stem"])
    return Predictions(scores, labels, stems, fps=config.fps)


def webrtc_predictions(items, config, mode: int = WEBRTC_MODE) -> Predictions:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_baselines", Path(__file__).resolve().parent / "eval_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scores, labels, stems = [], [], []
    for item in items:
        conditioned = highpass(item["audio"], config.sample_rate,
                               config.highpass_hz, config.highpass_order)
        decisions = module.webrtc_frames(conditioned, item["n_frames"], mode,
                                         config.sample_rate)
        n = min(len(decisions), len(item["labels"]))
        scores.append(decisions[:n].astype(np.float32))
        labels.append(np.asarray(item["labels"][:n], dtype=bool))
        stems.append(item["stem"])
    return Predictions(scores, labels, stems, fps=config.fps)


def segment_f1(predictions: Predictions, threshold: float, collar_s: float,
               hangover_ms: float = HANGOVER_MS) -> float:
    """Segment F1 with the post-processing the offline sweep selected."""
    config = PostprocessConfig(smooth_ms=0, min_speech_ms=0, hangover_ms=hangover_ms)
    totals = {"matched": 0, "n_reference": 0, "n_hypothesis": 0}
    for probs, labels in zip(predictions.per_clip_scores, predictions.per_clip_labels):
        decisions = apply_pipeline(probs, threshold, config, predictions.fps)
        result = match_segments(segments_from_frames(labels, predictions.fps),
                                segments_from_frames(decisions, predictions.fps),
                                collar_s)
        for key in totals:
            totals[key] += result[key]
    precision = totals["matched"] / max(totals["n_hypothesis"], 1)
    recall = totals["matched"] / max(totals["n_reference"], 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default="runs/bigru_clean")
    parser.add_argument("--augmented", default="runs/bigru_augmented")
    parser.add_argument("--split", default="test")
    parser.add_argument("--collar-s", type=float, default=DEFAULT_COLLAR_S,
                        dest="collar_s")
    parser.add_argument("--target-fa-per-hour", type=float, default=100.0,
                        dest="target_fa")
    parser.add_argument("--rir-dir",
                        default="~/Documents/research_training/kws-augmentation-kit/rirs")
    parser.add_argument("--musan-dir",
                        default="~/Documents/research_training/kws-augmentation-kit/musan")
    parser.add_argument("--limit-clips", type=int, default=None,
                        dest="limit_clips", help="cap clips, for smoke runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--figure", default=FIGURE)
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    models = {}
    for name, run in (("clean-trained", args.clean), ("augmented-trained", args.augmented)):
        model, payload = load_checkpoint(_resolve(run) / "best.pt", device=device)
        models[name] = {"model": model, "payload": payload,
                        "run": str(_resolve(run)),
                        "augmentation": payload.get("augmentation", {"enabled": False})}

    reference = models["clean-trained"]["payload"]
    config = DataConfig(**reference["data_config"])
    convention = reference["label_convention"]
    stats = (np.asarray(reference["feature_stats"]["mean"], dtype=np.float32),
             np.asarray(reference["feature_stats"]["std"], dtype=np.float32))

    split_data = load_split()
    directory = _resolve(split_data["dataset_dir"])
    val_stems = partition_stems(split_data, "val")
    test_stems = partition_stems(split_data, args.split)
    if args.limit_clips:
        val_stems = val_stems[:args.limit_clips]
        test_stems = test_stems[:args.limit_clips]

    print(f"robustness matrix on {args.split} ({len(test_stems)} clips, "
          f"{convention} labels)")
    print(f"  test conditions use hard-split RIRs and the MUSAN test pool, "
          f"disjoint from training")
    print(f"  thresholds come from the CLEAN validation split and are held fixed "
          f"across conditions\n")

    # thresholds: chosen once, on clean validation, per model
    clean_val = build_condition("clean", val_stems, directory, convention, config,
                                args.rir_dir, args.musan_dir)
    thresholds = {}
    for name, entry in models.items():
        predictions = model_predictions(entry["model"], clean_val, config, stats, device)
        thresholds[name] = threshold_for_fa_budget(predictions, args.target_fa)["threshold"]
        print(f"  {name:20s} threshold {thresholds[name]:.4f} (clean val)")

    silero_threshold = threshold_for_fa_budget(
        silero_predictions(clean_val, config), args.target_fa)["threshold"]
    print(f"  {'silero':20s} threshold {silero_threshold:.4f} (clean val)\n")

    results = {
        "split": args.split, "convention": convention,
        "collar_s": args.collar_s, "target_fa_per_hour": args.target_fa,
        "hangover_ms": HANGOVER_MS,
        "thresholds": {**thresholds, "silero": silero_threshold,
                       "chosen_on": "clean validation, held fixed across conditions"},
        "resource_disjointness": {
            "train_rirs": "train split", "test_rirs": "hard split",
            "train_musan": "80 percent filename-hash pool",
            "test_musan": "20 percent filename-hash pool",
            "echo_rirs_used": False,
        },
        "training": {name: entry["augmentation"] for name, entry in models.items()},
        "conditions": {},
    }

    for condition in CONDITIONS:
        print(f"  building {condition} ...", flush=True)
        items = build_condition(condition, test_stems, directory, convention, config,
                                args.rir_dir, args.musan_dir)
        entry = {}
        for name, spec in models.items():
            predictions = model_predictions(spec["model"], items, config, stats, device)
            free = threshold_free_metrics(predictions)
            entry[name] = {
                "eer": free["eer"], "roc_auc": free["roc_auc"], "pr_auc": free["pr_auc"],
                "at_threshold": score_at_threshold(predictions, thresholds[name]),
                "segment_f1": segment_f1(predictions, thresholds[name], args.collar_s),
            }
        silero = silero_predictions(items, config)
        free = threshold_free_metrics(silero)
        entry["silero"] = {
            "eer": free["eer"], "roc_auc": free["roc_auc"], "pr_auc": free["pr_auc"],
            "at_threshold": score_at_threshold(silero, silero_threshold),
            "segment_f1": segment_f1(silero, silero_threshold, args.collar_s),
        }
        webrtc = webrtc_predictions(items, config)
        entry[f"webrtc mode {WEBRTC_MODE}"] = {
            "eer": None, "roc_auc": None, "pr_auc": None,
            "at_threshold": score_at_threshold(webrtc, 0.5),
            "segment_f1": segment_f1(webrtc, 0.5, args.collar_s, hangover_ms=0.0),
        }
        results["conditions"][condition] = entry

    (out_dir / f"matrix_{args.split}.json").write_text(json.dumps(results, indent=2))

    # --- table ---
    systems = ["clean-trained", "augmented-trained", "silero", f"webrtc mode {WEBRTC_MODE}"]
    print(f"\n{'=' * 96}")
    print(f"ROBUSTNESS MATRIX, EER percent on {args.split} "
          f"(lower is better; webrtc is binary so it has no EER)")
    print("=" * 96)
    print(f"  {'condition':<14}" + "".join(f"{s:>20}" for s in systems))
    for condition in CONDITIONS:
        cells = ""
        for system in systems:
            value = results["conditions"][condition][system]["eer"]
            cells += f"{'-':>20}" if value is None else f"{value * 100:20.2f}"
        print(f"  {condition:<14}{cells}")

    print(f"\n  segment F1 (collar {args.collar_s*1000:.0f} ms, "
          f"{HANGOVER_MS:.0f} ms hangover)")
    print(f"  {'condition':<14}" + "".join(f"{s:>20}" for s in systems))
    for condition in CONDITIONS:
        cells = "".join(
            f"{results['conditions'][condition][s]['segment_f1']:20.3f}" for s in systems)
        print(f"  {condition:<14}{cells}")

    print(f"\n  degradation from clean, EER points")
    base = {s: results["conditions"]["clean"][s]["eer"] for s in systems[:3]}
    for condition in CONDITIONS:
        if condition == "clean":
            continue
        cells = "".join(
            f"{(results['conditions'][condition][s]['eer'] - base[s]) * 100:+20.2f}"
            for s in systems[:3])
        print(f"  {condition:<14}{cells}")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    names = list(CONDITIONS)
    x = np.arange(len(names))
    palette = {"clean-trained": "#c53030", "augmented-trained": "#2f855a",
               "silero": "#2b6cb0", f"webrtc mode {WEBRTC_MODE}": "0.55"}

    for ax, key, label, pct in ((axes[0], "eer", "EER (percent)", True),
                                (axes[1], "segment_f1", "segment F1", False)):
        width = 0.2
        for i, system in enumerate(systems):
            values = [results["conditions"][c][system][key] for c in names]
            if any(v is None for v in values):
                continue
            heights = [v * 100 if pct else v for v in values]
            ax.bar(x + (i - 1.5) * width, heights, width, label=system,
                   color=palette[system])
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False)

    axes[0].set_title("Frame-level error against condition", fontsize=10.5, pad=8)
    axes[1].set_title("Segment F1 against condition", fontsize=10.5, pad=8)
    fig.suptitle("Robustness to unseen rooms and unseen noise: clean-trained "
                 "against augmented-trained", fontsize=12)
    fig.tight_layout()
    figure_path = _resolve(args.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  results {out_dir / f'matrix_{args.split}.json'}")
    print(f"  figure  {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
