"""Sweep VAD post-processing on a trained model's posteriors.

No retraining. Posteriors come from the existing evaluation code, and the
threshold stays at the value validation chose for the false-alarm budget, held
fixed throughout. Every gain reported here is therefore attributable to the
post-processing and not to moving the operating point underneath it.

Settings are selected on validation segment F1 and reported on test.

    python scripts/postproc_sweep.py --run runs/<name>

numpy, plus the existing evaluation and plotting code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from vadexplore.config import DataConfig
from vadexplore.data import VADDataset, load_split
from vadexplore.evaluate import (
    DEFAULT_COLLAR_S,
    Predictions,
    _resolve,
    collect_predictions,
    match_segments,
    score_at_threshold,
    threshold_for_fa_budget,
)
from vadexplore.labels import make_labels, segments_from_frames
from vadexplore.loader import load_clip
from vadexplore.postprocess import PostprocessConfig, apply_pipeline
from vadexplore.train import load_checkpoint, resolve_device
from vadexplore.viz import plot_clip

OUT_DIR = "runs/postproc"
FIGURE_DIR = "explore_out/figures"

SMOOTH_GRID_MS = [0, 30, 50, 90]
MIN_SPEECH_GRID_MS = [0, 50, 100, 150, 200]
HANGOVER_GRID_MS = [0, 50, 100, 150, 200]
METHODS = ["median", "moving_average"]


def score(predictions: Predictions, threshold: float, config: PostprocessConfig,
          collar_s: float) -> dict:
    """Segment and frame metrics for one post-processing setting."""
    totals = {"matched": 0, "n_reference": 0, "n_hypothesis": 0}
    true_positive = false_positive = false_negative = true_negative = 0

    for probs, labels in zip(predictions.per_clip_scores, predictions.per_clip_labels):
        decisions = apply_pipeline(probs, threshold, config, predictions.fps)
        result = match_segments(segments_from_frames(labels, predictions.fps),
                                segments_from_frames(decisions, predictions.fps),
                                collar_s)
        for key in totals:
            totals[key] += result[key]
        true_positive += int((labels & decisions).sum())
        false_positive += int((~labels & decisions).sum())
        false_negative += int((labels & ~decisions).sum())
        true_negative += int((~labels & ~decisions).sum())

    precision = totals["matched"] / max(totals["n_hypothesis"], 1)
    recall = totals["matched"] / max(totals["n_reference"], 1)
    frame_precision = true_positive / max(true_positive + false_positive, 1)
    frame_recall = true_positive / max(true_positive + false_negative, 1)
    n_frames = true_positive + false_positive + false_negative + true_negative

    return {
        "config": config.to_dict(),
        "label": config.label(),
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "n_hypothesis_segments": totals["n_hypothesis"],
        "n_reference_segments": totals["n_reference"],
        "matched_segments": totals["matched"],
        "frame_precision": frame_precision,
        "frame_recall": frame_recall,
        "frame_f1": 2 * frame_precision * frame_recall
                    / max(frame_precision + frame_recall, 1e-12),
        "frame_accuracy": (true_positive + true_negative) / max(n_frames, 1),
        "frr": false_negative / max(true_positive + false_negative, 1),
        "far": false_positive / max(false_positive + true_negative, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--convention", default=None)
    parser.add_argument("--collar-s", type=float, default=DEFAULT_COLLAR_S,
                        dest="collar_s")
    parser.add_argument("--target-fa-per-hour", type=float, default=100.0,
                        dest="target_fa")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-examples", type=int, default=3, dest="n_examples")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR, dest="figure_dir")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out)
    figure_dir = _resolve(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, payload = load_checkpoint(_resolve(args.run) / "best.pt", device=device)
    data_config = DataConfig(**payload["data_config"])
    convention = args.convention or payload["label_convention"]
    stats = (np.asarray(payload["feature_stats"]["mean"], dtype=np.float32),
             np.asarray(payload["feature_stats"]["std"], dtype=np.float32))

    split_data = load_split()
    validation = collect_predictions(
        model, VADDataset("val", convention, split_data, data_config, stats=stats), device)
    evaluation = collect_predictions(
        model, VADDataset(args.split, convention, split_data, data_config, stats=stats),
        device)

    chosen = threshold_for_fa_budget(validation, args.target_fa)
    threshold = chosen["threshold"]

    print(f"post-processing sweep for {payload.get('run_name')} ({convention} labels)")
    print(f"  threshold {threshold:.4f}, chosen on val for the "
          f"{args.target_fa:g}/h budget and held fixed throughout")
    print(f"  collar {args.collar_s*1000:.0f} ms, "
          f"{len(validation.per_clip_scores)} val clips, "
          f"{len(evaluation.per_clip_scores)} {args.split} clips\n")

    raw = PostprocessConfig(smooth_ms=0, min_speech_ms=0, hangover_ms=0)
    report = {
        "run": str(_resolve(args.run)),
        "run_name": payload.get("run_name"),
        "split": args.split,
        "convention": convention,
        "collar_s": args.collar_s,
        "threshold": threshold,
        "threshold_chosen_on": "val",
        "threshold_held_fixed": True,
        "target_fa_per_hour": args.target_fa,
        "order": "smooth posterior, threshold, min speech duration, hangover",
    }

    # --- (a) raw baseline ---
    report["baseline"] = {"val": score(validation, threshold, raw, args.collar_s),
                          "test": score(evaluation, threshold, raw, args.collar_s)}

    # --- (b) each operation alone, best setting chosen on val ---
    ablations = {}
    for name, grid in (("smooth", SMOOTH_GRID_MS),
                       ("min_speech", MIN_SPEECH_GRID_MS),
                       ("hangover", HANGOVER_GRID_MS)):
        candidates = []
        for value in grid:
            if value == 0:
                continue
            if name == "smooth":
                for method in METHODS:
                    candidates.append(PostprocessConfig(smooth_method=method,
                                                        smooth_ms=value))
            elif name == "min_speech":
                candidates.append(PostprocessConfig(smooth_ms=0, min_speech_ms=value))
            else:
                candidates.append(PostprocessConfig(smooth_ms=0, hangover_ms=value))

        scored = [(score(validation, threshold, c, args.collar_s), c) for c in candidates]
        best_val, best_config = max(scored, key=lambda pair: pair[0]["segment_f1"])
        ablations[name] = {
            "best_config": best_config.to_dict(),
            "label": best_config.label(),
            "val": best_val,
            "test": score(evaluation, threshold, best_config, args.collar_s),
            "all_val": [{"label": c.label(), "segment_f1": s["segment_f1"]}
                        for s, c in scored],
        }
    report["ablations"] = ablations

    # --- (c) the full grid ---
    combinations = []
    for method in METHODS:
        for smooth_ms in SMOOTH_GRID_MS:
            for min_speech_ms in MIN_SPEECH_GRID_MS:
                for hangover_ms in HANGOVER_GRID_MS:
                    if smooth_ms == 0 and method != METHODS[0]:
                        continue  # no smoothing means the method is irrelevant
                    combinations.append(PostprocessConfig(
                        smooth_method=method, smooth_ms=smooth_ms,
                        min_speech_ms=min_speech_ms, hangover_ms=hangover_ms))

    print(f"  sweeping {len(combinations)} settings on val ...", flush=True)
    grid_results = [(score(validation, threshold, c, args.collar_s), c)
                    for c in combinations]
    best_val, best_config = max(grid_results, key=lambda pair: pair[0]["segment_f1"])
    best_test = score(evaluation, threshold, best_config, args.collar_s)

    report["grid"] = {
        "n_settings": len(combinations),
        "selected_on": "validation segment F1",
        "best_config": best_config.to_dict(),
        "best_label": best_config.label(),
        "val": best_val,
        "test": best_test,
        "top_10_on_val": [{"label": c.label(), "val_segment_f1": s["segment_f1"]}
                          for s, c in sorted(grid_results,
                                             key=lambda p: -p[0]["segment_f1"])[:10]],
    }

    # --- figures on representative clips ---
    directory = _resolve(split_data["dataset_dir"])
    gains = []
    for i, (probs, labels) in enumerate(zip(evaluation.per_clip_scores,
                                            evaluation.per_clip_labels)):
        reference = segments_from_frames(labels, evaluation.fps)
        raw_f1 = match_segments(reference,
                                segments_from_frames(probs >= threshold, evaluation.fps),
                                args.collar_s)["f1"]
        post = apply_pipeline(probs, threshold, best_config, evaluation.fps)
        post_f1 = match_segments(reference,
                                 segments_from_frames(post, evaluation.fps),
                                 args.collar_s)["f1"]
        gains.append((post_f1 - raw_f1, i, raw_f1, post_f1))

    gains.sort()
    picks = []
    if gains:
        picks.append(("largest gain", gains[-1]))
        picks.append(("median case", gains[len(gains) // 2]))
        if len(gains) > 2:
            picks.append(("largest regression", gains[0]))
    picks = picks[:args.n_examples]

    figures = []
    for reason, (gain, index, raw_f1, post_f1) in picks:
        stem = evaluation.stems[index]
        clip = load_clip(directory / stem)
        labels = make_labels(clip, fps=data_config.fps,
                             bridge_gap_s=data_config.bridge_gap_s)
        probs = evaluation.per_clip_scores[index]
        raw_decisions = (probs >= threshold).astype(np.int8)
        post_decisions = apply_pipeline(probs, threshold, best_config,
                                        evaluation.fps).astype(np.int8)

        path = figure_dir / f"postproc_{stem}.png"
        fig, _ = plot_clip(
            clip, labels=labels, max_seconds=10.0,
            extra_tracks={
                "posterior": (probs, "prob"),
                f"raw > {threshold:.2f}": raw_decisions,
                f"post ({best_config.label()})": post_decisions,
            },
            title=(f"{stem}   {reason}   segment F1 {raw_f1:.3f} -> {post_f1:.3f} "
                   f"({gain:+.3f})\n"
                   f"threshold {threshold:.4f} from val, held fixed; "
                   f"post-processing {best_config.label()}"),
            save=path,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append({"reason": reason, "stem": stem, "path": str(path),
                        "raw_segment_f1": raw_f1, "post_segment_f1": post_f1})
        print(f"  figure {reason:20s} {stem:20s} F1 {raw_f1:.3f} -> {post_f1:.3f}")

    report["figures"] = figures
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    # --- table ---
    rows = [("raw threshold (baseline)", report["baseline"])]
    for name in ("smooth", "min_speech", "hangover"):
        rows.append((f"+ {name} only ({ablations[name]['label']})", ablations[name]))
    rows.append((f"full pipeline ({report['grid']['best_label']})", report["grid"]))

    print(f"\n{'=' * 96}")
    print(f"POST-PROCESSING on {args.split}   (settings selected on val segment F1, "
          f"threshold fixed at {threshold:.4f})")
    print("=" * 96)
    print(f"  {'setting':<40} {'segF1':>7} {'segP':>7} {'segR':>7} "
          f"{'#hyp':>6} {'frameF1':>8} {'FRR%':>7}")
    baseline_f1 = report["baseline"]["test"]["segment_f1"]
    for name, entry in rows:
        test = entry["test"]
        marker = "" if entry is report["baseline"] else \
            f"  ({test['segment_f1'] - baseline_f1:+.3f})"
        print(f"  {name:<40} {test['segment_f1']:7.3f} {test['segment_precision']:7.3f} "
              f"{test['segment_recall']:7.3f} {test['n_hypothesis_segments']:6d} "
              f"{test['frame_f1']:8.3f} {test['frr']*100:7.2f}{marker}")
    print(f"\n  reference segments on {args.split}: "
          f"{report['baseline']['test']['n_reference_segments']}")
    print(f"  order applied: {report['order']}")
    print(f"\n  results {out_dir / 'results.json'}")
    for figure in figures:
        print(f"  figure  {figure['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
