"""Sweep the attention model's past window and find the elbow.

Windows are given in effective seconds; the model takes a per-layer value.

    python scripts/sweep_past_window.py --epochs 12
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from vadexplore.train import build_configs, load_config, train
from vadexplore.model import VADModel

DEFAULT_WINDOWS = [0.5, 1.0, 2.0, 4.0]
UNBOUNDED = "unbounded"
SWEEP_DIR = "runs/sweep_past_window"
FIGURE = "explore_out/figures/past_window_sweep.png"

EER_COLOR = "#2b6cb0"
REFERENCE_COLOR = "#c53030"
ELBOW_COLOR = "#2f855a"


def per_layer_frames(effective_seconds: float, attn_layers: int, fps: int) -> int:
    """Effective seconds to the per-layer past_window_frames the model takes.

    L layers each reaching back W frames reach back L * W in total.
    """
    if attn_layers < 1:
        raise ValueError(f"attn_layers must be at least 1, got {attn_layers}")
    return max(1, int(round(effective_seconds * fps / attn_layers)))


def best_epoch_by_eer(history_path: Path) -> dict:
    """Pick the epoch with the lowest validation EER.

    EER rather than frr_at_fa, which is too jumpy on a split this size to rank by.
    """
    history = json.loads(history_path.read_text())
    epochs = history["epochs"]
    best = min(epochs, key=lambda record: record["eer"])
    return {
        "epoch": best["epoch"],
        "eer": best["eer"],
        "auc": best["auc"],
        "frr_at_fa": best["frr_at_fa"],
        "fa_per_hour": best["fa_per_hour"],
        "val_loss": best["val_loss"],
        "n_epochs": len(epochs),
    }


def find_elbow(results: list[dict], tolerance_points: float) -> dict:
    """Smallest bounded window whose EER is within tolerance of unbounded."""
    reference = next((r for r in results if r["window"] == UNBOUNDED), None)
    bounded = sorted((r for r in results if r["window"] != UNBOUNDED),
                     key=lambda r: r["window"])
    if reference is None or not bounded:
        return {"found": False, "reason": "no unbounded reference run"}

    reference_eer = reference["eer"] * 100
    for result in bounded:
        gap = result["eer"] * 100 - reference_eer
        if gap <= tolerance_points:
            return {"found": True, "window_s": result["window"],
                    "eer_points_above_unbounded": gap,
                    "tolerance_points": tolerance_points,
                    "per_layer_frames": result["per_layer_frames"],
                    "effective_frames": result["effective_frames"]}
    return {"found": False, "tolerance_points": tolerance_points,
            "reason": f"no bounded window came within {tolerance_points} EER points",
            "closest_window_s": bounded[-1]["window"],
            "closest_gap_points": bounded[-1]["eer"] * 100 - reference_eer}


def make_figure(results: list[dict], elbow: dict, path: Path) -> None:
    bounded = sorted((r for r in results if r["window"] != UNBOUNDED),
                     key=lambda r: r["window"])
    reference = next((r for r in results if r["window"] == UNBOUNDED), None)
    windows = [r["window"] for r in bounded]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, key, label in ((axes[0], "eer", "validation EER (percent)"),
                           (axes[1], "frr_at_fa", "validation FRR at the FA budget (percent)")):
        values = [r[key] * 100 for r in bounded]
        ax.plot(windows, values, "o-", color=EER_COLOR, linewidth=1.8, markersize=6,
                label="bounded window")
        if reference is not None:
            ax.axhline(reference[key] * 100, color=REFERENCE_COLOR, linestyle="--",
                       linewidth=1.5,
                       label=f"unbounded ({reference[key] * 100:.2f}%)")
        if elbow.get("found") and key == "eer":
            ax.axvline(elbow["window_s"], color=ELBOW_COLOR, linewidth=1.5, alpha=0.7)
            ax.annotate(
                f"elbow {elbow['window_s']:g} s\n"
                f"{elbow['eer_points_above_unbounded']:+.2f} points vs unbounded",
                (elbow["window_s"], max(values)), xytext=(8, -6),
                textcoords="offset points", fontsize=9, color=ELBOW_COLOR,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=ELBOW_COLOR, alpha=0.95))
        ax.set_xscale("log", base=2)
        ax.set_xticks(windows)
        ax.set_xticklabels([f"{w:g}" for w in windows])
        ax.set_xlabel("effective past window (s, log scale)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_title("Accuracy against available history", fontsize=10.5, pad=8)
    axes[1].set_title("Operating point at the false-alarm budget", fontsize=10.5, pad=8)
    fig.suptitle("Causal attention past-window sweep", fontsize=11.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--windows", type=float, nargs="+", default=DEFAULT_WINDOWS,
                        help="effective past windows in seconds")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead-frames", type=int, default=5,
                        dest="lookahead_frames")
    parser.add_argument("--label", default="bridged", choices=["literal", "bridged"])
    parser.add_argument("--limit-clips", type=int, default=None, dest="limit_clips")
    parser.add_argument("--tolerance-points", type=float, default=0.3,
                        dest="tolerance_points",
                        help="EER points within unbounded that still counts as the elbow")
    parser.add_argument("--no-progress", action="store_true", dest="no_progress")
    parser.add_argument("--out", default=SWEEP_DIR)
    parser.add_argument("--figure", default=FIGURE)
    args = parser.parse_args(argv)

    base = load_config(args.config)
    fps = 100  # the project frame grid is fixed at 10 ms hop
    attn_layers = int(base["model"]["attn_layers"])

    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {"label": args.label, "lookahead_frames": args.lookahead_frames,
                "attn_layers": attn_layers, "epochs": args.epochs, "seed": args.seed,
                "fps": fps, "limit_clips": args.limit_clips,
                "tolerance_points": args.tolerance_points}

    windows = sorted(args.windows) + [UNBOUNDED]
    print(f"past-window sweep: {len(windows)} runs, {attn_layers} attention layers, "
          f"{fps} fps")
    print(f"  fixed: label {args.label}, lookahead {args.lookahead_frames} frames per layer, "
          f"{args.epochs} epochs, seed {args.seed}")
    print(f"  effective seconds to per-layer frames: round(seconds * {fps} / {attn_layers})\n")

    results = []
    for window in windows:
        name = (f"attn_pw_{UNBOUNDED}" if window == UNBOUNDED
                else f"attn_pw_{window:g}s")
        config = copy.deepcopy(base)
        config["name"] = name
        config["seed"] = args.seed
        config["out_root"] = str(out_dir)
        config["data"]["label"] = args.label
        config["data"]["limit_clips"] = args.limit_clips
        config["model"]["core"] = "causal_attn"
        config["model"]["lookahead_frames"] = args.lookahead_frames
        config["optim"]["epochs"] = args.epochs
        if args.device:
            config["device"] = args.device

        if window == UNBOUNDED:
            per_layer = None
        else:
            per_layer = per_layer_frames(window, attn_layers, fps)
        config["model"]["past_window_frames"] = per_layer

        # read the composed value back off the built model, so the conversion
        # is audited against the model rather than trusted
        _, model_config = build_configs(config)
        composed = VADModel(model_config).effective_past_window_frames

        if window == UNBOUNDED:
            print(f"  {name:24s} effective unbounded  ->  per_layer None  "
                  f"->  model reports {composed}")
        else:
            print(f"  {name:24s} effective {window:>4g} s  ->  per_layer {per_layer:>4d} frames  "
                  f"->  model reports {composed} frames "
                  f"({composed / fps:g} s effective)")
            if composed != per_layer * attn_layers:
                raise RuntimeError(
                    f"composition mismatch for {name}: model says {composed}, "
                    f"expected {per_layer * attn_layers}")

        result = train(config, verbose=True, no_progress=args.no_progress)
        summary = best_epoch_by_eer(Path(result["out_dir"]) / "history.json")
        timing = json.loads((Path(result["out_dir"]) / "timing.json").read_text())

        results.append({
            "window": window,
            "run": name,
            "per_layer_frames": per_layer,
            "effective_frames": composed,
            "effective_seconds": None if composed is None else composed / fps,
            "kv_cache_floats": None if per_layer is None else (
                (per_layer + args.lookahead_frames + 1) * attn_layers
                * model_config.attn_heads * (model_config.d_model // model_config.attn_heads) * 2),
            "total_seconds": timing["total_seconds"],
            **summary,
        })
        print()

    elbow = find_elbow(results, args.tolerance_points)
    payload = {"settings": settings, "results": results, "elbow": elbow,
               "best_epoch_selection": "lowest validation EER across epochs"}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    figure_path = Path(os.path.expanduser(args.figure))
    make_figure(results, elbow, figure_path)

    print("\npast-window sweep, per-run numbers are the lowest-EER epoch")
    print(f"  {'window':>10} {'per-layer':>10} {'effective':>10} {'EER%':>7} {'AUC':>7} "
          f"{'FRR@FA%':>8} {'epoch':>6} {'kv KiB':>8}")
    reference = next((r for r in results if r["window"] == UNBOUNDED), None)
    for result in results:
        window = "unbounded" if result["window"] == UNBOUNDED else f"{result['window']:g} s"
        per_layer = "-" if result["per_layer_frames"] is None else str(result["per_layer_frames"])
        effective = "-" if result["effective_frames"] is None else f"{result['effective_frames']}"
        cache = "-" if result["kv_cache_floats"] is None else f"{result['kv_cache_floats'] * 4 / 1024:.0f}"
        print(f"  {window:>10} {per_layer:>10} {effective:>10} {result['eer'] * 100:7.2f} "
              f"{result['auc']:7.4f} {result['frr_at_fa'] * 100:8.2f} {result['epoch']:6d} "
              f"{cache:>8}")

    if reference is not None:
        print(f"\n  unbounded reference EER {reference['eer'] * 100:.2f}%")
    print(f"\n  elbow rule: smallest window within {args.tolerance_points:g} EER points "
          f"of unbounded")
    if elbow.get("found"):
        print(f"  RECOMMENDED DEPLOYMENT WINDOW: {elbow['window_s']:g} s effective "
              f"({elbow['per_layer_frames']} frames per layer, "
              f"{elbow['effective_frames']} composed)")
        print(f"    costs {elbow['eer_points_above_unbounded']:+.2f} EER points against "
              f"unbounded attention")
    else:
        print(f"  no elbow found: {elbow['reason']}")
        if "closest_gap_points" in elbow:
            print(f"    closest is {elbow['closest_window_s']:g} s at "
                  f"{elbow['closest_gap_points']:+.2f} points; widen the sweep or the tolerance")

    print(f"\n  results {out_dir / 'results.json'}")
    print(f"  figure  {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=None))
