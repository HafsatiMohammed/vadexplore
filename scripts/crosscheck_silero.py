"""Cross-check the provided VAD labels against Silero VAD over the corpus.

Writes explore_out/silero_agreement.json, agreement figures into
explore_out/figures/, and per-clip triptychs for the worst disagreements into
explore_out/disagreements/.

The first run needs network access so torch.hub can cache the model. Every
later run is offline. If the model cannot be loaded the script stops with
instructions rather than inventing outputs.

    python scripts/crosscheck_silero.py [dataset_dir] [--out DIR] [--threshold P]
                                        [--tol-frames N] [--flag-iou X] [--worst N]
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

from vadexplore.labels import make_labels
from vadexplore.loader import load_dataset, read_audio
from vadexplore.silero import (
    NATIVE_RESOLUTION_MS,
    SileroUnavailable,
    agreement,
    disagreement_breakdown,
    load_silero,
    silero_speech_probs,
)
from vadexplore.viz import plot_clip

DEFAULT_DIR = "~/Downloads/vad_data"
DEFAULT_OUT = "explore_out"

# Named clips to inspect regardless of where they rank, from earlier analysis.
CLIPS_OF_INTEREST = {
    "1447-130552-0010": "suspected low-volume speech dropped by the aligner",
    "1334-135589-0065": "noisy clip",
}

LITERAL_COLOR = "#2b6cb0"
BRIDGE_COLOR = "#dd6b20"
ACCENT = "#c53030"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path, generated):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))


def _median(values):
    clean = [v for v in values if v is not None]
    return float(np.median(clean)) if clean else None


# --- figures --------------------------------------------------------------


def fig_agreement(rows, flag_iou, out_dir, generated):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    bins = np.linspace(0, 1, 51)

    for ax, metric, label in ((axes[0], "iou", "IoU"), (axes[1], "f1", "F1")):
        for key, color, name in (("literal", LITERAL_COLOR, "literal"),
                                 ("bridged", BRIDGE_COLOR, "bridged")):
            values = [r[key]["no_collar"][metric] for r in rows
                      if r[key]["no_collar"][metric] is not None]
            ax.hist(values, bins=bins, color=color, alpha=0.65, label=f"{name} (median {np.median(values):.3f})")
        ax.axvline(flag_iou, color=ACCENT, linestyle="--", linewidth=1.4)
        ax.annotate(f"flag below {flag_iou:g}", (flag_iou, ax.get_ylim()[1] * 0.9),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=ACCENT)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        _style(ax, f"Per-clip speech-class {label}, Silero against provided labels",
               f"{label} (no collar)", "clips")

    fig.suptitle("Silero VAD agreement with the provided labels", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "silero_agreement.png", generated)


def fig_boundary_bias(rows, out_dir, generated):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    onset = [r["literal"]["onset_bias_ms_median"] for r in rows
             if r["literal"]["onset_bias_ms_median"] is not None]
    offset = [r["literal"]["offset_bias_ms_median"] for r in rows
              if r["literal"]["offset_bias_ms_median"] is not None]

    span = max(abs(np.percentile(onset + offset, 1)), abs(np.percentile(onset + offset, 99)))
    bins = np.linspace(-span, span, 61)

    ax = axes[0]
    ax.hist(onset, bins=bins, color=LITERAL_COLOR, alpha=0.7,
            label=f"onsets (median {np.median(onset):+.0f} ms)")
    ax.hist(offset, bins=bins, color=BRIDGE_COLOR, alpha=0.6,
            label=f"offsets (median {np.median(offset):+.0f} ms)")
    ax.axvline(0, color="0.3", linewidth=1.2)
    ax.axvspan(-NATIVE_RESOLUTION_MS / 2, NATIVE_RESOLUTION_MS / 2, color="0.5", alpha=0.15)
    ax.annotate(f"Silero native resolution\n+/- {NATIVE_RESOLUTION_MS/2:.0f} ms",
                (0, ax.get_ylim()[1] * 0.97), xytext=(8, 0), textcoords="offset points",
                fontsize=8, va="top", color="0.3")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style(ax, "Boundary bias against literal labels\n(positive means Silero is late)",
           "median per-clip bias (ms)", "clips")

    ax = axes[1]
    combined = [r["literal"]["bias_ms_median"] for r in rows
                if r["literal"]["bias_ms_median"] is not None]
    bridged = [r["bridged"]["bias_ms_median"] for r in rows
               if r["bridged"]["bias_ms_median"] is not None]
    ax.hist(combined, bins=bins, color=LITERAL_COLOR, alpha=0.7,
            label=f"vs literal (median {np.median(combined):+.0f} ms)")
    ax.hist(bridged, bins=bins, color=BRIDGE_COLOR, alpha=0.6,
            label=f"vs bridged (median {np.median(bridged):+.0f} ms)")
    ax.axvline(0, color="0.3", linewidth=1.2)
    ax.axvspan(-NATIVE_RESOLUTION_MS / 2, NATIVE_RESOLUTION_MS / 2, color="0.5", alpha=0.15)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style(ax, "Boundary bias, onsets and offsets combined",
           "median per-clip bias (ms)", "clips")

    fig.suptitle("Where Silero places boundaries relative to the provided labels", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "silero_boundary_bias.png", generated)


# --- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tol-frames", type=int, default=2)
    parser.add_argument("--flag-iou", type=float, default=0.7)
    parser.add_argument("--worst", type=int, default=12)
    parser.add_argument("--collar-sweep", type=int, nargs="+", default=[0, 2, 4, 6, 8],
                        help="collar widths in frames for the boundary vs region sweep")
    parser.add_argument("--max-runs-in-json", type=int, default=15,
                        help="longest region runs kept per clip in the JSON")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        load_silero()
    except SileroUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(os.path.expanduser(args.out))
    fig_dir = out_dir / "figures"
    dis_dir = out_dir / "disagreements"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dis_dir.mkdir(parents=True, exist_ok=True)

    clips = load_dataset(args.directory, limit=args.limit)
    print(f"cross-checking {len(clips)} clips against Silero VAD "
          f"(threshold {args.threshold}, collar {args.tol_frames} frames)")

    rows = []
    cache = {}
    for i, clip in enumerate(clips, 1):
        labels = make_labels(clip)
        probs = silero_speech_probs(read_audio(clip), n_frames=labels["n_frames"])
        row = {
            "stem": clip.stem,
            "speaker_id": clip.speaker_id,
            "duration_s": clip.duration_s,
            "n_frames": labels["n_frames"],
        }
        for key in ("literal", "bridged"):
            row[key] = agreement(labels[key], probs, threshold=args.threshold,
                                 tol_frames=args.tol_frames)
            row[key + "_breakdown"] = disagreement_breakdown(
                labels[key], probs, threshold=args.threshold, tol_frames=args.tol_frames)
            row[key + "_sweep"] = {
                str(tol): {
                    k: v for k, v in disagreement_breakdown(
                        labels[key], probs, threshold=args.threshold, tol_frames=tol).items()
                    if k != "region_runs"
                }
                for tol in args.collar_sweep
            }
        # a clip is only suspect if it disagrees with BOTH conventions
        row["best_iou"] = max(row["literal"]["no_collar"]["iou"] or 0.0,
                              row["bridged"]["no_collar"]["iou"] or 0.0)
        rows.append(row)
        cache[clip.stem] = (clip, labels, probs)
        if i % 200 == 0:
            print(f"  {i}/{len(clips)}")

    # --- summary ---
    summary = {"threshold": args.threshold, "tol_frames": args.tol_frames,
               "flag_iou": args.flag_iou, "n_clips": len(rows)}
    for key in ("literal", "bridged"):
        summary[key] = {
            "median_iou": _median([r[key]["no_collar"]["iou"] for r in rows]),
            "median_f1": _median([r[key]["no_collar"]["f1"] for r in rows]),
            "median_kappa": _median([r[key]["no_collar"]["kappa"] for r in rows]),
            "median_accuracy": _median([r[key]["no_collar"]["accuracy"] for r in rows]),
            "median_iou_with_collar": _median([r[key]["with_collar"]["iou"] for r in rows]),
            "median_f1_with_collar": _median([r[key]["with_collar"]["f1"] for r in rows]),
            "median_kappa_with_collar": _median([r[key]["with_collar"]["kappa"] for r in rows]),
            "median_boundary_bias_ms": _median([r[key]["bias_ms_median"] for r in rows]),
            "median_onset_bias_ms": _median([r[key]["onset_bias_ms_median"] for r in rows]),
            "median_offset_bias_ms": _median([r[key]["offset_bias_ms_median"] for r in rows]),
            "n_below_flag_iou": int(sum(
                1 for r in rows if (r[key]["no_collar"]["iou"] or 0.0) < args.flag_iou)),
            "total_boundary_runs": int(sum(r[key + "_breakdown"]["n_boundary_runs"] for r in rows)),
            "total_region_runs": int(sum(r[key + "_breakdown"]["n_region_runs"] for r in rows)),
            "total_boundary_frames": int(sum(r[key + "_breakdown"]["boundary_frames"] for r in rows)),
            "total_region_frames": int(sum(r[key + "_breakdown"]["region_frames"] for r in rows)),
            "region_frames_silero_missed_speech": int(sum(
                run["n_frames"] for r in rows
                for run in r[key + "_breakdown"]["region_runs"]
                if run["kind"] == "silero_missed_speech")),
            "region_frames_silero_extra_speech": int(sum(
                run["n_frames"] for r in rows
                for run in r[key + "_breakdown"]["region_runs"]
                if run["kind"] == "silero_extra_speech")),
        }
    for key in ("literal", "bridged"):
        summary[key]["collar_sweep"] = {}
        for tol in args.collar_sweep:
            b = sum(r[key + "_sweep"][str(tol)]["boundary_frames"] for r in rows)
            g = sum(r[key + "_sweep"][str(tol)]["region_frames"] for r in rows)
            summary[key]["collar_sweep"][str(tol)] = {
                "collar_ms": tol * 10,
                "boundary_runs": sum(r[key + "_sweep"][str(tol)]["n_boundary_runs"] for r in rows),
                "region_runs": sum(r[key + "_sweep"][str(tol)]["n_region_runs"] for r in rows),
                "boundary_frames": b,
                "region_frames": g,
                "boundary_share": b / (b + g) if (b + g) else None,
            }

    summary["n_below_flag_iou_both_conventions"] = int(
        sum(1 for r in rows if r["best_iou"] < args.flag_iou))

    # --- figures ---
    generated = []
    fig_agreement(rows, args.flag_iou, fig_dir, generated)
    fig_boundary_bias(rows, fig_dir, generated)

    # --- disagreement triptychs ---
    worst = sorted(rows, key=lambda r: r["best_iou"])[:args.worst]
    selected = [(r, "worst agreement") for r in worst]
    chosen_stems = {r["stem"] for r in worst}
    for stem, reason in CLIPS_OF_INTEREST.items():
        match = next((r for r in rows if r["stem"] == stem), None)
        if match is None:
            print(f"  note: clip of interest {stem} not in dataset, skipped")
        elif stem not in chosen_stems:
            selected.append((match, f"clip of interest: {reason}"))
            chosen_stems.add(stem)
        else:
            selected[[r["stem"] for r, _ in selected].index(stem)] = (
                match, f"worst agreement, also clip of interest: {reason}")

    print(f"\nwriting {len(selected)} disagreement triptychs to {dis_dir}")
    for row, reason in selected:
        clip, labels, probs = cache[row["stem"]]
        breakdown = row["literal_breakdown"]
        fig, _ = plot_clip(
            clip, labels=labels,
            extra_tracks={"silero P(speech)": (probs, "prob"),
                          f"silero > {args.threshold:g}": (probs >= args.threshold).astype(np.int8)},
            title=(f"{clip.stem}   speaker {clip.speaker_id}   {clip.duration_s:.2f} s   "
                   f"[{reason}]\n"
                   f"vs literal: IoU {row['literal']['no_collar']['iou']:.3f}, "
                   f"F1 {row['literal']['no_collar']['f1']:.3f}, "
                   f"kappa {row['literal']['no_collar']['kappa']:.3f}   |   "
                   f"vs bridged: IoU {row['bridged']['no_collar']['iou']:.3f}   |   "
                   f"{breakdown['n_region_runs']} region disagreements, "
                   f"{breakdown['n_boundary_runs']} boundary-only"),
        )
        _save(fig, dis_dir / f"{clip.stem}.png", generated)
        print(f"  {row['best_iou']:.3f}  {clip.stem:20s} {reason}")

    # --- json ---
    payload = {
        "dataset_dir": os.path.expanduser(args.directory),
        "model": {
            "repo": "snakers4/silero-vad", "license": "MIT",
            "native_window_ms": NATIVE_RESOLUTION_MS,
            "native_fps": 1000.0 / NATIVE_RESOLUTION_MS,
            "grid_mapping": "linear interpolation from window centers onto 100 fps frame centers",
            "inherent_boundary_uncertainty_ms": NATIVE_RESOLUTION_MS / 2,
            "preprocessing": "80 Hz high-pass, matching the committed front-end",
            "output_used": "raw per-window probabilities, not get_speech_timestamps",
        },
        "summary": summary,
        "clips_of_interest": CLIPS_OF_INTEREST,
        "per_clip": rows,
    }
    # Keep only the longest region runs per clip. Tens of thousands of one and
    # two frame runs would dominate the file without being readable evidence.
    for row in rows:
        for key in ("literal", "bridged"):
            # the sweep is only meaningful in aggregate, and it is already in summary
            row.pop(key + "_sweep", None)
            runs = row[key + "_breakdown"]["region_runs"]
            row[key + "_breakdown"]["n_region_runs_total"] = len(runs)
            row[key + "_breakdown"]["region_runs"] = sorted(
                runs, key=lambda r: -r["n_frames"])[:args.max_runs_in_json]

    json_path = out_dir / "silero_agreement.json"
    json_path.write_text(json.dumps(payload, indent=2))
    generated.append(str(json_path))

    _print_summary(summary, args)
    print("\ngenerated files")
    for path in sorted(generated):
        print(f"  {path}  ({os.path.getsize(path):,} bytes)")
    return 0


def _print_summary(summary, args):
    print(f"\n{'=' * 78}\nSILERO AGREEMENT  ({summary['n_clips']} clips, "
          f"threshold {args.threshold}, collar {args.tol_frames} frames)\n{'=' * 78}")
    print(f"  {'':<10} {'IoU':>7} {'F1':>7} {'kappa':>7} {'acc':>7}   "
          f"{'IoU+collar':>11} {'F1+collar':>10} {'kappa+col':>10}")
    for key in ("literal", "bridged"):
        s = summary[key]
        print(f"  {key:<10} {s['median_iou']:7.3f} {s['median_f1']:7.3f} "
              f"{s['median_kappa']:7.3f} {s['median_accuracy']:7.3f}   "
              f"{s['median_iou_with_collar']:11.3f} {s['median_f1_with_collar']:10.3f} "
              f"{s['median_kappa_with_collar']:10.3f}")

    print(f"\n  boundary bias (positive means Silero is late), median over clips")
    for key in ("literal", "bridged"):
        s = summary[key]
        print(f"    vs {key:<8} onsets {s['median_onset_bias_ms']:+6.1f} ms   "
              f"offsets {s['median_offset_bias_ms']:+6.1f} ms   "
              f"combined {s['median_boundary_bias_ms']:+6.1f} ms")
    print(f"    Silero's own resolution is +/- {NATIVE_RESOLUTION_MS/2:.0f} ms, "
          f"so bias inside that band is not meaningful")

    print(f"\n  clips below IoU {args.flag_iou:g}")
    for key in ("literal", "bridged"):
        print(f"    vs {key:<8} {summary[key]['n_below_flag_iou']:4d}")
    print(f"    below against BOTH conventions: "
          f"{summary['n_below_flag_iou_both_conventions']} "
          f"(these are the real candidates for label review)")

    print(f"\n  disagreement kind (collar {args.tol_frames} frames)")
    for key in ("literal", "bridged"):
        s = summary[key]
        total = s["total_boundary_frames"] + s["total_region_frames"]
        if total == 0:
            continue
        print(f"    vs {key}")
        print(f"      boundary-only  {s['total_boundary_runs']:6d} runs, "
              f"{s['total_boundary_frames']:7d} frames ({s['total_boundary_frames']/total*100:4.1f}%)  "
              f"expected method difference")
        print(f"      region         {s['total_region_runs']:6d} runs, "
              f"{s['total_region_frames']:7d} frames ({s['total_region_frames']/total*100:4.1f}%)  "
              f"candidate label errors")
        print(f"        of the region frames, {s['region_frames_silero_missed_speech']} are labeled "
              f"speech Silero calls silence,")
        print(f"        {s['region_frames_silero_extra_speech']} are labeled silence Silero calls speech")

    print(f"\n  collar sensitivity: share of disagreed frames that is boundary-only")
    tols = sorted(int(t) for t in summary["literal"]["collar_sweep"])
    print("    " + "collar".ljust(10) + "".join(f"{t*10:>8} ms" for t in tols))
    for key in ("literal", "bridged"):
        cells = "".join(
            f"{summary[key]['collar_sweep'][str(t)]['boundary_share']*100:>9.1f}%" for t in tols)
        print(f"    vs {key:<7}" + cells)
    print(f"    Silero lags the reference by about "
          f"{summary['literal']['median_boundary_bias_ms']:+.0f} ms, so a collar narrower than")
    print(f"    that cannot absorb its causal latency and reports shifted boundaries as regions.")


if __name__ == "__main__":
    raise SystemExit(main())
