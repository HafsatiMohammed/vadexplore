"""Score the trained model against Silero and WebRTC on the same test split.

    python scripts/eval_baselines.py --run runs/<name>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import webrtcvad

from vadexplore.config import DataConfig
from vadexplore.data import load_split, partition_stems
from vadexplore.evaluate import (
    DEFAULT_COLLAR_S,
    DEFAULT_FA_TARGETS,
    Predictions,
    collect_predictions,
    curve_points,
    plot_det,
    score_at_threshold,
    segment_metrics,
    threshold_for_fa_budget,
    threshold_free_metrics,
)
from vadexplore.features import DEFAULT_SR
from vadexplore.labels import make_labels
from vadexplore.loader import load_clip, read_audio
from vadexplore.preprocess import highpass
from vadexplore.silero import silero_speech_probs
from vadexplore.train import load_checkpoint, resolve_device

WEBRTC_MODES = (0, 1, 2, 3)
WEBRTC_FRAME_MS = 10  # the only size that lands one decision per label frame
OUT_DIR = "runs/baselines"
FIGURE = "explore_out/figures/baseline_det.png"


def _resolve(path) -> Path:
    return Path(os.path.expanduser(str(path)))


def webrtc_frames(audio: np.ndarray, n_frames: int, mode: int,
                  sr: int = DEFAULT_SR) -> np.ndarray:
    """WebRTC decisions on the 100 fps grid, one per label frame.

    At 16 kHz WebRTC's 10 ms frame is 160 samples, which is the project hop, so
    nothing has to be resampled.
    """
    hop = sr * WEBRTC_FRAME_MS // 1000
    needed = n_frames * hop
    signal = np.asarray(audio, dtype=np.float32)
    if len(signal) < needed:
        signal = np.pad(signal, (0, needed - len(signal)))
    else:
        signal = signal[:needed]

    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    detector = webrtcvad.Vad(mode)
    decisions = np.empty(n_frames, dtype=bool)
    for i in range(n_frames):
        decisions[i] = detector.is_speech(pcm[i * hop:(i + 1) * hop].tobytes(), sr)
    return decisions


def gather(stems, directory, convention, config, system: str, mode: int | None = None):
    """Build a Predictions object for one system over one list of stems."""
    scores, labels, kept = [], [], []
    for stem in stems:
        clip = load_clip(directory / stem)
        reference = make_labels(clip, fps=config.fps,
                                bridge_gap_s=config.bridge_gap_s)[convention].astype(bool)
        # every system is scored on this same conditioned signal
        audio = highpass(read_audio(clip, target_sr=config.sample_rate),
                         config.sample_rate, config.highpass_hz, config.highpass_order)
        n_frames = len(reference)

        if system == "webrtc":
            frame_scores = webrtc_frames(audio, n_frames, mode).astype(np.float32)
        elif system == "silero":
            frame_scores = silero_speech_probs(
                audio, sr=config.sample_rate, fps=config.fps,
                n_frames=n_frames, apply_highpass=False)  # already conditioned
        else:
            raise ValueError(f"unknown system {system!r}")

        assert len(frame_scores) == n_frames, \
            f"{system} produced {len(frame_scores)} frames for {n_frames} labels"
        scores.append(frame_scores)
        labels.append(reference)
        kept.append(stem)
    return Predictions(scores, labels, kept, fps=config.fps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory with best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--convention", default=None,
                        help="defaults to the convention the checkpoint was trained on")
    parser.add_argument("--collar-s", type=float, default=DEFAULT_COLLAR_S,
                        dest="collar_s")
    parser.add_argument("--collar-sweep", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20], dest="collar_sweep",
                        help="extra collars for the fairness check")
    parser.add_argument("--fa-targets", type=float, nargs="+",
                        default=list(DEFAULT_FA_TARGETS), dest="fa_targets")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--figure", default=FIGURE)
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    model, payload = load_checkpoint(_resolve(args.run) / "best.pt", device=device)
    config = DataConfig(**payload["data_config"])
    convention = args.convention or payload["label_convention"]
    stats = (np.asarray(payload["feature_stats"]["mean"], dtype=np.float32),
             np.asarray(payload["feature_stats"]["std"], dtype=np.float32))

    split_data = load_split()
    directory = _resolve(split_data["dataset_dir"])
    val_stems = partition_stems(split_data, "val")
    test_stems = partition_stems(split_data, args.split)

    print(f"baselines on the {args.split} split, {convention} labels, "
          f"{len(test_stems)} clips")
    print(f"  every system sees the same {config.highpass_hz:g} Hz high-passed audio "
          f"and the same frames")
    print(f"  thresholds are chosen on val ({len(val_stems)} clips) and applied "
          f"unchanged to {args.split}\n")

    from vadexplore.data import VADDataset
    systems: dict[str, dict] = {}

    model_val = collect_predictions(
        model, VADDataset("val", convention, split_data, config, stats=stats), device)
    model_test = collect_predictions(
        model, VADDataset(args.split, convention, split_data, config, stats=stats), device)
    systems[f"model ({payload.get('run_name')})"] = {
        "kind": "scored", "val": model_val, "test": model_test}

    systems["silero"] = {
        "kind": "scored",
        "val": gather(val_stems, directory, convention, config, "silero"),
        "test": gather(test_stems, directory, convention, config, "silero"),
    }

    for mode in WEBRTC_MODES:
        print(f"  webrtc mode {mode} ...", flush=True)
        systems[f"webrtc mode {mode}"] = {
            "kind": "binary", "mode": mode,
            "val": gather(val_stems, directory, convention, config, "webrtc", mode),
            "test": gather(test_stems, directory, convention, config, "webrtc", mode),
        }

    report = {
        "split": args.split,
        "convention": convention,
        "collar_s": args.collar_s,
        "n_test_clips": len(test_stems),
        "discipline": (
            "Thresholds come from the validation split and are applied unchanged to "
            f"{args.split}. EER and AUC need no threshold and are computed on "
            f"{args.split} directly. WebRTC emits binary decisions, so it has no "
            "threshold and no curve; its aggressiveness mode is its operating point."
        ),
        "systems": {},
    }

    for name, entry in systems.items():
        record = {"kind": entry["kind"]}
        record["threshold_free"] = threshold_free_metrics(entry["test"])

        if entry["kind"] == "scored":
            record["curve"] = curve_points(entry["test"])
            record["operating_points"] = []
            for target in args.fa_targets:
                chosen = threshold_for_fa_budget(entry["val"], target)
                applied = score_at_threshold(entry["test"], chosen["threshold"])
                record["operating_points"].append({
                    "target_fa_per_hour": target,
                    "threshold": chosen["threshold"],
                    "threshold_chosen_on": "val",
                    "budget_met_on_val": chosen["budget_met"],
                    "test_frame": applied,
                    "test_segment": segment_metrics(entry["test"], chosen["threshold"],
                                                    args.collar_s),
                    "test_segment_f1_by_collar": {
                        f"{c:.3f}": segment_metrics(entry["test"], chosen["threshold"], c)["f1"]
                        for c in args.collar_sweep},
                })
        else:
            # a binary system has one operating point: threshold 0.5 over 0/1
            applied = score_at_threshold(entry["test"], 0.5)
            val_applied = score_at_threshold(entry["val"], 0.5)
            record["operating_points"] = [{
                "target_fa_per_hour": None,
                "threshold": 0.5,
                "threshold_chosen_on": "not applicable, binary output",
                "test_frame": applied,
                "val_frame": val_applied,
                "test_segment": segment_metrics(entry["test"], 0.5, args.collar_s),
                "test_segment_f1_by_collar": {
                    f"{c:.3f}": segment_metrics(entry["test"], 0.5, c)["f1"]
                    for c in args.collar_sweep},
            }]
        # Also score every system at the neutral 0.5 threshold. The FA-budget
        # threshold is chosen for one operating point and can be extreme for a
        # system whose score distribution differs from the model's, which makes
        # its segment numbers say more about the budget than about the system.
        record["at_default_threshold"] = {
            "threshold": 0.5,
            "note": "neutral 0.5, not tuned on any split",
            "test_frame": score_at_threshold(entry["test"], 0.5),
            "test_segment_f1_by_collar": {
                f"{c:.3f}": segment_metrics(entry["test"], 0.5, c)["f1"]
                for c in args.collar_sweep},
        }
        report["systems"][name] = record

    # which WebRTC mode validation would pick, so the choice is not made on test
    webrtc = {n: r for n, r in report["systems"].items() if r["kind"] == "binary"}
    if webrtc:
        best_mode = min(webrtc, key=lambda n: webrtc[n]["operating_points"][0]
                        ["val_frame"]["frr"] + webrtc[n]["operating_points"][0]["val_frame"]["far"])
        report["webrtc_mode_selected_on_val"] = best_mode

    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2))

    palette = {"model": "#2b6cb0", "silero": "#2f855a"}
    curves = {}
    for name, record in report["systems"].items():
        if record["kind"] == "scored":
            color = palette["silero"] if name.startswith("silero") else palette["model"]
            curves[name] = (record["curve"]["points"], color, "-")
    markers = []
    shapes = ["o", "s", "^", "D"]
    for i, (name, record) in enumerate(
            [(n, r) for n, r in report["systems"].items() if r["kind"] == "binary"]):
        frame = record["operating_points"][0]["test_frame"]
        if frame["fa_per_hour"] > 0 and frame["frr"] > 0:
            markers.append({"label": name, "fa_per_hour": frame["fa_per_hour"],
                            "frr": frame["frr"], "color": "#c53030",
                            "marker": shapes[i % len(shapes)]})
    figure_path = _resolve(args.figure)
    plot_det(curves, markers, figure_path,
             f"Model against reference VADs, {args.split} split ({convention} labels)")

    print()
    print(f"COMPARISON on {args.split}  ({convention} labels, collar "
          f"{args.collar_s*1000:.0f} ms)")
    collars = args.collar_sweep
    header_collars = "".join(f"{'F1@' + str(int(c*1000)) + 'ms':>9}" for c in collars)
    print(f"  {'system':<26} {'EER%':>7} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'FRR%':>7} {'fa/h':>8}{header_collars}")
    for name, record in report["systems"].items():
        free = record["threshold_free"]
        point = record["operating_points"][-1]
        frame = point["test_frame"]
        binary = record["kind"] == "binary"
        eer = "-" if binary else f"{free['eer']*100:7.2f}"
        auc = "-" if binary else f"{free['roc_auc']:8.4f}"
        prauc = "-" if binary else f"{free['pr_auc']:8.4f}"
        cells = "".join(f"{point['test_segment_f1_by_collar'][f'{c:.3f}']:9.3f}"
                        for c in collars)
        print(f"  {name:<26} {eer:>7} {auc:>8} {prauc:>8} "
              f"{frame['frr']*100:7.2f} {frame['fa_per_hour']:8.1f}{cells}")

    print(f"\n  {'system':<26} {'FRR%':>7} {'fa/h':>8}{header_collars}   at the neutral 0.5 threshold")
    for name, record in report["systems"].items():
        default = record["at_default_threshold"]
        cells = "".join(f"{default['test_segment_f1_by_collar'][f'{c:.3f}']:9.3f}"
                        for c in collars)
        print(f"  {name:<26} {default['test_frame']['frr']*100:7.2f} "
              f"{default['test_frame']['fa_per_hour']:8.1f}{cells}")

    if webrtc:
        print(f"  webrtc mode chosen on val: {report['webrtc_mode_selected_on_val']}")
    print(f"\n  results {out_dir / f'eval_{args.split}.json'}")
    print(f"  figure  {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
