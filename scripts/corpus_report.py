"""Whole-corpus analysis. Writes corpus_stats.json and figures.

    python scripts/corpus_report.py [dataset_dir] [--out DIR] [--bridge-gap S]
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
from matplotlib.gridspec import GridSpec
import torch
import torchaudio

from vadexplore import stats as S
from vadexplore.features import band_fraction, frame_energy, power_spectrogram
from vadexplore.labels import make_labels
from vadexplore.loader import load_dataset, read_audio
from vadexplore.viz import plot_clip

DEFAULT_DIR = "~/Downloads/vad_data"
DEFAULT_OUT = "explore_out"

SPEECH_COLOR = "#2b6cb0"
BRIDGE_COLOR = "#dd6b20"
ACCENT = "#c53030"
GRID = dict(alpha=0.25, linewidth=0.6)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out_dir: Path, name: str, generated: list[str]) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))


def fig_speakers(table, split, out_dir, generated):
    clips = np.array([table["clips"][s] for s in table["speakers"]])
    minutes = np.array([table["minutes"][s] for s in table["speakers"]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].hist(clips, bins=range(int(clips.min()), int(clips.max()) + 3), color=SPEECH_COLOR)
    _style(axes[0], f"Clips per speaker ({len(clips)} speakers)", "clips", "speakers")

    axes[1].hist(minutes, bins=15, color=SPEECH_COLOR)
    _style(axes[1], "Minutes per speaker", "minutes of audio", "speakers")

    colors = {"train": SPEECH_COLOR, "val": BRIDGE_COLOR, "test": ACCENT}
    part_of = {s: p for p, v in split["partitions"].items() for s in v["speakers"]}
    order = sorted(table["speakers"], key=lambda s: (part_of[s], -table["clips"][s]))
    axes[2].bar(range(len(order)), [table["clips"][s] for s in order],
                color=[colors[part_of[s]] for s in order])
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order, rotation=90, fontsize=6)
    _style(axes[2], "Proposed speaker-disjoint split", "speaker id", "clips")
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[p],
                   label=f"{p} ({split['partitions'][p]['n_clips']} clips, "
                         f"{split['partitions'][p]['clip_fraction']*100:.0f}%)")
                   for p in ("train", "val", "test")], fontsize=8, frameon=False)

    fig.suptitle("Corpus composition and split feasibility", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "speakers.png", generated)


def fig_durations(rows, out_dir, generated):
    durations = np.array([r["duration_s"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(durations, bins=40, color=SPEECH_COLOR)
    for value, label, color in [
        (durations.min(), "min", "0.4"),
        (np.median(durations), "median", ACCENT),
        (durations.max(), "max", "0.4"),
    ]:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.2)
        ax.annotate(f"{label} {value:.2f} s", (value, ax.get_ylim()[1] * 0.92),
                    rotation=90, fontsize=8, ha="right", va="top", color=color)
    _style(ax, f"Clip duration  ({len(durations)} clips, "
               f"{durations.sum()/3600:.2f} h total)", "duration (s)", "clips")
    fig.tight_layout()
    _save(fig, out_dir, "durations.png", generated)


def fig_class_balance(rows, overall, out_dir, generated):
    literal = np.array([r["speech_fraction_literal"] for r in rows])
    bridged = np.array([r["speech_fraction_bridged"] for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(literal, bins=bins, color=SPEECH_COLOR, alpha=0.75, label="literal")
    ax.hist(bridged, bins=bins, color=BRIDGE_COLOR, alpha=0.55, label="bridged")
    ax.axvline(overall["literal"], color=SPEECH_COLOR, linestyle="--", linewidth=1.4)
    ax.axvline(overall["bridged"], color=BRIDGE_COLOR, linestyle="--", linewidth=1.4)
    _style(ax, "Per-clip speech fraction", "speech fraction of clip", "clips")
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.02, 0.97,
            f"corpus speech fraction\n"
            f"literal  {overall['literal']*100:.1f}%  (imbalance {overall['literal']/(1-overall['literal']):.2f} : 1)\n"
            f"bridged  {overall['bridged']*100:.1f}%  (imbalance {overall['bridged']/(1-overall['bridged']):.2f} : 1)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.8"))
    fig.tight_layout()
    _save(fig, out_dir, "class_balance.png", generated)


def fig_segments_and_gaps(rows, bridging, elbow, out_dir, generated):
    """The bridging justification figure."""
    lengths = np.array([x for r in rows for x in r["segment_lengths_s"]])
    gaps = np.array([g for r in rows for g in r["gaps_s"]])
    gaps = gaps[gaps > 0]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    palette = ["#2b6cb0", "#2f855a", "#dd6b20", "#c53030"]

    ax = axes[0]
    ax.hist(lengths, bins=np.logspace(np.log10(max(lengths.min(), 1e-3)),
                                      np.log10(lengths.max()), 45), color=SPEECH_COLOR)
    ax.set_xscale("log")
    median = float(np.median(lengths))
    ax.axvline(median, color=ACCENT, linestyle="--", linewidth=1.3)
    ax.text(0.97, 0.97, f"median {median * 1000:.0f} ms\nmean {lengths.mean() * 1000:.0f} ms",
            transform=ax.transAxes, fontsize=9, ha="right", va="top", color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8"))
    _style(ax, f"Speech segment length  ({len(lengths)} segments)",
           "segment length (s, log scale)", "segments")

    ax = axes[1]
    edges = np.logspace(np.log10(max(gaps.min(), 1e-4)), np.log10(gaps.max()), 55)
    ax.hist(gaps, bins=edges, color="0.6")
    ax.set_xscale("log")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.18)

    for i, entry in enumerate(bridging):
        t = entry["threshold_s"]
        ax.axvline(t, color=palette[i], linewidth=1.7)
        # labels sit in the empty mid-height band, clear of the table above
        ax.annotate(f"{t * 1000:.0f} ms", (t, ymax * 0.60), rotation=90, fontsize=9,
                    ha="right", va="bottom", color=palette[i], fontweight="bold")

    header = f"{'thresh':>7}  {'gaps':>6}  {'seg/clip':>8}  {'speech':>7}"
    lines = [header, "-" * len(header)]
    for entry in bridging:
        lines.append(
            f"{entry['threshold_s'] * 1000:5.0f}ms  "
            f"{entry['gap_fraction_bridged'] * 100:5.1f}%  "
            f"{entry['mean_segments_per_clip']:8.1f}  "
            f"{entry['speech_fraction'] * 100:6.1f}%"
        )
    ax.text(0.985, 0.985, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            family="monospace", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.7"))
    _style(ax, f"Non-speech gaps, full range  ({len(gaps)} gaps, "
               "leading and trailing silence excluded)",
           "gap length (s, log scale)", "gaps")

    ax = axes[2]
    centers = np.array(elbow["bucket_centers_ms"])
    counts = np.array(elbow["bucket_counts"])
    keep = centers <= 400
    ax.bar(centers[keep], counts[keep], width=8.5, color="0.6")

    plateau = elbow["plateau_count_per_bucket"]
    ax.axhline(plateau, color="0.35", linestyle=":", linewidth=1.3)
    ax.annotate(f"plateau {plateau:.0f} gaps per 10 ms bucket",
                (400, plateau), xytext=(-6, 7), textcoords="offset points",
                fontsize=8.5, ha="right", color="0.3")

    elbow_ms = elbow["elbow_ms"]
    ax.axvspan(0, elbow_ms, color=SPEECH_COLOR, alpha=0.10)
    ax.axvline(elbow_ms, color="black", linewidth=1.8)
    ax.annotate(f"elbow {elbow_ms:.0f} ms",
                (elbow_ms, ax.get_ylim()[1] * 0.72), xytext=(10, 0),
                textcoords="offset points", fontsize=9, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6"))

    for i, entry in enumerate(bridging):
        t_ms = entry["threshold_s"] * 1000
        if t_ms <= 400:
            ax.axvline(t_ms, color=palette[i], linewidth=1.4, linestyle="--")
            ax.annotate(f"{t_ms:.0f}", (t_ms, ax.get_ylim()[1] * 0.97), rotation=90,
                        fontsize=8, ha="right", va="top", color=palette[i])

    _style(ax, "Gap detail, 10 ms buckets  (93 percent of gaps are exact 10 ms multiples)",
           "gap length (ms, linear)", "gaps")

    fig.suptitle("Segment and gap structure", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "segments_and_gaps.png", generated)


def fig_snr(snr, snr_hp, gate, n_skipped, cutoff_hz, out_dir, generated):
    snr, snr_hp = np.asarray(snr), np.asarray(snr_hp)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))

    bins = np.linspace(min(snr.min(), snr_hp.min()), max(snr.max(), snr_hp.max()), 50)
    ax = axes[0]
    ax.hist(snr, bins=bins, color=SPEECH_COLOR, alpha=0.8, label="as recorded")
    ax.hist(snr_hp, bins=bins, color="#2f855a", alpha=0.55,
            label=f"after {cutoff_hz:.0f} Hz high-pass")
    ax.axvline(gate["threshold_db"], color=ACCENT, linewidth=1.8)
    ax.annotate(
        f"gate {gate['threshold_db']:.0f} dB\n"
        f"below gate: {int((snr < gate['threshold_db']).sum())} as recorded,\n"
        f"{int((snr_hp < gate['threshold_db']).sum())} after high-pass",
        (gate["threshold_db"], ax.get_ylim()[1] * 0.96), xytext=(10, 0),
        textcoords="offset points", fontsize=8.5, va="top", color=ACCENT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ACCENT))
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style(ax, f"Per-clip SNR: labeled speech vs labeled silence "
               f"({len(snr)} clips, {n_skipped} skipped)", "SNR (dB)", "clips")

    ax = axes[1]
    gain = snr_hp - snr
    ax.scatter(snr, gain, s=7, alpha=0.4, color=SPEECH_COLOR, edgecolors="none")
    ax.axhline(0, color="0.4", linewidth=0.9)
    ax.axvline(gate["threshold_db"], color=ACCENT, linewidth=1.4, linestyle="--")
    _style(ax, f"SNR gained by the {cutoff_hz:.0f} Hz high-pass",
           "SNR as recorded (dB)", "SNR gain (dB)")

    fig.suptitle("SNR and the clean vs degraded gate", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "snr.png", generated)


def fig_rumble(rows, gate, out_dir, generated):
    lf_all = np.array([r["lf_fraction_all"] for r in rows])
    lf_sil = np.array([r["lf_fraction_silence"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))

    axes[0].hist(lf_all, bins=45, color="0.55", label="whole clip")
    axes[0].hist(lf_sil, bins=45, color=ACCENT, alpha=0.6, label="labeled silence only")
    axes[0].legend(fontsize=9, frameon=False)
    _style(axes[0], f"Share of energy below {S.RUMBLE_HZ:.0f} Hz",
           "sub-80 Hz energy fraction", "clips")

    axes[1].hist(lf_sil, bins=45, color=ACCENT)
    axes[1].axvline(gate["threshold_lf_fraction_in_silence"], color="black", linewidth=1.6)
    axes[1].annotate(
        f"proposed rumble gate {gate['threshold_lf_fraction_in_silence']:.2f}\n"
        f"{gate['n_above']} clips above ({gate['fraction_above']*100:.1f}%)\n"
        f"median {gate['median']:.3f}",
        (gate["threshold_lf_fraction_in_silence"], axes[1].get_ylim()[1] * 0.95),
        xytext=(12, 0), textcoords="offset points", fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.6"))
    _style(axes[1], "Sub-80 Hz energy within labeled silence",
           "sub-80 Hz energy fraction in silence", "clips")

    fig.suptitle("Low-frequency rumble check", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "rumble.png", generated)


def fig_position(profile, out_dir, generated):
    pos = np.array(profile["position"])
    prob = np.array(profile["mean_speech_probability"])

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(pos, prob, color=SPEECH_COLOR, linewidth=2)
    ax.fill_between(pos, 0, prob, color=SPEECH_COLOR, alpha=0.15)
    ax.axhline(prob.mean(), color="0.5", linestyle=":", linewidth=1.2)
    ax.annotate(f"clip mean {prob.mean()*100:.1f}%", (0.5, prob.mean()),
                xytext=(0, -14), textcoords="offset points", fontsize=8.5, color="0.35", ha="center")
    ax.set_ylim(0, 1)
    for x, value, label in [(0.05, profile["first_decile_mean"], "first decile"),
                            (0.95, profile["last_decile_mean"], "last decile")]:
        ax.annotate(f"{label}\n{value*100:.1f}%", (x, value), xytext=(0, 16),
                    textcoords="offset points", fontsize=8.5, ha="center", color=ACCENT)
        ax.plot([x], [value], "o", color=ACCENT, markersize=5)
    _style(ax, "Mean speech probability against normalized position in clip",
           "position within clip (0 = start, 1 = end)", "P(speech)")
    fig.tight_layout()
    _save(fig, out_dir, "position.png", generated)


def fig_highpass_demo(clip, cutoff_hz, out_dir, generated):
    """Before and after a high-pass on the worst rumble clip, in one figure."""
    labels = make_labels(clip)
    raw = read_audio(clip)
    filtered = torchaudio.functional.highpass_biquad(
        torch.from_numpy(raw), 16000, cutoff_hz
    ).numpy().astype(np.float32)

    def lf_in_silence(audio):
        power, freqs = power_spectrogram(audio, n_frames=labels["n_frames"])
        energy = frame_energy(power)
        low = band_fraction(power, freqs, 0.0, S.RUMBLE_HZ)
        silence = ~labels["literal"].astype(bool)
        total = float(energy[silence].sum())
        return float((low[silence] * energy[silence]).sum() / total) if total > 0 else 0.0

    before, after = lf_in_silence(raw), lf_in_silence(filtered)

    fig = plt.figure(figsize=(13, 13))
    outer = GridSpec(2, 1, figure=fig, hspace=0.16)
    for slot, audio, tag, value in [
        (outer[0], raw, "before, no filter", before),
        (outer[1], filtered, f"after, {cutoff_hz:.0f} Hz high-pass", after),
    ]:
        host = fig.add_subplot(slot)
        plot_clip(
            clip, labels=labels, audio=audio, max_seconds=8.0, ax=host,
            title=f"{tag}   sub-80 Hz share of silence energy: {value * 100:.1f}%",
        )

    fig.suptitle(f"Low-frequency rumble, {clip.stem} (speaker {clip.speaker_id})",
                 fontsize=12.5, y=0.995)
    _save(fig, out_dir, "rumble_highpass.png", generated)
    return {"lf_in_silence_before": before, "lf_in_silence_after": after,
            "cutoff_hz": cutoff_hz}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--bridge-gap", type=float, default=0.2)
    parser.add_argument("--highpass-hz", type=float, default=80.0)
    args = parser.parse_args()

    out_dir = Path(os.path.expanduser(args.out))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    clips = load_dataset(args.directory)
    rows = [S.analyze_clip(c, bridge_gap_s=args.bridge_gap) for c in clips]

    durations = [r["duration_s"] for r in rows]
    total_frames = sum(r["n_frames"] for r in rows)
    overall = {
        "literal": sum(r["speech_frames_literal"] for r in rows) / total_frames,
        "bridged": sum(r["speech_frames_bridged"] for r in rows) / total_frames,
    }
    table = S.speaker_table(rows)
    split = S.propose_split(table)
    bridging = S.bridging_analysis(clips, rows, fps=100)
    elbow = S.gap_elbow([g for r in rows for g in r['gaps_s']])

    measured = [r for r in rows if r["snr_db"] is not None and r["snr_db_highpassed"] is not None]
    snr = [r["snr_db"] for r in measured]
    snr_hp = [r["snr_db_highpassed"] for r in measured]
    n_skipped = len(rows) - len(measured)
    snr_gate = S.propose_snr_gate(snr)

    lf_sil = [r["lf_fraction_silence"] for r in rows]
    rumble_gate = S.propose_rumble_gate(lf_sil)

    # What the high-pass actually buys, split by whether a clip rumbles.
    cut = rumble_gate["threshold_lf_fraction_in_silence"]
    gains = np.array(snr_hp) - np.array(snr)
    rumbly = np.array([r["lf_fraction_silence"] > cut for r in measured])
    hp_effect = {
        "cutoff_hz": args.highpass_hz,
        "median_snr_db_before": float(np.median(snr)),
        "median_snr_db_after": float(np.median(snr_hp)),
        "n_below_gate_before": int((np.array(snr) < snr_gate["threshold_db"]).sum()),
        "n_below_gate_after": int((np.array(snr_hp) < snr_gate["threshold_db"]).sum()),
        "mean_gain_db_high_rumble": float(gains[rumbly].mean()) if rumbly.any() else None,
        "mean_gain_db_low_rumble": float(gains[~rumbly].mean()) if (~rumbly).any() else None,
        "corr_rumble_vs_snr": float(np.corrcoef(
            [r["lf_fraction_silence"] for r in measured], snr)[0, 1]),
    }
    top_rumble = sorted(rows, key=lambda r: -r["lf_fraction_silence"])[:10]
    profile = S.position_profile(rows)

    lengths = [x for r in rows for x in r["segment_lengths_s"]]
    gaps = [g for r in rows for g in r["gaps_s"]]

    generated: list[str] = []
    fig_speakers(table, split, fig_dir, generated)
    fig_durations(rows, fig_dir, generated)
    fig_class_balance(rows, overall, fig_dir, generated)
    fig_segments_and_gaps(rows, bridging, elbow, fig_dir, generated)
    fig_snr(snr, snr_hp, snr_gate, n_skipped, args.highpass_hz, fig_dir, generated)
    fig_rumble(rows, rumble_gate, fig_dir, generated)
    fig_position(profile, fig_dir, generated)

    worst = next(c for c in clips if c.stem == top_rumble[0]["stem"])
    highpass = fig_highpass_demo(worst, args.highpass_hz, fig_dir, generated)

    payload = {
        "dataset_dir": os.path.expanduser(args.directory),
        "frame_grid": {"sample_rate": 16000, "hop_ms": 10, "win_ms": 25, "fps": 100},
        "corpus": {
            "n_clips": len(rows),
            "total_hours": sum(durations) / 3600.0,
            "total_frames": total_frames,
            "n_speakers": len(table["speakers"]),
            "n_chapters": len({(r["speaker_id"], r["chapter_id"]) for r in rows}),
        },
        "duration_s": S.describe(durations, "s"),
        "duration_histogram": S.histogram(durations, bins=40),
        "clips_per_speaker": S.describe(table["clips"].values(), "clips"),
        "minutes_per_speaker": S.describe(table["minutes"].values(), "min"),
        "speaker_table": {
            "clips": table["clips"],
            "minutes": table["minutes"],
            "chapters": table["chapters"],
        },
        "proposed_split": split,
        "class_balance": {
            "overall_speech_fraction_literal": overall["literal"],
            "overall_speech_fraction_bridged": overall["bridged"],
            "bridge_gap_s": args.bridge_gap,
            "imbalance_ratio_literal": overall["literal"] / (1 - overall["literal"]),
            "imbalance_ratio_bridged": overall["bridged"] / (1 - overall["bridged"]),
            "per_clip_literal": S.describe([r["speech_fraction_literal"] for r in rows]),
            "per_clip_bridged": S.describe([r["speech_fraction_bridged"] for r in rows]),
            "per_clip_literal_histogram": S.histogram(
                [r["speech_fraction_literal"] for r in rows], bins=40, range_=(0, 1)),
            "per_clip_bridged_histogram": S.histogram(
                [r["speech_fraction_bridged"] for r in rows], bins=40, range_=(0, 1)),
        },
        "segments": {
            "n_segments": len(lengths),
            "length_s": S.describe(lengths, "s"),
            "length_histogram": S.histogram(lengths, bins=45),
            "segments_per_clip": S.describe([r["n_segments"] for r in rows], "segments"),
        },
        "gaps": {
            "n_gaps": len(gaps),
            "note": "between consecutive speech segments only, leading and trailing silence excluded",
            "length_s": S.describe(gaps, "s"),
            "length_histogram": S.histogram(gaps, bins=45),
            "bridging_candidates": bridging,
            "elbow": elbow,
        },
        "snr": {
            "n_measured": len(snr),
            "n_skipped_no_silence": n_skipped,
            "db": S.describe(snr, "dB"),
            "histogram": S.histogram(snr, bins=45),
            "proposed_gate": snr_gate,
            "db_after_highpass": S.describe(snr_hp, "dB"),
            "histogram_after_highpass": S.histogram(snr_hp, bins=45),
            "highpass_effect": hp_effect,
            "gate_rationale": (
                "Clips at or above the gate are treated as clean and are eligible for "
                "RIR and additive-noise augmentation. Clips below it are already "
                "degraded and are used raw. Convolving a degraded clip with an RIR "
                "reverberates its existing noise floor along with the speech, so the "
                "result is not clean speech placed in a known room and the "
                "controlled-degradation assumption no longer holds."
            ),
        },
        "rumble": {
            "band_hz": [0.0, S.RUMBLE_HZ],
            "lf_fraction_all": S.describe([r["lf_fraction_all"] for r in rows]),
            "lf_fraction_silence": S.describe(lf_sil),
            "lf_fraction_all_histogram": S.histogram([r["lf_fraction_all"] for r in rows], bins=45),
            "lf_fraction_silence_histogram": S.histogram(lf_sil, bins=45),
            "proposed_gate": rumble_gate,
            "top_clips_by_lf_in_silence": [
                {"stem": r["stem"], "speaker_id": r["speaker_id"],
                 "lf_fraction_silence": r["lf_fraction_silence"],
                 "lf_fraction_all": r["lf_fraction_all"],
                 "snr_db": r["snr_db"]}
                for r in top_rumble
            ],
            "highpass_demo": highpass,
        },
        "position_profile": profile,
    }

    json_path = out_dir / "corpus_stats.json"
    json_path.write_text(json.dumps(payload, indent=2))
    generated.append(str(json_path))

    _print_summary(payload, table, split, bridging, elbow, snr_gate, rumble_gate,
                   profile, top_rumble, highpass)

    for path in sorted(generated):
        print(path)
    return 0


def _print_summary(p, table, split, bridging, elbow, snr_gate, rumble_gate, profile, top_rumble, highpass):
    c, d = p["corpus"], p["duration_s"]
    print("\ncorpus")
    print(f"  clips {c['n_clips']}   hours {c['total_hours']:.2f}   "
          f"speakers {c['n_speakers']}   chapters {c['n_chapters']}")
    print(f"  duration  min {d['min']:.2f}s  median {d['median']:.2f}s  "
          f"max {d['max']:.2f}s  mean {d['mean']:.2f}s")
    cps, mps = p["clips_per_speaker"], p["minutes_per_speaker"]
    print(f"  clips/speaker   min {cps['min']:.0f}  median {cps['median']:.0f}  max {cps['max']:.0f}")
    print(f"  minutes/speaker min {mps['min']:.1f}  median {mps['median']:.1f}  max {mps['max']:.1f}")

    print(f"\n{'speaker':>9} {'clips':>6} {'minutes':>8} {'chapters':>9}")
    for s in table["speakers"]:
        print(f"{s:>9} {table['clips'][s]:6d} {table['minutes'][s]:8.1f} {table['chapters'][s]:9d}")

    print("\nproposed speaker-disjoint split")
    for part in ("train", "val", "test"):
        v = split["partitions"][part]
        print(f"  {part:<5} {v['n_speakers']:2d} speakers  {v['n_clips']:4d} clips "
              f"({v['clip_fraction']*100:4.1f}%)  {v['hours']:.2f} h")
        print(f"        {', '.join(v['speakers'])}")

    cb = p["class_balance"]
    print("\nclass balance")
    print(f"  literal  speech {cb['overall_speech_fraction_literal']*100:.1f}% of all frames "
          f"({cb['imbalance_ratio_literal']:.2f} : 1 speech to non-speech)")
    print(f"  bridged  speech {cb['overall_speech_fraction_bridged']*100:.1f}% of all frames "
          f"({cb['imbalance_ratio_bridged']:.2f} : 1)")

    seg, gap = p["segments"], p["gaps"]
    print("\nsegments and gaps")
    print(f"  segments {seg['n_segments']}  median length {seg['length_s']['median']*1000:.0f} ms  "
          f"mean {seg['segments_per_clip']['mean']:.1f} per clip")
    print(f"  gaps     {gap['n_gaps']}  median {gap['length_s']['median']*1000:.0f} ms  "
          f"p95 {gap['length_s']['p95']*1000:.0f} ms")
    print(f"  elbow {elbow['elbow_ms']:.0f} ms, plateau "
          f"{elbow['plateau_count_per_bucket']:.0f} gaps per 10 ms bucket")
    print(f"\n  {'threshold':>10} {'gaps bridged':>14} {'seg/clip':>10} {'speech frac':>12} {'delta':>8}")
    for e in bridging:
        print(f"  {e['threshold_s']*1000:8.0f}ms {e['gap_fraction_bridged']*100:13.1f}% "
              f"{e['mean_segments_per_clip']:10.1f} {e['speech_fraction']*100:11.1f}% "
              f"{e['speech_fraction_delta']*100:+7.1f}%")

    sn = p["snr"]
    print("\nsnr")
    print(f"  measured on {sn['n_measured']} clips, {sn['n_skipped_no_silence']} skipped (no silence frames)")
    print(f"  min {sn['db']['min']:.1f}  p05 {sn['db']['p05']:.1f}  median {sn['db']['median']:.1f}  "
          f"p95 {sn['db']['p95']:.1f}  max {sn['db']['max']:.1f} dB")
    hp = sn["highpass_effect"]
    print(f"\n  effect of the {hp['cutoff_hz']:.0f} Hz high-pass on SNR")
    print(f"    median {hp['median_snr_db_before']:.1f} -> {hp['median_snr_db_after']:.1f} dB")
    print(f"    clips below the {snr_gate['threshold_db']:.0f} dB gate: "
          f"{hp['n_below_gate_before']} -> {hp['n_below_gate_after']}")
    print(f"    mean gain: high-rumble clips {hp['mean_gain_db_high_rumble']:+.1f} dB, "
          f"the rest {hp['mean_gain_db_low_rumble']:+.1f} dB")
    print(f"    corr(sub-80 Hz share in silence, SNR) = {hp['corr_rumble_vs_snr']:.2f}")

    ru = p["rumble"]
    print(f"\nlow-frequency rumble, below {S.RUMBLE_HZ:.0f} Hz")
    print(f"  whole clip      median {ru['lf_fraction_all']['median']:.4f}  max {ru['lf_fraction_all']['max']:.4f}")
    print(f"  labeled silence median {ru['lf_fraction_silence']['median']:.4f}  "
          f"max {ru['lf_fraction_silence']['max']:.4f}")
    print(f"\n  top clips by sub-80 Hz energy share in silence")
    for r in top_rumble[:6]:
        print(f"    {r['stem']:20s} speaker {r['speaker_id']:>5}  "
              f"silence {r['lf_fraction_silence']:.3f}  whole {r['lf_fraction_all']:.3f}")

    print("\nposition")
    print(f"  P(speech)  first decile {profile['first_decile_mean']*100:.1f}%   "
          f"middle {profile['middle_mean']*100:.1f}%   "
          f"last decile {profile['last_decile_mean']*100:.1f}%")

    print("\nproposed thresholds")
    # smallest candidate at or above the elbow: past it, extra bridging only
    # deletes genuine pauses at the plateau rate for no artifact benefit
    above = [e for e in bridging if e["threshold_s"] * 1000 >= elbow["elbow_ms"]]
    chosen = min(above, key=lambda e: e["threshold_s"]) if above else bridging[-1]
    print(f"  1. bridging gap   {chosen['threshold_s']*1000:.0f} ms   "
          f"closes {chosen['gap_fraction_bridged']*100:.1f}% of gaps, "
          f"speech {chosen['speech_fraction']*100:.1f}%, "
          f"{chosen['mean_segments_per_clip']:.1f} seg/clip")
    print(f"  2. SNR gate       {snr_gate['threshold_db']:.0f} dB   "
          f"{snr_gate['n_clean']} clean (augmentable) / {snr_gate['n_degraded']} degraded (use raw)")
    print(f"  3. rumble gate    {rumble_gate['threshold_lf_fraction_in_silence']:.2f} "
          f"sub-80 Hz share in silence   {rumble_gate['n_above']} clips above")
    print(f"     {highpass['cutoff_hz']:.0f} Hz high-pass takes the worst clip from "
          f"{highpass['lf_in_silence_before']*100:.1f}% to "
          f"{highpass['lf_in_silence_after']*100:.1f}% sub-80 Hz energy in silence")


if __name__ == "__main__":
    raise SystemExit(main())
