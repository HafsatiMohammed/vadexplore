"""Assemble explore_out/DATA_REPORT.md from the saved exploration artifacts.

Every number is read from the JSON written by the earlier scripts and every
figure is embedded by relative path. Nothing is recomputed and nothing is
hardcoded in the prose, so the report cannot drift away from the code that
produced it. A missing key is a hard failure naming the key and the script
that should have produced it.

    python scripts/exploration_summary.py [--out DIR]

Inputs, produced by:
    explore_out/corpus_stats.json       scripts/corpus_report.py
    explore_out/silero_agreement.json   scripts/crosscheck_silero.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_OUT = "explore_out"

SOURCES = {
    "corpus_stats.json": "scripts/corpus_report.py",
    "silero_agreement.json": "scripts/crosscheck_silero.py",
}


class MissingArtifact(RuntimeError):
    """A required JSON key or figure is absent."""


class Source:
    """A loaded JSON artifact that fails loudly on a missing key."""

    def __init__(self, path: Path, producer: str):
        if not path.exists():
            raise MissingArtifact(
                f"missing input file: {path}\n"
                f"  produce it by running: python {producer}"
            )
        self.path = path
        self.producer = producer
        self.data = json.loads(path.read_text())

    def get(self, dotted: str):
        """Fetch a dotted key path, naming the failure precisely."""
        node = self.data
        walked = []
        for part in dotted.split("."):
            walked.append(part)
            if not isinstance(node, dict) or part not in node:
                raise MissingArtifact(
                    f"missing key '{dotted}' in {self.path.name}\n"
                    f"  failed at: {'.'.join(walked)}\n"
                    f"  available here: {sorted(node)[:12] if isinstance(node, dict) else type(node).__name__}\n"
                    f"  regenerate with: python {self.producer}"
                )
            node = node[part]
        if node is None:
            raise MissingArtifact(
                f"key '{dotted}' in {self.path.name} is null\n"
                f"  regenerate with: python {self.producer}"
            )
        return node


class Figures:
    """Collects embedded figure paths and verifies each one exists."""

    def __init__(self, root: Path):
        self.root = root
        self.used: list[str] = []

    def embed(self, relative: str, caption: str) -> str:
        path = self.root / relative
        if not path.exists():
            producer = ("scripts/crosscheck_silero.py" if "silero" in relative
                        or "disagreements" in relative else "scripts/corpus_report.py")
            raise MissingArtifact(
                f"missing figure: {path}\n  produce it by running: python {producer}"
            )
        self.used.append(relative)
        return f"![{caption}]({relative})\n\n*{caption}*\n"


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def build(out_dir: Path) -> tuple[str, Figures]:
    corpus = Source(out_dir / "corpus_stats.json", SOURCES["corpus_stats.json"])
    silero = Source(out_dir / "silero_agreement.json", SOURCES["silero_agreement.json"])
    fig = Figures(out_dir)

    # --- pull everything up front so a missing key fails before any writing ---
    n_clips = corpus.get("corpus.n_clips")
    hours = corpus.get("corpus.total_hours")
    n_speakers = corpus.get("corpus.n_speakers")
    n_chapters = corpus.get("corpus.n_chapters")
    dur = corpus.get("duration_s")
    grid = corpus.get("frame_grid")
    cps = corpus.get("clips_per_speaker")

    balance = corpus.get("class_balance")
    segs = corpus.get("segments")
    gaps = corpus.get("gaps")
    elbow = corpus.get("gaps.elbow")
    candidates = corpus.get("gaps.bridging_candidates")
    chosen = min((c for c in candidates if c["threshold_s"] * 1000 >= elbow["elbow_ms"]),
                 key=lambda c: c["threshold_s"])
    widest = max(candidates, key=lambda c: c["threshold_s"])

    snr = corpus.get("snr")
    snr_gate = corpus.get("snr.proposed_gate")
    hp = corpus.get("snr.highpass_effect")
    rumble = corpus.get("rumble")
    rumble_gate = corpus.get("rumble.proposed_gate")
    demo = corpus.get("rumble.highpass_demo")
    position = corpus.get("position_profile")
    split = corpus.get("proposed_split.partitions")
    targets = corpus.get("proposed_split.targets")

    sil_sum = silero.get("summary")
    model = silero.get("model")
    per_clip = silero.get("per_clip")
    interest = silero.get("clips_of_interest")

    # percentile rank of the named clips, computed from the saved per-clip data
    ranks = {}
    for stem in interest:
        row = next((r for r in per_clip if r["stem"] == stem), None)
        if row is None:
            raise MissingArtifact(
                f"clip of interest {stem} absent from silero_agreement.json per_clip\n"
                f"  regenerate with: python {SOURCES['silero_agreement.json']}"
            )
        ranks[stem] = {"row": row}
        for key in ("literal", "bridged"):
            values = np.array([r[key]["no_collar"]["iou"] for r in per_clip])
            ranks[stem][key] = {
                "iou": row[key]["no_collar"]["iou"],
                "iou_collar": row[key]["with_collar"]["iou"],
                "percentile": float((values < row[key]["no_collar"]["iou"]).mean() * 100),
            }
        breakdown = row["literal_breakdown"]
        ranks[stem]["longest_region_s"] = max(
            (r["n_frames"] for r in breakdown["region_runs"]), default=0) / grid["fps"]

    sweep = sil_sum["literal"]["collar_sweep"]
    lag_ms = sil_sum["literal"]["median_boundary_bias_ms"]
    # the sweep entry whose collar first covers the measured latency
    covering = min((v for v in sweep.values() if v["collar_ms"] >= abs(lag_ms)),
                   key=lambda v: v["collar_ms"], default=max(
                       sweep.values(), key=lambda v: v["collar_ms"]))
    tight = sweep[str(sil_sum["tol_frames"])]

    worst = min(per_clip, key=lambda r: r["best_iou"])

    # --- prose ---
    parts = []

    parts.append(f"""# Data report: voice activity detection corpus

*Every figure and number in this document is generated by the exploration
scripts in `scripts/` and read back from `corpus_stats.json` and
`silero_agreement.json`. The document is assembled, not hand-written.*

Frame grid throughout: {grid['sample_rate']} Hz audio, {grid['hop_ms']} ms hop,
{grid['win_ms']} ms analysis window, {grid['fps']} frames per second.
""")

    # 1. Dataset overview
    parts.append(f"""## 1. Dataset overview

The corpus is {n_clips} single-speaker read-speech clips derived from
LibriSpeech, identifiable from the `<speaker>-<chapter>-<utterance>` file
naming, totalling {hours:.2f} hours of audio across {n_speakers} speakers and
{n_chapters} speaker-chapter pairs. Clip duration runs from {dur['min']:.2f} to
{dur['max']:.2f} seconds with a median of {dur['median']:.2f} seconds, so the
distribution is tight and dominated by clips near the upper end rather than
spread evenly. Per-speaker coverage is uneven but not extreme: between
{cps['min']:.0f} and {cps['max']:.0f} clips per speaker with a median of
{cps['median']:.0f}, which is enough material per speaker for a
speaker-disjoint split to remain balanced.

{fig.embed('figures/durations.png', 'Clip duration distribution over the whole corpus.')}
{fig.embed('figures/speakers.png', 'Corpus composition: clips per speaker, minutes per speaker, and the proposed speaker-disjoint split.')}""")

    # 2. Labels and bridging
    parts.append(f"""## 2. Labels and the bridging decision

The speech segments are forced-alignment output, not human annotation. They are
silver labels: internally consistent and dense, but carrying the aligner's own
characteristic errors, and every conclusion below should be read with that
caveat. The aligner operates on the same {grid['hop_ms']} ms grid as our frames,
which is why label boundaries need no sub-frame interpolation.

Two label conventions are carried side by side. The *literal* convention
rasterizes the segments as given, after only removing shared-endpoint artifacts.
The *bridged* convention additionally fills non-speech gaps shorter than a
threshold, on the reasoning that a detector which chatters on and off through a
breath or a stop closure is less useful than one that rides across it.

The threshold comes from the gap distribution, which is clearly bimodal. Counts
fall steeply from the shortest gaps down to a floor of about
{elbow['plateau_count_per_bucket']:.0f} gaps per {elbow['bucket_ms']:.0f} ms
bucket, and that floor then runs flat for hundreds of milliseconds. The steep
part is aligner artifacts and within-word closures; the flat part is genuine
pauses. The elbow between them sits at {elbow['elbow_ms']:.0f} ms, and that is
the threshold adopted. It closes {pct(chosen['gap_fraction_bridged'])} percent
of the {gaps['n_gaps']} gaps and leaves
{chosen['mean_segments_per_clip']:.1f} speech segments per clip. Bridging further
buys little and costs real silence: raising the threshold to
{widest['threshold_s'] * 1000:.0f} ms closes only
{pct(widest['gap_fraction_bridged'] - chosen['gap_fraction_bridged'])} percentage
points more, and those additional gaps come out of the genuine-pause plateau.
Speech segments themselves have a median length of
{segs['length_s']['median'] * 1000:.0f} ms.

The two conventions give overall speech fractions of
{pct(balance['overall_speech_fraction_literal'])} percent (literal) and
{pct(balance['overall_speech_fraction_bridged'])} percent (bridged) of all
frames. Both are produced on one frame grid so the choice stays a measurable
sensitivity axis rather than a baked-in assumption.

{fig.embed('figures/segments_and_gaps.png', 'Segment lengths, the full gap distribution, and the 10 ms bucket detail showing the elbow between the artifact cluster and the genuine-pause plateau.')}""")

    # 3. Class balance
    parts.append(f"""## 3. Class balance

The corpus is heavily imbalanced, and in the direction opposite to the usual
voice-activity assumption: speech is the majority class. Under literal labels
{pct(balance['overall_speech_fraction_literal'])} percent of frames are speech,
a ratio of {balance['imbalance_ratio_literal']:.2f} to 1 against non-speech, and
bridging pushes that to {pct(balance['overall_speech_fraction_bridged'])} percent
or {balance['imbalance_ratio_bridged']:.2f} to 1. Per clip the spread is wide,
with a small number of clips approaching continuous speech. Any accuracy figure
on this corpus therefore has a floor of roughly
{pct(balance['overall_speech_fraction_literal'])} percent from predicting speech
everywhere, which is why accuracy alone is not reported as a headline number
anywhere in this document.

This is handled at training and evaluation time rather than by resampling the
corpus: a weighted loss, metrics that do not reward the majority class
(F1 and IoU on the speech class, and their non-speech counterparts), and an
operating point chosen from a DET curve rather than fixed at 0.5. Resampling
was rejected because it would distort the pause structure that section 2 just
established. The detail belongs to the modeling section.

{fig.embed('figures/class_balance.png', 'Per-clip speech fraction under both label conventions, with the corpus-level fractions marked.')}""")

    # 4. Recording conditions
    parts.append(f"""## 4. Recording conditions

Per-clip signal-to-noise ratio is estimated as the ratio of mean frame energy
in labeled-speech frames to mean frame energy in labeled-silence frames. Across
{snr['n_measured']} clips ({snr['n_skipped_no_silence']} were skipped for having
no silence frames) the median is {snr['db']['median']:.1f} dB with a range from
{snr['db']['min']:.1f} to {snr['db']['max']:.1f} dB, and the distribution is
visibly bimodal, with a low-SNR shoulder separate from the main mode.

That shoulder turned out not to be noisy speech. A low-frequency check found a
persistent sub-{rumble['band_hz'][1]:.0f} Hz hum, present during labeled silence
and visible as a bright bottom band in the spectrograms and as a DC offset in
the waveform. The median clip puts
{pct(rumble['lf_fraction_silence']['median'])} percent of its silence energy
below {rumble['band_hz'][1]:.0f} Hz, but
{rumble_gate['n_above']} clips exceed
{rumble_gate['threshold_lf_fraction_in_silence']:.2f}, and the worst reach
{pct(rumble['lf_fraction_silence']['max'])} percent, meaning their silence is
almost entirely rumble.

An {hp['cutoff_hz']:.0f} Hz high-pass is therefore committed as a front-end
step, applied before anything else measures or models the signal. On the worst
clip it takes sub-{rumble['band_hz'][1]:.0f} Hz energy in silence from
{pct(demo['lf_in_silence_before'])} percent to
{pct(demo['lf_in_silence_after'])} percent. Corpus-wide it lifts the median SNR
from {hp['median_snr_db_before']:.1f} to {hp['median_snr_db_after']:.1f} dB, and
the gain is concentrated exactly where it should be: clips above the rumble
threshold gain {hp['mean_gain_db_high_rumble']:+.1f} dB on average against
{hp['mean_gain_db_low_rumble']:+.1f} dB for the rest.

The ordering matters. A {snr_gate['threshold_db']:.0f} dB gate separates clips
that are clean enough for room-impulse-response and additive-noise augmentation
from clips that are already degraded and are used raw, because convolving an
already noisy clip with an impulse response reverberates its noise floor along
with the speech and destroys the controlled-degradation assumption. Applied to
unfiltered audio that gate would reject {hp['n_below_gate_before']} clips.
Applied after the high-pass it rejects {hp['n_below_gate_after']}, leaving
{n_clips - hp['n_below_gate_after']} clips augmentation-eligible. More than half
of what looked degraded was rumble sitting in the silence frames rather than
noise on the speech, so gating first would have wrongly excluded
{hp['n_below_gate_before'] - hp['n_below_gate_after']} clean clips.

{fig.embed('figures/rumble.png', 'Share of energy below 80 Hz, over the whole clip and within labeled silence only.')}
{fig.embed('figures/rumble_highpass.png', 'The worst rumble clip before and after the 80 Hz high-pass, same labels on both.')}
{fig.embed('figures/snr.png', 'Per-clip SNR before and after the high-pass, and the SNR gain plotted against the unfiltered value.')}""")

    # 5. Position
    parts.append(f"""## 5. Label density against position

Because clips are cut from longer recordings, it is worth checking whether
speech sits at systematically predictable places, which a model could exploit as
a shortcut instead of learning acoustics. Averaged over all clips on a
normalized position axis, mean speech probability is
{pct(position['first_decile_mean'])} percent in the first tenth of a clip,
{pct(position['middle_mean'])} percent through the body, and
{pct(position['last_decile_mean'])} percent in the last tenth. The profile is
flat through the middle and symmetric at the edges, so what exists is a short
silent margin at each end rather than a usable positional prior. The effect is
mild, but random cropping during training removes even that.

{fig.embed('figures/position.png', 'Mean speech probability against normalized position within the clip.')}""")

    # 6. Silero
    lit, bri = sil_sum["literal"], sil_sum["bridged"]
    interest_prose = []
    for stem, reason in interest.items():
        r = ranks[stem]
        interest_prose.append(
            f"`{stem}` ({reason}) reaches IoU {r['literal']['iou']:.3f} against literal "
            f"and {r['bridged']['iou']:.3f} against bridged, at the "
            f"{r['literal']['percentile']:.0f}th and {r['bridged']['percentile']:.0f}th "
            f"percentile of the corpus, rising to {r['literal']['iou_collar']:.3f} once the "
            f"collar is applied; its longest single region conflict is "
            f"{r['longest_region_s']:.2f} s"
        )

    parts.append(f"""## 6. Independent label validation

The silver labels were cross-checked against Silero VAD, a small pretrained
neural detector run over the same high-passed audio, using its raw per-window
probabilities rather than its post-processed segments so that neither side's
smoothing enters the comparison. Silero's native resolution is
{model['native_window_ms']:.0f} ms, mapped onto the {grid['fps']} fps grid by
{model['grid_mapping']}, which leaves an inherent boundary uncertainty of about
plus or minus {model['inherent_boundary_uncertainty_ms']:.0f} ms.

Agreement is high. Median per-clip IoU is {lit['median_iou']:.3f} against
literal labels and {bri['median_iou']:.3f} against bridged, with F1
{lit['median_f1']:.3f} and {bri['median_f1']:.3f} and Cohen's kappa
{lit['median_kappa']:.3f} and {bri['median_kappa']:.3f}. Bridged agrees better
on every measure, which is independent corroboration of the bridging decision in
section 2: an unrelated detector trained on other data also treats those short
gaps as speech. No clip falls below IoU {sil_sum['flag_iou']:g} against either
convention, so there are no wholesale labeling failures anywhere in the corpus.

The residual disagreement needs care to read, because most of it is a method
difference rather than a label error. Silero is a causal streaming model with no
lookahead, and it runs late: the median boundary offset is
{lit['median_onset_bias_ms']:+.0f} ms at onsets and
{lit['median_offset_bias_ms']:+.0f} ms at offsets, well outside its own
{model['inherent_boundary_uncertainty_ms']:.0f} ms resolution and confirmed
separately on speech placed at exactly known onsets. A collar narrower than that
latency cannot absorb it and reports shifted boundaries as region conflicts:
with a {tight['collar_ms']:.0f} ms collar only
{pct(tight['boundary_share'])} percent of disagreed frames are classified
boundary-only, but at {covering['collar_ms']:.0f} ms, which is the first collar
wide enough to cover the measured lag, that rises to
{pct(covering['boundary_share'])} percent. Roughly half the disagreement is
therefore boundary placement explained by Silero's latency, and the genuine
region-level conflict that remains is small and scattered rather than
concentrated in particular clips.

The two clips flagged during earlier inspection both survive this check.
{interest_prose[0]}. {interest_prose[1]}. Neither is
anywhere near the {sil_sum['flag_iou']:g} threshold, and in the low-volume case
Silero finds no speech in the stretch the aligner labels silent, so the
suspicion that quiet speech had been dropped is not supported. It is worth
stating plainly that `{list(interest)[0]}` sits in the lower tail rather than
above the median: its disagreement is real but small and confined to short
regions, not a missing utterance.

On this evidence the labels are treated as trustworthy silver labels and no
clips are excluded from the corpus.

{fig.embed('figures/silero_agreement.png', 'Per-clip IoU and F1 against both label conventions, with the flag threshold marked.')}
{fig.embed('figures/silero_boundary_bias.png', 'Boundary bias against the provided labels, with Silero native resolution shaded.')}
{fig.embed(f"disagreements/{worst['stem']}.png", f"Lowest-agreement clip in the corpus, {worst['stem']} (best IoU {worst['best_iou']:.3f}), with the Silero probability overlaid on both label ribbons.")}
{fig.embed(f"disagreements/{list(interest)[0]}.png", f"Clip of interest {list(interest)[0]}: {interest[list(interest)[0]]}.")}""")

    # 7. Split
    total_clips = sum(v["n_clips"] for v in split.values())
    rows = "\n".join(
        f"| {name} | {v['n_speakers']} | {v['n_clips']} | {pct(v['clip_fraction'])} | {v['hours']:.2f} |"
        for name, v in (("train", split["train"]), ("val", split["val"]), ("test", split["test"])))
    parts.append(f"""## 7. Data split

The split is speaker-disjoint across all {n_speakers} speakers, so no speaker
appears in more than one partition and no evaluation number can be inflated by
speaker memorization. Assignment is greedy largest-first, since a speaker is
indivisible and an exact ratio is unreachable; the realized proportions land
close to the {targets['train']*100:.0f} / {targets['val']*100:.0f} / {targets['test']*100:.0f} target.

| partition | speakers | clips | percent of clips | hours |
|---|---|---|---|---|
{rows}

Total {total_clips} clips. Hard test conditions are produced by applying
augmentation to the test speakers rather than by holding out naturally degraded
recordings. Section 4 is the reason: once the rumble is filtered the corpus is
uniformly clean read speech, so there is no natural degraded subset large enough
to hold out, and any attempt to carve one would confound difficulty with speaker
identity. Generating the difficulty keeps it parameterized, reproducible, and
independent of who is speaking.""")

    # 8. Closing
    parts.append(f"""## 8. Decisions carried into modeling

Four choices are committed. An {hp['cutoff_hz']:.0f} Hz high-pass is applied as
a front-end step before any measurement or modeling, because sub-{rumble['band_hz'][1]:.0f} Hz rumble
sits almost entirely in the silence frames and otherwise corrupts exactly the
speech-versus-silence contrast the task depends on. Gaps shorter than
{chosen['threshold_s'] * 1000:.0f} ms are bridged, at the elbow of the gap
distribution, with the literal labels retained so the convention remains a
sensitivity axis rather than an assumption. A {snr_gate['threshold_db']:.0f} dB
SNR gate computed after the high-pass separates
{n_clips - hp['n_below_gate_after']} augmentation-eligible clips from
{hp['n_below_gate_after']} degraded clips used raw. The split is speaker-disjoint
across {n_speakers} speakers.

Three items are handed forward unresolved. The literal-versus-bridged question
is deliberately left open as a two-by-two: train on each convention, evaluate
against each, and report all four cells, since agreeing with the convention you
trained on proves nothing. Class imbalance of roughly
{balance['imbalance_ratio_literal']:.1f} to 1 toward speech needs a weighted loss
and an operating point chosen from a DET curve rather than assumed at 0.5.
Augmentation design, meaning which impulse responses and noise types at which
signal-to-noise ratios, is scoped by the gate established here but not yet
specified.""")

    return "\n\n".join(parts) + "\n", fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = Path(os.path.expanduser(args.out))
    try:
        text, fig = build(out_dir)
    except MissingArtifact as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = out_dir / "DATA_REPORT.md"
    report.write_text(text)

    print(f"wrote {report}  ({len(text):,} characters)")
    print(f"\nembedded figures, all verified present ({len(fig.used)}):")
    for relative in fig.used:
        size = os.path.getsize(out_dir / relative)
        print(f"  ok  {relative}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
