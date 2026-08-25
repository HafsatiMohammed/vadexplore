"""Corpus-level analysis for the VAD dataset.

Pure computation. Everything returns plain Python or numpy so it can be
serialized to JSON or handed to a plotting layer. No matplotlib here, so the
numbers can be recomputed in a training job without a display stack.

Reuses the loader, the label geometry in `vadexplore.labels`, and the feature
functions in `vadexplore.features` rather than reframing audio locally.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from vadexplore.features import (
    DEFAULT_SR,
    HIGHPASS_HZ,
    band_fraction,
    frame_energy,
    highpass,
    power_spectrogram,
)
from vadexplore.labels import (
    DEFAULT_FPS,
    bridge_segments,
    make_labels,
    normalize_segments,
    rasterize,
)
from vadexplore.loader import read_audio

BRIDGE_CANDIDATES_S = (0.10, 0.15, 0.20, 0.30)
RUMBLE_HZ = 80.0
POSITION_BINS = 100


# --- small helpers --------------------------------------------------------


def describe(values, unit: str = "") -> dict:
    """Standard summary of a 1D sample, JSON ready."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "unit": unit}
    return {
        "n": int(arr.size),
        "unit": unit,
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "total": float(arr.sum()),
    }


def histogram(values, bins=30, range_=None) -> dict:
    """Histogram reduced to counts and edges, so it survives a JSON round trip."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(arr, bins=bins, range=range_)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def gaps_between(segments) -> list[float]:
    """Non-speech gaps strictly between consecutive speech segments.

    Leading silence before the first segment and trailing silence after the
    last are excluded on purpose: they are not candidates for bridging, and
    including them would put a long tail into the histogram that no threshold
    would ever act on.
    """
    normalized = normalize_segments(segments)
    return [normalized[i + 1][0] - normalized[i][1] for i in range(len(normalized) - 1)]


# --- per-clip pass --------------------------------------------------------


def _snr_db(energy: np.ndarray, speech: np.ndarray, silence: np.ndarray):
    """Speech to silence energy ratio in dB, or None if undefined."""
    if not (speech.any() and silence.any()):
        return None
    speech_e, silence_e = float(energy[speech].mean()), float(energy[silence].mean())
    if speech_e <= 0 or silence_e <= 0:
        return None
    return float(10.0 * np.log10(speech_e / silence_e))


def analyze_clip(clip, fps: int = DEFAULT_FPS, bridge_gap_s: float = 0.2,
                 highpass_hz: float = HIGHPASS_HZ) -> dict:
    """Everything measurable from one clip, labels and audio together.

    SNR and the rumble measures need labeled frames to split speech from
    silence, so they are computed here where both are already in hand.
    """
    labels = make_labels(clip, fps=fps, bridge_gap_s=bridge_gap_s)
    n_frames = labels["n_frames"]
    literal = labels["literal"].astype(bool)
    speech = literal
    silence = ~literal

    segments = labels["segments_literal"]
    row = {
        "stem": clip.stem,
        "speaker_id": clip.speaker_id,
        "chapter_id": clip.chapter_id,
        "duration_s": float(clip.duration_s),
        "n_frames": int(n_frames),
        "n_segments": len(segments),
        "segment_lengths_s": [float(e - s) for s, e in segments],
        "gaps_s": [float(g) for g in gaps_between(segments)],
        "speech_frames_literal": int(literal.sum()),
        "speech_frames_bridged": int(labels["bridged"].sum()),
        "speech_fraction_literal": float(literal.mean()) if n_frames else 0.0,
        "speech_fraction_bridged": float(labels["bridged"].mean()) if n_frames else 0.0,
    }

    audio = read_audio(clip, target_sr=DEFAULT_SR)
    power, freqs = power_spectrogram(audio, sr=DEFAULT_SR, n_frames=n_frames)
    energy = frame_energy(power)
    low = band_fraction(power, freqs, 0.0, RUMBLE_HZ)

    # SNR needs both classes present, otherwise the ratio is meaningless
    row["snr_db"] = _snr_db(energy, speech, silence)

    # The same SNR after the candidate high-pass. Sub-80 Hz rumble sits almost
    # entirely in the silence frames, so it depresses the denominator and makes
    # a clean clip look noisy. Measuring both tells us whether a clip is really
    # degraded or merely rumbling, which is what the augmentation gate needs.
    hp_power, _ = power_spectrogram(highpass(audio, DEFAULT_SR, highpass_hz),
                                    sr=DEFAULT_SR, n_frames=n_frames)
    hp_energy = frame_energy(hp_power)
    row["snr_db_highpassed"] = _snr_db(hp_energy, speech, silence)
    row["highpass_hz"] = float(highpass_hz)

    # Energy-weighted band share, not a mean of per-frame ratios: loud frames
    # should dominate a statement about where the clip's energy sits.
    total_e = float(energy.sum())
    row["lf_fraction_all"] = float((low * energy).sum() / total_e) if total_e > 0 else 0.0
    silence_e_sum = float(energy[silence].sum()) if silence.any() else 0.0
    row["lf_fraction_silence"] = (
        float((low[silence] * energy[silence]).sum() / silence_e_sum) if silence_e_sum > 0 else 0.0
    )
    row["silence_frames"] = int(silence.sum())

    # speech probability against normalized position, one row per clip
    row["position_curve"] = _position_curve(literal, POSITION_BINS).tolist()

    return row


def _position_curve(literal: np.ndarray, n_bins: int) -> np.ndarray:
    """Resample a clip's frame labels onto a fixed 0 to 1 position grid.

    Every clip contributes the same number of points regardless of length, so
    the corpus average is not dominated by the long clips.
    """
    if literal.size == 0:
        return np.zeros(n_bins)
    src = (np.arange(literal.size) + 0.5) / literal.size
    dst = (np.arange(n_bins) + 0.5) / n_bins
    return np.interp(dst, src, literal.astype(np.float64))


def analyze_corpus(clips, fps: int = DEFAULT_FPS, bridge_gap_s: float = 0.2,
                   highpass_hz: float = HIGHPASS_HZ) -> list[dict]:
    """Run `analyze_clip` over every clip."""
    return [analyze_clip(c, fps=fps, bridge_gap_s=bridge_gap_s,
                         highpass_hz=highpass_hz) for c in clips]


# --- aggregation ----------------------------------------------------------


def speaker_table(rows) -> dict:
    """Clips and minutes per speaker, sorted by clip count descending."""
    clips_per = Counter()
    seconds_per = defaultdict(float)
    chapters_per = defaultdict(set)
    for row in rows:
        clips_per[row["speaker_id"]] += 1
        seconds_per[row["speaker_id"]] += row["duration_s"]
        chapters_per[row["speaker_id"]].add(row["chapter_id"])

    order = sorted(clips_per, key=lambda s: (-clips_per[s], s))
    return {
        "speakers": order,
        "clips": {s: clips_per[s] for s in order},
        "minutes": {s: seconds_per[s] / 60.0 for s in order},
        "chapters": {s: len(chapters_per[s]) for s in order},
    }


def propose_split(table: dict, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Speaker-disjoint train/val/test proposal.

    Greedy largest-first assignment: walk speakers from most clips to fewest
    and give each to whichever partition is furthest below its target share.
    Speaker-disjoint splits cannot hit an exact ratio, since a speaker is
    indivisible, and largest-first keeps the error bounded by the largest
    remaining speaker rather than letting it accumulate.
    """
    targets = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    total_clips = sum(table["clips"].values())

    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for speaker in table["speakers"]:  # already largest first
        deficit = {p: targets[p] - counts[p] / max(total_clips, 1) for p in targets}
        pick = max(deficit, key=lambda p: deficit[p])
        assigned[pick].append(speaker)
        counts[pick] += table["clips"][speaker]

    out = {"targets": targets, "partitions": {}}
    for part, speakers in assigned.items():
        clips = sum(table["clips"][s] for s in speakers)
        minutes = sum(table["minutes"][s] for s in speakers)
        out["partitions"][part] = {
            "speakers": sorted(speakers),
            "n_speakers": len(speakers),
            "n_clips": clips,
            "clip_fraction": clips / total_clips if total_clips else 0.0,
            "hours": minutes / 60.0,
        }
    return out


def bridging_analysis(clips, rows, candidates=BRIDGE_CANDIDATES_S, fps: int = DEFAULT_FPS) -> list[dict]:
    """What each candidate bridging threshold would actually do.

    Reports the share of gaps it closes, the resulting segment count per clip,
    and the overall speech fraction, so the choice is made on consequences
    rather than on the shape of the gap histogram alone.
    """
    all_gaps = np.array([g for row in rows for g in row["gaps_s"]], dtype=np.float64)
    total_frames = sum(row["n_frames"] for row in rows)
    literal_speech = sum(row["speech_frames_literal"] for row in rows)

    results = []
    for threshold in candidates:
        speech_frames = 0
        segment_counts = []
        for clip in clips:
            bridged = bridge_segments(clip.segments, threshold)
            speech_frames += int(rasterize(bridged, clip.duration_s, fps=fps).sum())
            segment_counts.append(len(bridged))

        results.append({
            "threshold_s": float(threshold),
            "gaps_bridged": int((all_gaps < threshold).sum()),
            "gap_fraction_bridged": float((all_gaps < threshold).mean()) if all_gaps.size else 0.0,
            "mean_segments_per_clip": float(np.mean(segment_counts)),
            "speech_fraction": speech_frames / total_frames if total_frames else 0.0,
            "speech_fraction_delta": (speech_frames - literal_speech) / total_frames if total_frames else 0.0,
        })
    return results


def propose_snr_gate(snr_values, percentile: float = 20.0) -> dict:
    """Pick an SNR threshold separating clean from already-degraded clips.

    The gate protects a controlled-degradation setup. Convolving an already
    noisy clip with a room impulse response reverberates its noise floor along
    with the speech, so the resulting sample is not "clean speech in room X"
    and the augmentation stops being a controlled variable. Degraded clips are
    kept raw instead.

    The threshold is data driven: a round number near the chosen low
    percentile, so the cut sits under the main mode rather than through it.
    """
    arr = np.asarray(list(snr_values), dtype=np.float64)
    if arr.size == 0:
        return {"threshold_db": None, "n_clean": 0, "n_degraded": 0}

    raw = float(np.percentile(arr, percentile))
    threshold = float(np.round(raw))
    return {
        "threshold_db": threshold,
        "percentile_used": percentile,
        "raw_percentile_db": raw,
        "n_clean": int((arr >= threshold).sum()),
        "n_degraded": int((arr < threshold).sum()),
        "clean_fraction": float((arr >= threshold).mean()),
    }


def propose_rumble_gate(lf_silence_values, percentile: float = 90.0) -> dict:
    """Pick a threshold on sub-80 Hz energy share within labeled silence."""
    arr = np.asarray(list(lf_silence_values), dtype=np.float64)
    if arr.size == 0:
        return {"threshold": None, "n_above": 0}

    raw = float(np.percentile(arr, percentile))
    threshold = float(np.round(raw, 2))
    return {
        "threshold_lf_fraction_in_silence": threshold,
        "percentile_used": percentile,
        "raw_percentile": raw,
        "n_above": int((arr > threshold).sum()),
        "fraction_above": float((arr > threshold).mean()),
        "median": float(np.median(arr)),
    }


def position_profile(rows) -> dict:
    """Mean speech probability against normalized position, over all clips."""
    curves = np.array([row["position_curve"] for row in rows], dtype=np.float64)
    mean = curves.mean(axis=0)
    centers = (np.arange(len(mean)) + 0.5) / len(mean)
    edge = max(1, len(mean) // 10)
    return {
        "position": centers.tolist(),
        "mean_speech_probability": mean.tolist(),
        "first_decile_mean": float(mean[:edge].mean()),
        "last_decile_mean": float(mean[-edge:].mean()),
        "middle_mean": float(mean[edge:-edge].mean()),
    }


def gap_elbow(gaps, bucket_ms: float = 10.0, floor_lo_ms: float = 200.0,
              floor_hi_ms: float = 400.0, factor: float = 1.5) -> dict:
    """Locate where the short-gap cluster meets the genuine-pause floor.

    The gap distribution has two regimes. Below roughly 100 ms the counts fall
    off steeply: those are aligner artifacts and within-word closures. Above
    it the counts flatten into a broad plateau of real pauses that is roughly
    uniform out to a second or more.

    The elbow is the first bucket whose count comes within `factor` times the
    plateau level, measured as the median count between `floor_lo_ms` and
    `floor_hi_ms`. Bridging up to the elbow removes the artifact cluster.
    Bridging past it starts deleting genuine pauses at the plateau rate.
    """
    arr = np.asarray(list(gaps), dtype=np.float64) * 1000.0
    if arr.size == 0:
        return {"elbow_ms": None}

    # Buckets are centered on exact multiples of `bucket_ms`. The aligner emits
    # gaps on a 10 ms grid, and float subtraction puts a nominal 10 ms gap at
    # 9.99999 ms, so edge-aligned buckets would split one spike across two.
    half = bucket_ms / 2.0
    edges = np.arange(-half, max(arr.max(), floor_hi_ms) + bucket_ms, bucket_ms)
    counts, _ = np.histogram(arr, bins=edges)
    centers = edges[:-1] + half

    in_floor = (centers >= floor_lo_ms) & (centers <= floor_hi_ms)
    floor_level = float(np.median(counts[in_floor])) if in_floor.any() else 0.0

    candidates = np.flatnonzero((counts <= factor * floor_level) & (centers > bucket_ms))
    elbow = float(centers[candidates[0]]) if candidates.size else None

    return {
        "elbow_ms": elbow,
        "plateau_count_per_bucket": floor_level,
        "bucket_ms": bucket_ms,
        "factor": factor,
        "bucket_centers_ms": centers[centers <= 500].tolist(),
        "bucket_counts": counts[centers <= 500].tolist(),
    }
