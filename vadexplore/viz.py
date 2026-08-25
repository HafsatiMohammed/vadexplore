"""Per-clip inspection figure for VAD.

Everything here sits on the fixed project frame grid, so the waveform, the
log-mel image, and every label ribbon share one time axis and one x-range. A
feature at time t lands on the same x pixel in every panel, which is the only
way an alignment bug is visible by eye.

`logmel` is re-exported from `vadexplore.features`. It lives there rather than
here so that training and evaluation can import the canonical feature function
without pulling in matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from vadexplore.features import DEFAULT_N_MELS, DEFAULT_SR, logmel
from vadexplore.labels import DEFAULT_FPS, frame_times, make_labels
from vadexplore.loader import read_audio

__all__ = ["logmel", "plot_clip"]

# Plotting only. A long clip has far more samples than the figure has pixels,
# so the waveform is drawn as a min/max envelope instead.
WAVE_MAX_POINTS = 4000

SPEECH_COLOR = "#2b6cb0"
BRIDGE_COLOR = "#dd6b20"
TRACK_COLORS = ["#38a169", "#805ad5", "#d53f8c", "#00838f"]


def _envelope(x: np.ndarray, n_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket (min, max) of a signal, so decimation cannot hide a peak."""
    k = len(x) // n_buckets
    usable = x[: k * n_buckets].reshape(n_buckets, k)
    return usable.min(axis=1), usable.max(axis=1)


def _is_binary(values: np.ndarray) -> bool:
    """True when an extra track holds only 0 and 1, so it wants a ribbon."""
    if values.dtype == bool:
        return True
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(np.all((finite == 0) | (finite == 1)))


def _draw_ribbon(ax, values: np.ndarray, fps: int, color: str, label: str) -> None:
    """Filled band over frames where `values` is nonzero.

    Uses frame edges with step="post", so each frame is painted from its own
    start to the next frame's start. That keeps the band aligned with the
    spectrogram column above it frame for frame.
    """
    edges = np.arange(len(values) + 1, dtype=np.float64) / fps
    mask = np.concatenate([np.asarray(values).astype(bool), [False]])

    ax.fill_between(edges, 0, 1, where=mask, step="post", color=color, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, labelpad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _draw_prob(ax, values: np.ndarray, fps: int, color: str, label: str) -> None:
    """Line in [0, 1] for a posterior or score track."""
    times = frame_times(len(values), fps)
    ax.plot(times, values, color=color, linewidth=0.9)
    ax.axhline(0.5, color="0.7", linewidth=0.6, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=7)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, labelpad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _normalize_tracks(extra_tracks) -> list[tuple[str, np.ndarray, bool]]:
    """Resolve extra_tracks into (name, values, is_binary).

    A bare array is auto-detected. A (values, kind) tuple with kind "binary" or
    "prob" overrides that, for the case of a posterior that happens to be
    saturated to 0 and 1 on a short clip.
    """
    resolved = []
    for name, entry in (extra_tracks or {}).items():
        if isinstance(entry, tuple):
            values, kind = entry
            if kind not in ("binary", "prob"):
                raise ValueError(f"track {name!r}: kind must be 'binary' or 'prob', got {kind!r}")
            binary = kind == "binary"
        else:
            values, binary = entry, None
        values = np.asarray(values).squeeze()
        if values.ndim != 1:
            raise ValueError(f"track {name!r}: expected 1D, got shape {values.shape}")
        resolved.append((name, values, _is_binary(values) if binary is None else binary))
    return resolved


def plot_clip(
    clip,
    labels: dict | None = None,
    extra_tracks: dict | None = None,
    max_seconds: float | None = None,
    ax=None,
    save=None,
    title: str | None = None,
    n_mels: int = DEFAULT_N_MELS,
    audio: np.ndarray | None = None,
):
    """Stacked waveform, log-mel, and label ribbons for one clip.

    All panels share one x-axis in seconds and one x-range, so vertical
    alignment is exact. The colorbar sits in its own gridspec column rather
    than being stolen from the spectrogram axes, which would have made that
    one panel narrower than the rest and broken the alignment.

    `audio` overrides the samples read from disk, which is how a filtered or
    augmented version of a clip gets plotted against the same labels.

    `labels` defaults to `make_labels(clip)`. `extra_tracks` maps a name to a
    frame-rate array, drawn as a ribbon when binary and as a line when it is a
    probability. `max_seconds` crops the view for dense clips. Pass `ax` to
    build the stack inside an existing subplot slot. `save` writes a PNG.

    Returns (figure, {panel_name: axes}).
    """
    if labels is None:
        labels = make_labels(clip)
    fps = labels["fps"]

    if audio is None:
        audio = read_audio(clip, target_sr=DEFAULT_SR)
    audio = np.asarray(audio, dtype=np.float32)
    tracks = _normalize_tracks(extra_tracks)

    # One frame count governs every panel, so cropping cannot desynchronize them.
    n_frames = labels["n_frames"]
    if max_seconds is not None:
        n_frames = min(n_frames, int(round(max_seconds * fps)))
    n_frames = max(1, n_frames)
    t_end = n_frames / fps

    audio = audio[: int(round(t_end * DEFAULT_SR))]
    mel = logmel(audio, sr=DEFAULT_SR, n_mels=n_mels, n_frames=n_frames)

    rows = [("wave", 2.4), ("mel", 3.0), ("literal", 0.45), ("bridged", 0.45)]
    rows += [(name, 0.45 if binary else 1.0) for name, _, binary in tracks]

    height = 0.55 + sum(h for _, h in rows) * 0.9
    if ax is None:
        fig = plt.figure(figsize=(12, height))
        gs = GridSpec(
            len(rows), 2,
            figure=fig,
            height_ratios=[h for _, h in rows],
            width_ratios=[1, 0.014],
            hspace=0.12,
            wspace=0.015,
        )
    else:
        fig = ax.get_figure()
        gs = ax.get_subplotspec().subgridspec(
            len(rows), 2,
            height_ratios=[h for _, h in rows],
            width_ratios=[1, 0.014],
            hspace=0.12,
            wspace=0.015,
        )
        ax.remove()

    axes: dict = {}
    first = None
    for i, (name, _) in enumerate(rows):
        panel = fig.add_subplot(gs[i, 0], sharex=first)
        first = first or panel
        axes[name] = panel

    # waveform
    wave_ax = axes["wave"]
    if len(audio) > WAVE_MAX_POINTS * 2:
        lo, hi = _envelope(audio, WAVE_MAX_POINTS)
        times = (np.arange(WAVE_MAX_POINTS) + 0.5) * t_end / WAVE_MAX_POINTS
        wave_ax.fill_between(times, lo, hi, color="0.35", linewidth=0)
    else:
        wave_ax.plot(np.arange(len(audio)) / DEFAULT_SR, audio, color="0.35", linewidth=0.6)
    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    wave_ax.set_ylim(-1.08 * max(peak, 1e-6), 1.08 * max(peak, 1e-6))
    wave_ax.set_ylabel("amplitude", fontsize=8)
    wave_ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        wave_ax.spines[side].set_visible(False)

    # log-mel, extent pinned to frame edges so column i covers [i/fps, (i+1)/fps)
    mel_ax = axes["mel"]
    image = mel_ax.imshow(
        mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, t_end, 0.0, mel.shape[0]),
        cmap="magma",
    )
    mel_ax.set_ylabel("mel bin", fontsize=8)
    mel_ax.tick_params(labelsize=7)

    cax = fig.add_subplot(gs[1, 1])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("log mel energy", fontsize=7)
    bar.ax.tick_params(labelsize=6)

    _draw_ribbon(axes["literal"], labels["literal"][:n_frames], fps, SPEECH_COLOR, "literal")
    _draw_ribbon(axes["bridged"], labels["bridged"][:n_frames], fps, BRIDGE_COLOR,
                 f"bridged\n{labels['bridge_gap_s']:g}s")

    for i, (name, values, binary) in enumerate(tracks):
        panel = axes[name]
        values = values[:n_frames]
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        if binary:
            _draw_ribbon(panel, values, fps, color, name)
        else:
            _draw_prob(panel, values, fps, color, name)

    for name, panel in axes.items():
        panel.set_xlim(0.0, t_end)
        if name != rows[-1][0]:
            panel.tick_params(labelbottom=False)
    axes[rows[-1][0]].set_xlabel("time (s)", fontsize=9)
    axes[rows[-1][0]].tick_params(labelsize=7)

    if title is None:
        title = (
            f"{clip.stem}   speaker {clip.speaker_id}   {clip.duration_s:.2f} s"
            + (f"   (first {t_end:.1f} s shown)" if t_end < clip.duration_s - 1e-6 else "")
        )
    fig.suptitle(title, fontsize=10, y=0.995)

    if save is not None:
        save = Path(os.path.expanduser(str(save)))
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")

    return fig, axes
