"""Acoustic features on the project frame grid.

The grid is fixed project-wide and matches `vadexplore.labels`: 16 kHz audio,
10 ms hop, 25 ms analysis window, 40 log-mel bins, so 100 frames per second.

Backend is torchaudio. torch is already a project dependency and torchaudio is
its companion, so this costs no new dependency stack. librosa would have
dragged in numba and soxr for the same spectrogram.

This module deliberately does not import matplotlib. `logmel` is the canonical
feature function for the whole project, including training and evaluation, and
those must not pull in a plotting stack to get a spectrogram. `vadexplore.viz`
re-exports it for convenience.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from vadexplore.labels import DEFAULT_FPS, WIN_S
from vadexplore.preprocess import HIGHPASS_HZ_DEFAULT as _HP, highpass  # noqa: F401

DEFAULT_SR = 16000
DEFAULT_N_MELS = 40
DEFAULT_WIN_MS = WIN_S * 1000  # 25.0
DEFAULT_HOP_MS = 1000.0 / DEFAULT_FPS  # 10.0

LOG_FLOOR = 1e-10

# The committed front-end step, re-exported from `preprocess` so that callers
# already importing it from here keep working. See the rumble row in
# DECISIONS.md for why it is committed.
HIGHPASS_HZ = _HP


def _resolve_n_frames(x: np.ndarray, hop: int, n_frames: int | None) -> int:
    """Frame count for a signal, matching `labels.n_frames_for` on duration."""
    if n_frames is None:
        n_frames = int(round(len(x) / hop))
    return max(0, int(n_frames))


def _pad_to_frames(x: np.ndarray, hop: int, win: int, n_frames: int) -> np.ndarray:
    """Right-pad or trim so a center=False STFT emits exactly `n_frames`."""
    needed = (n_frames - 1) * hop + win
    if len(x) < needed:
        return np.pad(x, (0, needed - len(x)))
    return x[:needed]


def logmel(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    win_ms: float = DEFAULT_WIN_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    n_fft: int | None = None,
    f_min: float = 0.0,
    f_max: float | None = None,
    n_frames: int | None = None,
) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, n_frames), float32.

    Frame i spans [i * hop, i * hop + win) in samples, so feature frame i and
    label frame i begin at the same instant. That is the whole point of the
    padding below.

    torchaudio's default center=True would instead center frame i on i * hop
    and emit 1 + len(audio) // hop frames, one more than the label grid has,
    with every frame offset by half a window. Here center=False keeps the left
    edges aligned, and the signal is padded on the right to exactly
    (n_frames - 1) * hop + win samples so the frame count comes out exact
    rather than short by the window tail.

    With no `n_frames` given the count is round(len(audio) / hop), which is the
    same rule `labels.n_frames_for` applies to the clip duration.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"expected mono 1D audio, got shape {x.shape}")

    hop = int(round(sr * hop_ms / 1000.0))
    win = int(round(sr * win_ms / 1000.0))
    n_fft = int(n_fft) if n_fft is not None else win

    n_frames = _resolve_n_frames(x, hop, n_frames)
    if n_frames == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    x = _pad_to_frames(x, hop, win, n_frames)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win,
        hop_length=hop,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=False,
    )
    spec = mel(torch.from_numpy(x))
    spec = torch.log(spec + LOG_FLOOR)

    out = spec.numpy().astype(np.float32)
    assert out.shape == (n_mels, n_frames), f"{out.shape} != {(n_mels, n_frames)}"
    return out


def power_spectrogram(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    win_ms: float = DEFAULT_WIN_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    n_fft: int | None = None,
    n_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear power spectrogram on the same frame grid as `logmel`.

    Returns (power of shape (n_freq, n_frames), frequency of each bin in Hz).

    This exists because the mel filterbank cannot answer questions about
    specific low frequencies. At 40 mels over 0 to 8000 Hz the first triangle
    spans 40 to 80 Hz and nothing covers 0 to 40 Hz at all, so a sub-80 Hz
    rumble measurement taken from mel bins would silently discard the bottom
    half of the band it is meant to measure. Framing, padding, and window are
    identical to `logmel`, so frame i means the same thing in both.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"expected mono 1D audio, got shape {x.shape}")

    hop = int(round(sr * hop_ms / 1000.0))
    win = int(round(sr * win_ms / 1000.0))
    n_fft = int(n_fft) if n_fft is not None else win

    n_frames = _resolve_n_frames(x, hop, n_frames)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    if n_frames == 0:
        return np.zeros((len(freqs), 0), dtype=np.float32), freqs
    x = _pad_to_frames(x, hop, win, n_frames)

    spec = torch.stft(
        torch.from_numpy(x),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win),
        center=False,
        return_complex=True,
    )
    power = spec.abs().pow(2).numpy().astype(np.float32)
    assert power.shape == (len(freqs), n_frames), f"{power.shape} != {(len(freqs), n_frames)}"
    return power, freqs


def frame_energy(power: np.ndarray) -> np.ndarray:
    """Total energy per frame, summed over frequency."""
    return power.sum(axis=0)


def band_fraction(power: np.ndarray, freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """Per-frame share of energy inside [f_lo, f_hi). Zero-energy frames give 0."""
    band = (freqs >= f_lo) & (freqs < f_hi)
    total = power.sum(axis=0)
    return np.divide(power[band].sum(axis=0), total, out=np.zeros_like(total), where=total > 0)

