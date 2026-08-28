"""Zero-phase high-pass, shared by every path that touches audio."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF

DEFAULT_SR = 16000
DEFAULT_HIGHPASS_HZ = 80.0
HIGHPASS_HZ_DEFAULT = DEFAULT_HIGHPASS_HZ  # alias for features.HIGHPASS_HZ
DEFAULT_HIGHPASS_ORDER = 2


def butter_highpass_sos(cutoff_hz: float, sr: int, order: int = DEFAULT_HIGHPASS_ORDER) -> np.ndarray:
    """Butterworth high-pass as cascaded biquads, shape (order / 2, 6).

    Rows are [b0, b1, b2, a0, a1, a2] with a0 normalized to 1. Even orders only.
    """
    if order <= 0 or order % 2:
        raise ValueError(f"order must be a positive even number, got {order}")
    if not 0 < cutoff_hz < sr / 2:
        raise ValueError(f"cutoff {cutoff_hz} Hz must lie in (0, {sr / 2})")

    w0 = 2.0 * np.pi * cutoff_hz / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)

    sections = []
    for i in range(order // 2):
        q = 1.0 / (2.0 * np.cos(np.pi * (2 * i + 1) / (2 * order)))
        alpha = sin_w0 / (2.0 * q)
        b = np.array([(1 + cos_w0) / 2.0, -(1 + cos_w0), (1 + cos_w0) / 2.0])
        a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
        sections.append(np.concatenate([b / a[0], a / a[0]]))
    return np.asarray(sections, dtype=np.float64)


def highpass(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    cutoff_hz: float | None = DEFAULT_HIGHPASS_HZ,
    order: int = DEFAULT_HIGHPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass. cutoff_hz=None passes through.

    Forward then backward, so nothing shifts against the labels. Side effect: the
    magnitude response is squared, so order 2 gives 6 dB at the cutoff, not 3.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"expected mono 1D audio, got shape {x.shape}")
    if cutoff_hz is None:
        return x
    if x.size == 0:
        return x

    tensor = torch.from_numpy(x.astype(np.float64))
    for section in butter_highpass_sos(cutoff_hz, sr, order):
        tensor = AF.filtfilt(
            tensor,
            a_coeffs=torch.from_numpy(section[3:].copy()),
            b_coeffs=torch.from_numpy(section[:3].copy()),
            clamp=False,
        )
    return tensor.numpy().astype(np.float32)
