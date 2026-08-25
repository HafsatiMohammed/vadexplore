"""Signal conditioning applied before any feature extraction.

One high-pass lives here and everything else imports it, so the analysis
scripts, the Silero cross-check, and the training data layer all see the same
preprocessed signal. See the rumble row in DECISIONS.md for why it is
committed: sub-80 Hz hum sits almost entirely in the silence frames, which is
exactly the contrast the task depends on.

numpy and torchaudio only. The Butterworth coefficients are designed here
rather than taken from scipy, which is not a dependency; the design is
validated against `scipy.signal.butter` to within 1e-9 dB in
`tests/test_data.py`.
"""

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

    Each row is [b0, b1, b2, a0, a1, a2] with a0 normalized to 1. Section i
    realizes one Butterworth pole pair, whose Q comes from the pole angle
    `pi * (2i + 1) / (2 * order)`; cascading them gives the maximally flat
    response. Only even orders, since an odd order needs a leftover real pole
    that does not fit the biquad cascade.
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
    """Zero-phase Butterworth high-pass. `cutoff_hz=None` is a pass-through.

    Zero phase because the filter runs forward and then backward, so the two
    group delays cancel exactly. That matters here: a causal filter would shift
    the audio relative to labels that were aligned on the unfiltered signal,
    and a few milliseconds of drift is the same order as the boundary effects
    being measured. The cost is that the effective magnitude response is
    squared, so an order-2 section attenuates 6 dB at the cutoff rather than 3.

    Passing `cutoff_hz=None` disables filtering, which is what makes the
    with-versus-without ablation a one-line config change.
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
