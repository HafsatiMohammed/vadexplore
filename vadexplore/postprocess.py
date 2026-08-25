"""Standard VAD post-processing on frame-level speech posteriors.

Three composable operations, applied in one fixed order:

    smooth the posterior  ->  threshold  ->  min speech duration  ->  hangover

The order is not arbitrary. Smoothing acts on the continuous score, where it
can suppress a spike without committing to a decision. Duration filtering has
to come after thresholding, since a duration only exists once there are
segments. Hangover comes last so it extends the segments that survived, rather
than extending spurious bursts that the duration filter is about to delete.

The threshold stays at whatever value validation chose. Post-processing is
scored against the same operating point as the raw baseline, so any gain is
attributable to the operations and not to re-tuning the threshold underneath
them.

Parameters are in milliseconds. Conversion to frames happens here, once,
against the fixed 100 fps grid.

numpy only, plus the canonical segment conversion from `vadexplore.labels`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from vadexplore.labels import DEFAULT_FPS, segments_from_frames

SMOOTHING_METHODS = ("median", "moving_average", "none")


def window_frames(window_ms: float, fps: int = DEFAULT_FPS) -> int:
    """Milliseconds to an odd, centered window length in frames.

    Rounded up to odd so the window is symmetric about the frame it is
    smoothing. An even window would shift every value half a frame, which is
    exactly the kind of small systematic boundary bias this whole project has
    been careful to avoid.
    """
    n = max(1, int(round(float(window_ms) * fps / 1000.0)))
    return n if n % 2 else n + 1


def duration_frames(duration_ms: float, fps: int = DEFAULT_FPS) -> int:
    """Milliseconds to a whole number of frames, rounded to nearest."""
    return max(0, int(round(float(duration_ms) * fps / 1000.0)))


# --- the three operations -------------------------------------------------


def smooth(probs: np.ndarray, method: str = "median", window_ms: float = 0.0,
           fps: int = DEFAULT_FPS) -> np.ndarray:
    """Smooth a posterior before it is thresholded.

    `median` removes an isolated spike outright while leaving a real transition
    where it was, because a step edge is a fixed point of the median. A moving
    average instead spreads the spike over the window and rounds off the edge,
    which shifts the crossing point. Both are offered so the sweep can show the
    difference rather than assert it.

    A window of one frame, or `method="none"`, is the identity.
    """
    if method not in SMOOTHING_METHODS:
        raise ValueError(f"method must be one of {SMOOTHING_METHODS}, got {method!r}")

    values = np.asarray(probs, dtype=np.float64)
    if method == "none" or values.size == 0:
        return values.astype(np.float32)

    n = window_frames(window_ms, fps)
    if n <= 1:
        return values.astype(np.float32)

    half = n // 2
    # edge padding, not zero padding: zeros would drag the first and last
    # frames of every clip toward non-speech regardless of what is there
    padded = np.pad(values, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, n)

    if method == "median":
        out = np.median(windows, axis=-1)
    else:
        out = windows.mean(axis=-1)
    return out.astype(np.float32)


def min_speech_duration(decisions: np.ndarray, min_ms: float = 0.0,
                        fps: int = DEFAULT_FPS) -> np.ndarray:
    """Delete speech segments shorter than `min_ms`.

    A 10 or 20 ms burst of speech is not something a listener registers, and
    downstream it shows up as a spurious trigger. Measured earlier on this
    corpus, about half of all false-positive runs at threshold 0.5 were three
    frames or shorter, so this is the operation with the most to remove.
    """
    frames = np.asarray(decisions).astype(bool).copy()
    minimum = duration_frames(min_ms, fps)
    if minimum <= 1 or frames.size == 0:
        return frames

    for start, end in segments_from_frames(frames, fps):
        lo, hi = int(round(start * fps)), int(round(end * fps))
        if hi - lo < minimum:
            frames[lo:hi] = False
    return frames


def hangover(decisions: np.ndarray, hang_ms: float = 0.0,
             fps: int = DEFAULT_FPS) -> np.ndarray:
    """Extend every speech segment by `hang_ms` past its offset.

    The classic VAD hangover. It does two things at once: it stops the quiet
    tail of a word being clipped, since energy decays below the threshold
    before the word is actually over, and it bridges brief dips inside
    continuous speech by letting one segment run into the next.

    Extension is computed from the segments as they were on entry, so a
    segment extended into another does not then extend again from the new
    offset and cascade.
    """
    frames = np.asarray(decisions).astype(bool).copy()
    extension = duration_frames(hang_ms, fps)
    if extension <= 0 or frames.size == 0:
        return frames

    for _, end in segments_from_frames(frames, fps):
        hi = int(round(end * fps))
        frames[hi:min(hi + extension, len(frames))] = True
    return frames


# --- the pipeline ---------------------------------------------------------


@dataclass(frozen=True)
class PostprocessConfig:
    """One post-processing setting. All zeros is the raw thresholded baseline."""

    smooth_method: str = "median"
    smooth_ms: float = 0.0
    min_speech_ms: float = 0.0
    hangover_ms: float = 0.0

    @property
    def is_identity(self) -> bool:
        """True when the pipeline reduces to a bare threshold."""
        smoothing_off = self.smooth_ms <= 0 or self.smooth_method == "none"
        return smoothing_off and self.min_speech_ms <= 0 and self.hangover_ms <= 0

    def label(self) -> str:
        parts = []
        if self.smooth_ms > 0 and self.smooth_method != "none":
            parts.append(f"{self.smooth_method[:3]}{self.smooth_ms:g}")
        if self.min_speech_ms > 0:
            parts.append(f"minsp{self.min_speech_ms:g}")
        if self.hangover_ms > 0:
            parts.append(f"hang{self.hangover_ms:g}")
        return "+".join(parts) if parts else "raw"

    def to_dict(self) -> dict:
        return asdict(self)


def apply_pipeline(probs: np.ndarray, threshold: float,
                   config: PostprocessConfig = PostprocessConfig(),
                   fps: int = DEFAULT_FPS) -> np.ndarray:
    """Posterior to final decisions, in the fixed order.

    `threshold` is the validation-chosen operating point and is never adjusted
    here, so a comparison against the raw baseline isolates the effect of the
    operations.
    """
    values = np.asarray(probs, dtype=np.float64)
    smoothed = smooth(values, config.smooth_method, config.smooth_ms, fps)
    decisions = smoothed >= threshold
    decisions = min_speech_duration(decisions, config.min_speech_ms, fps)
    decisions = hangover(decisions, config.hangover_ms, fps)

    assert len(decisions) == len(values), "post-processing must preserve length"
    return decisions
