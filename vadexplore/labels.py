"""Label geometry: segment cleanup, frame rasterization, and the inverse.

The frame grid is fixed project-wide: 16 kHz audio, 10 ms hop, 25 ms analysis
window, so 100 frames per second. Frame i covers [i/fps, (i+1)/fps).

Note that rasterization uses the hop grid, not the 25 ms window. A frame's
label is defined by the 10 ms of time it advances, so consecutive frames tile
the clip exactly once. The 25 ms window matters for feature extraction, where
neighbouring frames deliberately overlap, and it must not leak into label
geometry or every boundary would be counted more than once.

`normalize_segments` is the single canonical cleanup. Everything else here
runs it first, so callers can pass raw loader segments safely.

numpy only.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TOL_S = 0.01  # one 10 ms frame
DEFAULT_FPS = 100     # 10 ms hop
WIN_S = 0.025         # analysis window, for features later, not for labels

# Guards threshold comparisons against float noise. Gaps are computed by
# subtraction, and 0.7 - 0.5 is 0.19999999999999996, so a bare "gap < 0.2"
# fires on a gap that is nominally exactly 0.2. Every threshold here is a
# round number a caller typed, so the comparisons are nudged to respect the
# number as written rather than its float representation.
_EPS = 1e-12


def _strictly_less(value: float, threshold: float) -> bool:
    """value < threshold, treating a nominally equal pair as not less."""
    return value < threshold - _EPS


def normalize_segments(
    segments: list[tuple[float, float]],
    tol_s: float = DEFAULT_TOL_S,
) -> list[tuple[float, float]]:
    """Clean forced-alignment artifacts out of a segment list.

    Drops zero-length segments, sorts by start, and merges runs whose gap is
    smaller than `tol_s` into maximal non-overlapping intervals. That covers
    the three things the aligner produces: segments touching at a shared
    endpoint, zero-length segments, and float jitter around a shared boundary
    (9.059999 against 9.06).

    This removes artifacts only. Genuine gaps of `tol_s` or more survive
    untouched, because bridging real silence is a separate decision with its
    own parameter, not something to smuggle in here.

    A gap of exactly `tol_s` survives. At the default that is one whole
    frame of silence, which the frame grid can represent, so it is real
    non-speech rather than something to smooth away.

    Idempotent: every gap in the output is at least `tol_s`, so a second pass
    is a no-op.
    """
    positive = [(float(s), float(e)) for s, e in segments if float(e) - float(s) > 0]
    if not positive:
        return []

    positive.sort(key=lambda seg: seg[0])

    merged: list[tuple[float, float]] = []
    cur_start, cur_end = positive[0]
    for start, end in positive[1:]:
        if _strictly_less(start - cur_end, tol_s):
            # touching, overlapping, or jitter apart, so extend the run.
            # max() matters for a segment fully contained in the current run.
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    return merged


def n_frames_for(duration_s: float, fps: int = DEFAULT_FPS) -> int:
    """Frame count for a clip. The one place the rounding rule lives."""
    return max(0, int(round(float(duration_s) * fps)))


def frame_times(n_frames: int, fps: int = DEFAULT_FPS) -> np.ndarray:
    """Start time in seconds of each frame, for plotting and alignment."""
    return np.arange(int(n_frames), dtype=np.float64) / float(fps)


def frame_overlap_s(
    segments: list[tuple[float, float]],
    n_frames: int,
    fps: int = DEFAULT_FPS,
) -> np.ndarray:
    """Seconds of speech inside each frame interval.

    Works segment by segment rather than frame by frame: a segment touches a
    contiguous index range, so each one is a single slice update. Cost scales
    with the covered frames, never with audio samples.

    Expects normalized segments. Overlapping input would double count.
    """
    n_frames = int(n_frames)
    overlap = np.zeros(n_frames, dtype=np.float64)
    if n_frames == 0:
        return overlap

    edges = np.arange(n_frames + 1, dtype=np.float64) / float(fps)
    for start, end in segments:
        lo = max(0, int(np.floor(start * fps)))
        hi = min(n_frames, int(np.ceil(end * fps)))
        if hi <= lo:
            continue
        covered = np.minimum(end, edges[lo + 1 : hi + 1]) - np.maximum(start, edges[lo:hi])
        overlap[lo:hi] += np.maximum(covered, 0.0)

    return overlap


def rasterize(
    segments: list[tuple[float, float]],
    duration_s: float,
    fps: int = DEFAULT_FPS,
    rule: str = "majority",
) -> np.ndarray:
    """Turn speech segments into an int8 frame label array.

    Frame i covers [i/fps, (i+1)/fps). Under "majority" a frame is speech when
    speech covers at least half of it; under "any" when it covers any of it at
    all. "any" is the more generous rule, so its frame count is always greater
    than or equal to "majority" on the same input.

    Segments are normalized internally, so raw loader segments are safe to
    pass. Anything past `duration_s` is dropped by the frame grid.
    """
    if rule not in ("majority", "any"):
        raise ValueError(f"unknown rule {rule!r}, expected 'majority' or 'any'")

    n_frames = n_frames_for(duration_s, fps)
    overlap = frame_overlap_s(normalize_segments(segments), n_frames, fps)

    if rule == "any":
        labels = overlap > _EPS
    else:
        labels = overlap + _EPS >= 0.5 / float(fps)  # at least half, ties count as speech

    return labels.astype(np.int8)


def bridge_segments(
    segments: list[tuple[float, float]],
    max_gap_s: float,
) -> list[tuple[float, float]]:
    """Merge consecutive segments separated by less than `max_gap_s`.

    This is the deliberate, parameterized counterpart to the artifact cleanup
    in `normalize_segments`: here we are knowingly relabeling short genuine
    silences (breaths, stop closures) as speech, because a VAD that chatters
    on and off through them is worse than one that rides across.

    Only gaps between two segments are bridged. The leading silence before the
    first segment and the trailing silence after the last are real non-speech
    and are never touched, since there is nothing on the far side to join to.
    """
    normalized = normalize_segments(segments)
    if len(normalized) < 2:
        return normalized

    bridged: list[tuple[float, float]] = []
    cur_start, cur_end = normalized[0]
    for start, end in normalized[1:]:
        if _strictly_less(start - cur_end, max_gap_s):
            cur_end = end
        else:
            bridged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    bridged.append((cur_start, cur_end))

    return normalize_segments(bridged)


def segments_from_frames(
    labels: np.ndarray,
    fps: int = DEFAULT_FPS,
) -> list[tuple[float, float]]:
    """Inverse of `rasterize`: 0/1 frames back to (start, end) segments.

    A run of speech frames [i, j] becomes (i/fps, (j+1)/fps), so boundaries
    land on the frame grid. Round-tripping a segment list therefore recovers
    the same regions to within one frame at each edge, which is the resolution
    the grid has.
    """
    flags = np.asarray(labels).astype(bool).astype(np.int8)
    if flags.size == 0:
        return []

    # +1 where a run starts, -1 where it ends, found by diffing a padded array
    padded = np.concatenate(([0], flags, [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return [(float(a) / fps, float(b) / fps) for a, b in zip(starts, ends)]


def make_labels(clip, fps: int = DEFAULT_FPS, bridge_gap_s: float = 0.2) -> dict:
    """Build the literal and bridged frame labels for one clip.

    `literal` rasterizes the normalized segments. `bridged` rasterizes those
    same segments after short gaps are filled. Both arrays share one frame
    grid derived from the clip duration, so they can be compared elementwise
    and stacked without any realignment.
    """
    segments_literal = normalize_segments(clip.segments)
    segments_bridged = bridge_segments(segments_literal, bridge_gap_s)

    literal = rasterize(segments_literal, clip.duration_s, fps=fps)
    bridged = rasterize(segments_bridged, clip.duration_s, fps=fps)

    n_frames = n_frames_for(clip.duration_s, fps)
    assert len(literal) == len(bridged) == n_frames
    # bridging only ever fills gaps, so it can never remove a speech frame
    assert int(bridged.sum()) >= int(literal.sum())

    return {
        "fps": fps,
        "bridge_gap_s": bridge_gap_s,
        "n_frames": n_frames,
        "literal": literal,
        "bridged": bridged,
        "segments_literal": segments_literal,
        "segments_bridged": segments_bridged,
    }
