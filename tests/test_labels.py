"""Tests for the canonical segment cleanup."""

from __future__ import annotations

import pytest

from vadexplore.labels import normalize_segments


def test_touching_segments_merge():
    assert normalize_segments([(0.1, 0.3), (0.3, 0.5)]) == [(0.1, 0.5)]


def test_zero_length_segment_is_dropped():
    assert normalize_segments([(1.0, 1.5), (2.0, 2.0)]) == [(1.0, 1.5)]


def test_only_zero_length_returns_empty():
    assert normalize_segments([(2.0, 2.0)]) == []


def test_float_jitter_merges():
    merged = normalize_segments([(1.0, 2.0), (1.9999, 3.0)])
    assert len(merged) == 1
    assert merged[0] == pytest.approx((1.0, 3.0))


def test_genuine_gap_is_preserved():
    assert normalize_segments([(0.1, 0.3), (0.6, 0.9)]) == [(0.1, 0.3), (0.6, 0.9)]


def test_gap_just_over_tolerance_is_preserved():
    # 0.011 s apart, wider than one frame, so this is real silence
    assert len(normalize_segments([(0.1, 0.3), (0.311, 0.5)])) == 2


def test_unsorted_input_is_sorted_then_merged():
    assert normalize_segments([(0.6, 0.9), (0.1, 0.3)]) == [(0.1, 0.3), (0.6, 0.9)]


def test_contained_segment_does_not_shrink_the_run():
    assert normalize_segments([(0.0, 5.0), (1.0, 2.0)]) == [(0.0, 5.0)]


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [(0.1, 0.3), (0.3, 0.5)],
        [(1.0, 2.0), (1.9999, 3.0), (2.0, 2.0), (5.0, 6.0)],
        [(0.6, 0.9), (0.1, 0.3)],
    ],
)
def test_idempotent(segments):
    once = normalize_segments(segments)
    assert normalize_segments(once) == once


def test_tolerance_is_configurable():
    assert normalize_segments([(0.1, 0.3), (0.35, 0.5)], tol_s=0.1) == [(0.1, 0.5)]


# --- rasterization --------------------------------------------------------

import numpy as np

from vadexplore.labels import (
    bridge_segments,
    frame_times,
    make_labels,
    n_frames_for,
    rasterize,
    segments_from_frames,
)


def test_single_segment_exact_frame_indices():
    # [(0.10, 0.25)] in a 0.30 s clip at 100 fps, so 30 frames of 10 ms.
    # Both boundaries sit exactly on the grid: frame 10 is [0.100, 0.110) and
    # frame 24 is [0.240, 0.250). Frames 10 through 24 are fully covered,
    # frame 25 gets nothing, so both rules agree here.
    segments, duration_s = [(0.10, 0.25)], 0.30
    expected = set(range(10, 25))

    majority = rasterize(segments, duration_s, rule="majority")
    any_rule = rasterize(segments, duration_s, rule="any")

    assert len(majority) == 30
    assert set(np.flatnonzero(majority).tolist()) == expected
    assert set(np.flatnonzero(any_rule).tolist()) == expected
    assert majority.sum() == 15 and any_rule.sum() == 15
    assert set(np.flatnonzero(majority).tolist()) <= set(np.flatnonzero(any_rule).tolist())


def test_off_grid_segment_separates_the_two_rules():
    # [(0.106, 0.254)] in a 0.30 s clip. Frame 10 is [0.100, 0.110) and holds
    # 0.110 - 0.106 = 4 ms of speech, which is 40 percent, so "any" takes it
    # and "majority" does not. Frame 25 is [0.250, 0.260) and holds
    # 0.254 - 0.250 = 4 ms, same verdict. Frames 11 through 24 are full.
    segments, duration_s = [(0.106, 0.254)], 0.30

    majority = rasterize(segments, duration_s, rule="majority")
    any_rule = rasterize(segments, duration_s, rule="any")

    assert set(np.flatnonzero(majority).tolist()) == set(range(11, 25))
    assert set(np.flatnonzero(any_rule).tolist()) == set(range(10, 26))
    assert majority.sum() == 14 and any_rule.sum() == 16
    assert set(np.flatnonzero(majority).tolist()) < set(np.flatnonzero(any_rule).tolist())


def test_exactly_half_covered_frame_is_speech():
    # frame 10 is [0.100, 0.110) and gets 0.105 to 0.110, exactly 5 ms.
    # "at least 50 percent" means this counts.
    assert rasterize([(0.105, 0.115)], 0.30, rule="majority")[10] == 1


def test_frame_count_rounds():
    assert len(rasterize([(0.0, 1.0)], 3.017)) == round(3.017 * 100) == 302
    assert n_frames_for(3.017) == 302


def test_rasterize_dtype_and_rule_validation():
    assert rasterize([(0.1, 0.2)], 1.0).dtype == np.int8
    with pytest.raises(ValueError, match="unknown rule"):
        rasterize([(0.1, 0.2)], 1.0, rule="most")


def test_rasterize_normalizes_raw_segments():
    # touching plus a zero-length artifact, the shape the loader hands over
    raw = [(0.10, 0.18), (0.18, 0.25), (0.20, 0.20)]
    assert np.array_equal(rasterize(raw, 0.30), rasterize([(0.10, 0.25)], 0.30))


def test_any_covers_majority_on_random_segments():
    rng = np.random.default_rng(0)
    for _ in range(50):
        starts = np.sort(rng.uniform(0, 4.9, size=6))
        segments = [(float(s), float(s + rng.uniform(0.001, 0.4))) for s in starts]
        majority = rasterize(segments, 5.0, rule="majority")
        any_rule = rasterize(segments, 5.0, rule="any")
        assert any_rule.sum() >= majority.sum()
        assert np.all(any_rule >= majority)


# --- bridging -------------------------------------------------------------


def test_bridging_merges_gap_under_threshold():
    # gap is 0.64 - 0.50 = 0.14 s
    segments = [(0.0, 0.5), (0.64, 1.0)]
    assert bridge_segments(segments, 0.2) == [(0.0, 1.0)]
    assert bridge_segments(segments, 0.1) == [(0.0, 0.5), (0.64, 1.0)]


def test_bridging_is_strict_at_the_threshold():
    # a gap of exactly max_gap_s is not bridged, the rule is strictly less than
    assert len(bridge_segments([(0.0, 0.5), (0.7, 1.0)], 0.2)) == 2
    assert len(bridge_segments([(0.0, 0.5), (0.699, 1.0)], 0.2)) == 1


def test_bridging_never_touches_leading_or_trailing_silence():
    labels = make_labels(_FakeClip([(0.3, 0.7)], 1.0), bridge_gap_s=0.5)

    for name in ("literal", "bridged"):
        frames = labels[name]
        assert len(frames) == 100
        assert frames[:30].sum() == 0, f"{name} leaked into leading silence"
        assert frames[70:].sum() == 0, f"{name} leaked into trailing silence"
        assert set(np.flatnonzero(frames).tolist()) == set(range(30, 70))

    assert labels["segments_bridged"] == [(0.3, 0.7)]


def test_bridged_never_has_fewer_speech_frames():
    # mixed gaps: 0.05 s (bridged), 0.30 s (kept), 0.15 s (bridged)
    segments = [(0.10, 0.40), (0.45, 0.80), (1.10, 1.50), (1.65, 2.00)]
    labels = make_labels(_FakeClip(segments, 3.0), bridge_gap_s=0.2)

    assert labels["bridged"].sum() > labels["literal"].sum()
    assert np.all(labels["bridged"] >= labels["literal"])
    assert labels["segments_bridged"] == [(0.10, 0.80), (1.10, 2.00)]


# --- make_labels ----------------------------------------------------------


class _FakeClip:
    """Just the two attributes make_labels reads."""

    def __init__(self, segments, duration_s):
        self.segments = segments
        self.duration_s = duration_s


def test_make_labels_shape_and_keys():
    labels = make_labels(_FakeClip([(0.1, 0.5), (0.62, 1.4)], 2.0))

    assert labels["fps"] == 100
    assert labels["bridge_gap_s"] == 0.2
    assert labels["n_frames"] == 200
    assert len(labels["literal"]) == len(labels["bridged"]) == 200
    assert labels["literal"].dtype == np.int8 and labels["bridged"].dtype == np.int8
    assert labels["segments_literal"] == [(0.1, 0.5), (0.62, 1.4)]
    assert labels["segments_bridged"] == [(0.1, 1.4)]


# --- frame_times and the inverse ------------------------------------------


def test_frame_times():
    times = frame_times(5)
    assert np.allclose(times, [0.0, 0.01, 0.02, 0.03, 0.04])
    assert len(frame_times(302)) == 302


def test_segments_from_frames_hand_checked():
    frames = np.zeros(30, dtype=np.int8)
    frames[10:25] = 1
    assert segments_from_frames(frames) == [(0.10, 0.25)]


def test_segments_from_frames_multiple_runs_and_edges():
    frames = np.array([1, 1, 0, 1, 0, 0, 1], dtype=np.int8)
    assert segments_from_frames(frames) == [(0.0, 0.02), (0.03, 0.04), (0.06, 0.07)]


def test_segments_from_frames_empty_cases():
    assert segments_from_frames(np.zeros(10, dtype=np.int8)) == []
    assert segments_from_frames(np.array([], dtype=np.int8)) == []


def test_round_trip_within_one_frame():
    segments = [(0.104, 0.253), (0.61, 0.895), (1.2, 1.9)]
    duration_s = 2.5
    recovered = segments_from_frames(rasterize(segments, duration_s))
    expected = normalize_segments(segments)

    assert len(recovered) == len(expected)
    for (got_start, got_end), (want_start, want_end) in zip(recovered, expected):
        assert abs(got_start - want_start) <= 0.01
        assert abs(got_end - want_end) <= 0.01


def test_round_trip_is_exact_on_grid_aligned_segments():
    segments = [(0.10, 0.25), (0.60, 0.90)]
    recovered = segments_from_frames(rasterize(segments, 1.5))
    assert recovered == pytest.approx(segments)


def test_round_trip_of_frames_is_exactly_stable():
    # frames -> segments -> frames must be the identity, no drift
    rng = np.random.default_rng(1)
    frames = (rng.random(200) > 0.6).astype(np.int8)
    again = rasterize(segments_from_frames(frames), 2.0)
    assert np.array_equal(frames, again)


def test_one_frame_gap_survives_normalization():
    # 10 ms is exactly one frame, which the grid can represent, so it is real
    # silence and must not be smoothed away. Guards the float boundary:
    # 0.03 - 0.02 evaluates to 0.009999999999999998.
    assert normalize_segments([(0.01, 0.02), (0.03, 0.04)]) == [(0.01, 0.02), (0.03, 0.04)]
    assert rasterize([(0.01, 0.02), (0.03, 0.04)], 0.10).tolist() == [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
