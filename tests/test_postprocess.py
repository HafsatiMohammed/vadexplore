"""Tests for the VAD post-processing operations and the fixed pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from vadexplore.postprocess import (
    PostprocessConfig,
    apply_pipeline,
    duration_frames,
    hangover,
    min_speech_duration,
    smooth,
    window_frames,
)

FPS = 100


def frames(pattern: str) -> np.ndarray:
    """'..##..' to a boolean array, so test cases stay readable."""
    return np.array([c == "#" for c in pattern], dtype=bool)


def render(array: np.ndarray) -> str:
    return "".join("#" if v else "." for v in np.asarray(array).astype(bool))


def segments_of(array: np.ndarray):
    from vadexplore.labels import segments_from_frames
    return segments_from_frames(np.asarray(array).astype(bool), FPS)


# --- conversions ----------------------------------------------------------


def test_window_frames_is_odd_and_centered():
    assert window_frames(10) == 1
    assert window_frames(30) == 3
    assert window_frames(50) == 5
    assert window_frames(90) == 9
    # even results are rounded up, so the window stays symmetric
    assert window_frames(20) == 3
    assert window_frames(40) == 5
    assert all(window_frames(ms) % 2 == 1 for ms in range(0, 500, 7))


def test_duration_frames_rounds_to_nearest():
    assert duration_frames(0) == 0
    assert duration_frames(50) == 5
    assert duration_frames(100) == 10
    assert duration_frames(25) == 2   # 2.5 rounds to even under banker's rounding
    assert duration_frames(-10) == 0


# --- smoothing ------------------------------------------------------------


def test_median_window_one_is_the_identity():
    rng = np.random.default_rng(0)
    probs = rng.random(50).astype(np.float32)
    assert np.array_equal(smooth(probs, "median", window_ms=10), probs)


def test_no_smoothing_method_is_the_identity():
    probs = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    assert np.array_equal(smooth(probs, "none", window_ms=90), probs)
    assert np.array_equal(smooth(probs, "median", window_ms=0), probs)


def test_median_window_three_removes_a_one_frame_spike():
    probs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = smooth(probs, "median", window_ms=30)
    assert out[3] == pytest.approx(0.0)
    assert np.allclose(out, 0.0)


def test_median_window_three_removes_a_one_frame_dropout():
    probs = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.allclose(smooth(probs, "median", window_ms=30), 1.0)


def test_median_preserves_a_real_step_edge():
    """A step is a fixed point of the median, which is why it beats averaging."""
    probs = np.array([0.0] * 6 + [1.0] * 6, dtype=np.float32)
    out = smooth(probs, "median", window_ms=50)
    assert np.array_equal(out, probs)

    averaged = smooth(probs, "moving_average", window_ms=50)
    assert not np.array_equal(averaged, probs)   # the average rounds the edge off


def test_moving_average_smooths_a_spike_without_removing_it():
    probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    out = smooth(probs, "moving_average", window_ms=30)
    assert out[2] == pytest.approx(1 / 3, abs=1e-6)
    assert out[1] == pytest.approx(1 / 3, abs=1e-6)  # spread, not deleted


def test_smoothing_uses_edge_padding_not_zeros():
    """Zero padding would drag the first and last frames toward non-speech."""
    probs = np.ones(9, dtype=np.float32)
    assert np.allclose(smooth(probs, "moving_average", window_ms=50), 1.0)
    assert np.allclose(smooth(probs, "median", window_ms=50), 1.0)


def test_smoothing_preserves_length():
    rng = np.random.default_rng(1)
    for n in (1, 2, 7, 100):
        probs = rng.random(n).astype(np.float32)
        for method in ("median", "moving_average"):
            for window_ms in (10, 30, 90, 250):
                assert len(smooth(probs, method, window_ms)) == n


def test_unknown_smoothing_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        smooth(np.zeros(5), "gaussian", 30)


# --- minimum speech duration ----------------------------------------------


def test_min_speech_removes_a_two_frame_blip_and_keeps_a_twenty_frame_segment():
    decisions = np.zeros(40, dtype=bool)
    decisions[5:7] = True      # 2 frames, 20 ms
    decisions[15:35] = True    # 20 frames, 200 ms

    out = min_speech_duration(decisions, min_ms=50)   # 5 frames

    assert not out[5:7].any(), "the 2 frame blip survived"
    assert out[15:35].all(), "the 20 frame segment was removed"
    assert out.sum() == 20


@pytest.mark.parametrize("length, min_ms, survives", [
    (2, 50, False), (4, 50, False), (5, 50, True), (6, 50, True),
    (9, 100, False), (10, 100, True),
])
def test_min_speech_boundary_is_at_the_stated_duration(length, min_ms, survives):
    decisions = np.zeros(30, dtype=bool)
    decisions[10:10 + length] = True
    assert bool(min_speech_duration(decisions, min_ms).any()) is survives


def test_min_speech_zero_is_the_identity():
    decisions = frames("..##...#....####..")
    assert np.array_equal(min_speech_duration(decisions, 0), decisions)


def test_min_speech_only_removes_speech_never_adds():
    rng = np.random.default_rng(2)
    decisions = rng.random(200) > 0.6
    out = min_speech_duration(decisions, 100)
    assert np.all(decisions[out]), "a frame was turned on"
    assert out.sum() <= decisions.sum()


# --- hangover -------------------------------------------------------------


def test_hangover_extends_by_exactly_n_frames():
    decisions = frames("..###.............")
    out = hangover(decisions, hang_ms=30)   # 3 frames
    assert render(out) == "..######.........."
    assert out.sum() == decisions.sum() + 3


def test_hangover_merges_two_segments_closer_than_the_extension():
    decisions = frames("..##...##.........")   # a 3 frame gap between them

    # 5 frames of extension is wider than the gap, so the two become one
    merged = hangover(decisions, hang_ms=50)
    assert render(merged) == "..############...."
    assert len(segments_of(merged)) == 1

    # 3 frames exactly closes the gap, which also merges them
    exact = hangover(decisions, hang_ms=30)
    assert render(exact) == "..##########......"
    assert len(segments_of(exact)) == 1

    # 2 frames leaves a frame of silence, so they stay separate
    kept = hangover(decisions, hang_ms=20)
    assert render(kept) == "..####.####......."
    assert len(segments_of(kept)) == 2


def test_hangover_does_not_cascade():
    """Extension is measured from the original offsets, not the new ones."""
    decisions = frames("#....#....#.......")
    out = hangover(decisions, hang_ms=30)   # 3 frames for each of 3 segments
    assert render(out) == "####.####.####...."
    assert out.sum() == decisions.sum() + 3 * 3


def test_hangover_clips_at_the_end_of_the_clip():
    decisions = frames("........#")
    out = hangover(decisions, hang_ms=200)
    assert len(out) == len(decisions)
    assert out[-1]


def test_hangover_zero_is_the_identity():
    decisions = frames("..##...#....####..")
    assert np.array_equal(hangover(decisions, 0), decisions)


def test_hangover_only_adds_speech_never_removes():
    rng = np.random.default_rng(3)
    decisions = rng.random(200) > 0.7
    out = hangover(decisions, 100)
    assert np.all(out[decisions]), "a frame was turned off"
    assert out.sum() >= decisions.sum()


# --- the pipeline ---------------------------------------------------------


def test_pipeline_preserves_length():
    rng = np.random.default_rng(4)
    config = PostprocessConfig(smooth_method="median", smooth_ms=50,
                               min_speech_ms=100, hangover_ms=100)
    for n in (1, 3, 17, 500):
        probs = rng.random(n).astype(np.float32)
        out = apply_pipeline(probs, 0.5, config)
        assert len(out) == n
        assert out.dtype == bool


def test_pipeline_with_no_operations_is_a_bare_threshold():
    rng = np.random.default_rng(5)
    probs = rng.random(200).astype(np.float32)
    raw = PostprocessConfig(smooth_ms=0, min_speech_ms=0, hangover_ms=0)
    assert raw.is_identity
    assert np.array_equal(apply_pipeline(probs, 0.42, raw), probs >= 0.42)


def test_pipeline_order_min_speech_runs_before_hangover():
    """Order matters, and the wrong one keeps a blip the right one deletes.

    A 2 frame burst with a 5 frame minimum and a 5 frame hangover: filtering
    first deletes it, so hangover has nothing to extend. Extending first would
    grow it to 7 frames, which then passes the duration filter and survives.
    """
    probs = np.zeros(20, dtype=np.float32)
    probs[2:4] = 1.0

    config = PostprocessConfig(smooth_ms=0, min_speech_ms=50, hangover_ms=50)
    correct = apply_pipeline(probs, 0.5, config)
    assert not correct.any(), f"blip survived the correct order: {render(correct)}"

    wrong_order = min_speech_duration(hangover(probs >= 0.5, 50), 50)
    assert wrong_order.any(), "the test cannot tell the two orders apart"


def test_pipeline_order_smoothing_runs_before_thresholding():
    """Smoothing the posterior is not the same as smoothing the decisions.

    A posterior chattering either side of the threshold averages to a value
    above it, so smoothing first yields one clean segment. Applied after
    thresholding, no filter can recover that: the information about how far
    below the threshold each frame sat is already gone.
    """
    probs = np.tile([0.45, 0.65], 10).astype(np.float32)

    thresholded_first = probs >= 0.5
    assert thresholded_first.sum() == 10
    assert len(segments_of(thresholded_first)) == 10   # fully fragmented

    config = PostprocessConfig(smooth_method="moving_average", smooth_ms=30)
    smoothed_first = apply_pipeline(probs, 0.5, config)
    assert smoothed_first.all()
    assert len(segments_of(smoothed_first)) == 1
    assert smoothed_first.sum() > thresholded_first.sum()


def test_pipeline_is_deterministic():
    rng = np.random.default_rng(6)
    probs = rng.random(300).astype(np.float32)
    config = PostprocessConfig(smooth_ms=50, min_speech_ms=100, hangover_ms=50)
    first = apply_pipeline(probs, 0.5, config)
    assert np.array_equal(first, apply_pipeline(probs, 0.5, config))


def test_config_label_is_readable():
    assert PostprocessConfig(smooth_ms=0, min_speech_ms=0, hangover_ms=0).label() == "raw"
    assert PostprocessConfig(smooth_ms=50, min_speech_ms=100,
                             hangover_ms=50).label() == "med50+minsp100+hang50"


def test_empty_input_is_handled():
    empty = np.zeros(0, dtype=np.float32)
    assert len(apply_pipeline(empty, 0.5, PostprocessConfig(smooth_ms=50))) == 0
    assert len(min_speech_duration(np.zeros(0, bool), 100)) == 0
    assert len(hangover(np.zeros(0, bool), 100)) == 0
