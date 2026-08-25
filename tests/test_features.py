"""Tests for logmel, focused on the label grid alignment contract."""

from __future__ import annotations

import numpy as np
import pytest

from vadexplore.features import DEFAULT_N_MELS, logmel
from vadexplore.labels import n_frames_for


@pytest.mark.parametrize("duration_s", [0.30, 1.0, 3.017, 12.345])
def test_frame_count_matches_the_label_grid(duration_s):
    audio = np.zeros(int(round(duration_s * 16000)), dtype=np.float32)
    mel = logmel(audio)
    assert mel.shape[0] == DEFAULT_N_MELS
    assert abs(mel.shape[1] - n_frames_for(duration_s)) <= 1
    # the right-pad makes it exact, not merely within one frame
    assert mel.shape[1] == n_frames_for(duration_s)


def test_frame_count_can_be_forced():
    assert logmel(np.zeros(16000, dtype=np.float32), n_frames=77).shape == (40, 77)


def test_dtype_and_log_floor():
    mel = logmel(np.zeros(16000, dtype=np.float32))
    assert mel.dtype == np.float32
    assert np.all(np.isfinite(mel))  # silence must not produce -inf


def test_energy_lands_in_the_right_frames():
    # Burst from 0.500 s to 0.600 s, so hop-aligned frames 50 through 59.
    # Frame i spans [i * hop, i * hop + win), which is 25 ms or 2.5 hops, so
    # the window reaches 15 ms past its own frame. Frames 48 and 49 therefore
    # see the onset legitimately, and frame 60 starts exactly at the burst end
    # so it sees nothing. Expect exactly 48 through 59.
    audio = np.zeros(16000, dtype=np.float32)
    audio[8000:9600] = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)
    energy = logmel(audio).mean(axis=0)

    loud = np.flatnonzero(energy > energy.min() + 1.0)
    assert (loud.min(), loud.max()) == (48, 59)


def test_window_lookahead_is_bounded_to_two_frames():
    # Guards the property above: a hop-aligned onset must never light a frame
    # more than ceil(win/hop) - 1 = 2 frames early, or labels and features
    # would be misaligned by more than the window explains.
    audio = np.zeros(16000, dtype=np.float32)
    audio[8000:9600] = 0.5
    energy = logmel(audio).mean(axis=0)
    loud = np.flatnonzero(energy > energy.min() + 1.0)
    assert loud.min() >= 50 - 2


def test_rejects_multichannel():
    with pytest.raises(ValueError, match="mono 1D"):
        logmel(np.zeros((2, 16000), dtype=np.float32))


def test_empty_audio():
    assert logmel(np.zeros(0, dtype=np.float32)).shape == (40, 0)
