"""Tests for vadexplore.loader, all built on synthetic fixtures."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from vadexplore.loader import load_clip, load_dataset, read_audio, summarize

KNOWN_STEM = "1272-128104-0000"


def write_pair(directory, stem, segments, duration_s=5.0, sr=16000, channels=1):
    """Drop a <stem>.wav / <stem>.json pair into `directory`."""
    frames = int(duration_s * sr)
    samples = np.zeros((frames, channels), dtype=np.float32)
    wav_path = directory / f"{stem}.wav"
    sf.write(str(wav_path), samples, sr)

    json_path = directory / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {"speech_segments": [{"start_time": s, "end_time": e} for s, e in segments]}
        )
    )
    return wav_path, json_path


@pytest.fixture
def clip_dir(tmp_path):
    write_pair(tmp_path, KNOWN_STEM, [(0.5, 1.2), (2.0, 2.8), (3.5, 4.9)])
    return tmp_path


def test_parses_expected_number_of_segments(clip_dir):
    clip = load_clip(clip_dir / KNOWN_STEM)
    assert len(clip.segments) == 3
    assert clip.segments == [(0.5, 1.2), (2.0, 2.8), (3.5, 4.9)]
    assert clip.warnings == []


def test_segments_are_sorted_by_start(tmp_path):
    write_pair(tmp_path, KNOWN_STEM, [(3.5, 4.9), (0.5, 1.2), (2.0, 2.8)])
    clip = load_clip(tmp_path / KNOWN_STEM)
    assert clip.segments == [(0.5, 1.2), (2.0, 2.8), (3.5, 4.9)]


def test_speaker_and_chapter_from_known_stem(clip_dir):
    clip = load_clip(clip_dir / KNOWN_STEM)
    assert clip.stem == KNOWN_STEM
    assert clip.speaker_id == "1272"
    assert clip.chapter_id == "128104"


def test_inverted_segment_counted_without_raising(tmp_path):
    write_pair(tmp_path, KNOWN_STEM, [(2.5, 1.0)])
    clip = load_clip(tmp_path / KNOWN_STEM)
    assert clip.n_zero_length == 1
    assert clip.segments == [(2.5, 1.0)]  # raw values kept


@pytest.mark.parametrize(
    "segments, needle",
    [
        ([(1.0, 2.0), (1.5, 3.0)], "genuine overlap"),
        ([(1.0, 99.0)], "exceeds audio duration"),
        ([(-0.5, 1.0)], "negative start"),
        ([], "empty"),
    ],
)
def test_other_validations_warn(tmp_path, segments, needle):
    write_pair(tmp_path, KNOWN_STEM, segments)
    clip = load_clip(tmp_path / KNOWN_STEM)
    assert any(needle in w for w in clip.warnings), clip.warnings


def test_missing_json_is_reported_by_load_dataset(tmp_path):
    write_pair(tmp_path, KNOWN_STEM, [(0.5, 1.2)])
    sf.write(str(tmp_path / "1272-128104-0001.wav"), np.zeros(1600, dtype=np.float32), 16000)

    clips = load_dataset(tmp_path)
    assert [c.stem for c in clips] == [KNOWN_STEM]
    assert clips.missing_json == ["1272-128104-0001"]
    assert clips.missing_wav == []


def test_load_dataset_limit(tmp_path):
    for i in range(4):
        write_pair(tmp_path, f"1272-128104-000{i}", [(0.5, 1.2)])
    assert len(load_dataset(tmp_path, limit=2)) == 2


def test_read_audio_downmixes_and_resamples(tmp_path):
    sr, duration_s = 44100, 1.0
    t = np.arange(int(sr * duration_s)) / sr
    left = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    stereo = np.stack([left, -left], axis=1)

    stem = "1272-128104-0000"
    sf.write(str(tmp_path / f"{stem}.wav"), stereo, sr)
    (tmp_path / f"{stem}.json").write_text(json.dumps({"speech_segments": [{"start_time": 0.0, "end_time": 1.0}]}))

    clip = load_clip(tmp_path / stem)
    assert clip.sample_rate == 44100
    assert clip.n_channels == 2

    audio = read_audio(clip, target_sr=16000)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == pytest.approx(16000, abs=2)
    # the two channels are exact opposites, so the downmix cancels to near silence
    # (the floor is PCM_16 quantization in the written file, not the downmix)
    assert np.max(np.abs(audio)) < 1e-4


def test_read_audio_preserves_tone_through_resampling(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    stem = "1272-128104-0000"
    sf.write(str(tmp_path / f"{stem}.wav"), np.stack([tone, tone], axis=1), sr)
    (tmp_path / f"{stem}.json").write_text(json.dumps({"speech_segments": []}))

    audio = read_audio(load_clip(tmp_path / stem), target_sr=16000)

    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.fft.rfftfreq(len(audio), 1 / 16000)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(440.0, abs=5.0)


# --- forced-alignment boundary artifacts ---------------------------------


def test_touching_segments_are_counted_not_warned(tmp_path):
    # shared endpoints and float jitter, the shape 900 of 957 real clips have
    write_pair(tmp_path, KNOWN_STEM, [(0.5, 1.2), (1.2, 2.0), (2.0, 3.0), (2.999999, 4.0)])
    clip = load_clip(tmp_path / KNOWN_STEM)

    assert clip.n_touching == 3
    assert clip.n_real_overlap == 0
    assert clip.n_zero_length == 0
    assert clip.warnings == []


def test_real_overlap_is_warned(tmp_path):
    # 100 ms deep, far past the one frame tolerance
    write_pair(tmp_path, KNOWN_STEM, [(0.5, 1.2), (1.1, 2.0)])
    clip = load_clip(tmp_path / KNOWN_STEM)

    assert clip.n_real_overlap == 1
    assert clip.n_touching == 0
    assert any("genuine overlap" in w for w in clip.warnings)


def test_zero_length_segment_does_not_count_as_overlap(tmp_path):
    write_pair(tmp_path, KNOWN_STEM, [(0.5, 2.0), (1.0, 1.0)])
    clip = load_clip(tmp_path / KNOWN_STEM)

    assert clip.n_zero_length == 1
    assert clip.n_real_overlap == 0
    assert clip.warnings == []


# --- path handling --------------------------------------------------------


def test_tilde_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / "vad_data"
    data_dir.mkdir()
    write_pair(data_dir, KNOWN_STEM, [(0.5, 1.2)])

    clips = load_dataset("~/vad_data")
    assert [c.stem for c in clips] == [KNOWN_STEM]

    clip = load_clip("~/vad_data/" + KNOWN_STEM)
    assert clip.wav_path.exists()
    assert read_audio(clip).dtype == np.float32


def test_load_dataset_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_dataset(tmp_path / "nope")


def test_load_dataset_raises_on_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no .wav/.json pairs"):
        load_dataset(tmp_path)


def test_load_dataset_raises_when_only_unpaired_files(tmp_path):
    sf.write(str(tmp_path / "1272-128104-0000.wav"), np.zeros(1600, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="1 wav without json"):
        load_dataset(tmp_path)


# --- summary line ---------------------------------------------------------


def test_summarize_reports_artifact_counts(tmp_path):
    write_pair(tmp_path, "1272-128104-0000", [(0.5, 1.2), (1.2, 2.0)])   # touching
    write_pair(tmp_path, "1462-170138-0000", [(0.5, 1.2), (1.1, 2.0)])   # real overlap
    line = summarize(load_dataset(tmp_path))

    assert "clips=2" in line
    assert "speakers=2" in line
    assert "clips_with_touching=1" in line
    assert "clips_with_real_overlap=1" in line
    assert "clips_with_zero_length=0" in line
