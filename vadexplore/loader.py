"""Dataset loading for the VAD exploration set.

Looking at the datset; I already saw this type file naming LibriSpeech (speaker-chapter-utterance),
and after some Listening I can confirm that same speaker share the same ID.

The dataset looks like  is a flat directory of paired files sharing a stem:

    <speaker>-<chapter>-<utterance>.wav    audio
    <speaker>-<chapter>-<utterance>.json   labels, {"speech_segments": [{"start_time", "end_time"}, ...]}

Metadata scanning (`load_clip`, `load_dataset`)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from vadexplore.labels import DEFAULT_TOL_S

DEFAULT_SR = 16000


@dataclass
class Clip:
    """ A clip have the following metadata:
        stem
        wav path 
        json path 
        speaker id
        chapter id
        duration in secods 
        sample rate 
        segments list of (start, end) float tuples sorted by start  
        Warnings list of strings in case something is wrong 
    """
    stem: str
    wav_path: Path
    json_path: Path
    speaker_id: str
    chapter_id: str
    duration_s: float
    sample_rate: int
    n_channels: int
    segments: list[tuple[float, float]]
    warnings: list[str] = field(default_factory=list)
    # Segment issue counts, classified at a one frame tolerance. Only
    # n_real_overlap indicates an actual problem, see _validate.
    n_zero_length: int = 0
    n_touching: int = 0
    n_real_overlap: int = 0

def _resolve(path) -> Path:
    """Expand ~. An unexpanded ~ globs to nothing, which reads as an empty dataset."""
    return Path(os.path.expanduser(str(path)))


def _split_stem(stem: str) -> tuple[str, str]:
    """ Get speaker and chapter from stem
    """
    parts = stem.split("-")
    speaker = parts[0] if len(parts) > 0 else ""
    chapter = parts[1] if len(parts) > 1 else ""
    return speaker, chapter


def _classify_segments(
    segments: list[tuple[float, float]],
    tol_s: float = DEFAULT_TOL_S,
) -> tuple[int, int, int]:
    """Count adjacency issues as (zero_length, touching, real_overlap).

    Only real overlap, deeper than tol_s, indicates a problem. Zero-length
    segments stay out of the adjacency pass so they are not counted twice.
    """
    zero_length = sum(1 for start, end in segments if end - start <= 0)

    positive = sorted(
        ((start, end) for start, end in segments if end - start > 0),
        key=lambda seg: seg[0],
    )

    touching = 0
    real_overlap = 0
    prev_end = None
    for start, end in positive:
        if prev_end is not None:
            if start < prev_end - tol_s:
                real_overlap += 1
            elif start <= prev_end + tol_s:
                touching += 1
        # running max, so a contained segment does not reset the boundary
        prev_end = end if prev_end is None else max(prev_end, end)

    return zero_length, touching, real_overlap


def _validate(
    segments: list[tuple[float, float]],
    duration_s: float,
    tol_s: float = DEFAULT_TOL_S,
) -> tuple[list[str], int, int, int]:
    """Warnings plus the segment issue counts. Never raises."""
    warnings: list[str] = []

    if not segments:
        warnings.append("speech_segments is empty")

    for start, end in segments:
        if start < 0:
            warnings.append(f"negative start time {start:g}")
        if end > duration_s:
            warnings.append(f"segment end {end:g}s exceeds audio duration {duration_s:g}s")

    zero_length, touching, real_overlap = _classify_segments(segments, tol_s)
    if real_overlap:
        warnings.append(
            f"{real_overlap} genuine overlap(s) deeper than {tol_s * 1000:g} ms"
        )

    return warnings, zero_length, touching, real_overlap


def load_clip(stem_or_path) -> Clip:
    """Load metadata and labels for one clip
    """
    path = _resolve(stem_or_path)
    base = path.with_suffix("") if path.suffix in (".wav", ".json") else path
    wav_path = base.with_suffix(".wav")
    json_path = base.with_suffix(".json")
    stem = base.name

    info = sf.info(str(wav_path))
    duration_s = float(info.frames) / float(info.samplerate)

    with open(json_path) as fh:
        labels = json.load(fh)

    raw = labels.get("speech_segments", [])
    segments = [(float(s["start_time"]), float(s["end_time"])) for s in raw]
    segments.sort(key=lambda seg: seg[0])

    speaker_id, chapter_id = _split_stem(stem)
    warnings, n_zero_length, n_touching, n_real_overlap = _validate(segments, duration_s)

    return Clip(
        stem=stem,
        wav_path=wav_path,
        json_path=json_path,
        speaker_id=speaker_id,
        chapter_id=chapter_id,
        duration_s=duration_s,
        sample_rate=int(info.samplerate),
        n_channels=int(info.channels),
        segments=segments,
        warnings=warnings,
        n_zero_length=n_zero_length,
        n_touching=n_touching,
        n_real_overlap=n_real_overlap,
    )


def find_pairs(directory) -> tuple[list[str], list[str], list[str]]:
    """Scan a directory for stems, returning (paired, wav_only, json_only).
    All three lists are sorted. The unpaired lists are what `load_dataset`
    reports as missing files.
    """
    directory = _resolve(directory)
    wavs = {p.stem for p in directory.glob("*.wav")}
    jsons = {p.stem for p in directory.glob("*.json")}
    return sorted(wavs & jsons), sorted(wavs - jsons), sorted(jsons - wavs)


class _ClipList(list):
    """A plain list of clips that also carries the stems missing half a pair in case we have them"""

    def __init__(self, clips=(), missing_json=(), missing_wav=()):
        super().__init__(clips)
        self.missing_json = list(missing_json)
        self.missing_wav = list(missing_wav)


def load_dataset(directory, limit: int | None = None) -> list[Clip]:
    """Load every clip in directory that has both a .wav and a .json
    if Stems missing one half of the pair are skipped and recorded on the returned
    list as .missing_json and .missing_wav
    """
    directory = _resolve(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")

    paired, wav_only, json_only = find_pairs(directory)
    if not paired:
        raise ValueError(
            f"no .wav/.json pairs found in {directory} "
            f"({len(wav_only)} wav without json, {len(json_only)} json without wav)"
        )

    if limit is not None:
        paired = paired[:limit]

    return _ClipList(
        (load_clip(directory / stem) for stem in paired),
        missing_json=wav_only,
        missing_wav=json_only,
    )


def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono signal by linear interpolation
    I could use librosa but this is an opportunity to showcase my understanding of audio signal processing
    """
    if sr == target_sr:
        return x
    # For downsampling we need to do an anti aliasing low pass filter
    # respect the new Nyquist frequency   
    if target_sr < sr: 
        cutoff = 0.5 * target_sr / sr
        n_taps = 101
        t = np.arange(n_taps) - (n_taps - 1) / 2.0
        taps = 2 * cutoff * np.sinc(2 * cutoff * t) * np.hamming(n_taps) 
        taps /= taps.sum()
        x = np.convolve(x, taps, mode="same") 

    n_out = int(round(len(x) * target_sr / sr))
    src = np.arange(len(x), dtype=np.float64)
    dst = np.arange(n_out, dtype=np.float64) * (sr / target_sr)
    return np.interp(dst, src, x)


def read_audio(clip: Clip, target_sr: int = DEFAULT_SR) -> np.ndarray:
    """ Decode a clip to mono float32 at target_sr
    """
    x, sr = sf.read(str(_resolve(clip.wav_path)), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    x = _resample(x.astype(np.float64), sr, target_sr)
    return x.astype(np.float32)


def summarize(clips) -> str:
    """One-line hygiene summary over a list of clips."""
    speakers = {c.speaker_id for c in clips}
    total_minutes = sum(c.duration_s for c in clips) / 60.0
    n_zero = sum(1 for c in clips if c.n_zero_length)
    n_touch = sum(1 for c in clips if c.n_touching)
    n_overlap = sum(1 for c in clips if c.n_real_overlap)
    return (
        f"clips={len(clips)} speakers={len(speakers)} minutes={total_minutes:.1f} "
        f"clips_with_zero_length={n_zero} clips_with_touching={n_touch} "
        f"clips_with_real_overlap={n_overlap}"
    )


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m vadexplore.loader <dataset_dir>", file=sys.stderr)
        return 2

    try:
        clips = load_dataset(argv[0])
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(summarize(clips))

    total_segments = sum(len(c.segments) for c in clips)
    print(f"speech segments:  {total_segments}")
    print(f"zero-length segments: {sum(c.n_zero_length for c in clips)}")
    print(f"touching segment pairs: {sum(c.n_touching for c in clips)}")
    print(f"real overlapping pairs: {sum(c.n_real_overlap for c in clips)}")

    flagged = [c for c in clips if c.warnings]
    print(f"clips with warnings: {len(flagged)}")
    for clip in flagged:
        for warning in clip.warnings:
            print(f"  {clip.stem}: {warning}")

    for label, stems in (("json", clips.missing_json), ("wav", clips.missing_wav)):
        if stems:
            print(f"stems missing .{label}: {len(stems)}")
            for stem in stems:
                print(f"  {stem}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
