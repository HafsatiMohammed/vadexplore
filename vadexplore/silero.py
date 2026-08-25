"""Cross-check the provided labels against Silero VAD.

Silero VAD is a small pretrained neural voice activity detector, MIT licensed,
CPU only, no account and no payment. It is fetched through torch.hub from
snakers4/silero-vad.

The first run needs network access. torch.hub caches the repository and weights
under ~/.cache/torch/hub, and every later run is offline. If the fetch fails
this module raises `SileroUnavailable` with instructions. It never substitutes
a fallback detector or synthetic probabilities: a fabricated second opinion is
worse than none, because the whole point is an independent check.

Audio is high-passed at 80 Hz before Silero sees it, matching the committed
front-end in DECISIONS.md, so both sides of the comparison run on the same
preprocessed signal.

torch and torchaudio only.
"""

from __future__ import annotations

import numpy as np
import torch

from vadexplore.features import DEFAULT_SR, highpass
from vadexplore.labels import DEFAULT_FPS

SILERO_REPO = "snakers4/silero-vad"
SILERO_MODEL = "silero_vad"

# Silero at 16 kHz consumes exactly 512 samples per call and rejects any other
# size, so its native rate is 31.25 windows per second and each window covers
# 32 ms. At 8 kHz the window is 256 samples.
SILERO_WINDOW = {16000: 512, 8000: 256}
NATIVE_RESOLUTION_MS = 32.0

_MODEL = None


class SileroUnavailable(RuntimeError):
    """Raised when the pretrained model cannot be fetched or loaded."""


def load_silero(force_reload: bool = False):
    """Load Silero VAD from the torch.hub cache, fetching it if needed.

    Cached in a module global, since the corpus cross-check loads it once and
    runs it over hundreds of clips.
    """
    global _MODEL
    if _MODEL is not None and not force_reload:
        return _MODEL

    try:
        model, _ = torch.hub.load(
            repo_or_dir=SILERO_REPO,
            model=SILERO_MODEL,
            trust_repo=True,
            onnx=False,
            force_reload=force_reload,
        )
    except Exception as exc:
        raise SileroUnavailable(
            "Could not load Silero VAD from torch.hub.\n"
            f"  repo: {SILERO_REPO}\n"
            f"  cause: {type(exc).__name__}: {exc}\n\n"
            "This is expected when offline and the model is not cached yet.\n"
            "Run once with network access to populate ~/.cache/torch/hub, after "
            "which it works offline:\n"
            "  python -c \"import torch; torch.hub.load('snakers4/silero-vad', "
            "'silero_vad', trust_repo=True)\"\n\n"
            "No fallback detector is substituted, because a fabricated second "
            "opinion would defeat the purpose of the cross-check."
        ) from exc

    model.eval()
    _MODEL = model
    return model


def silero_speech_probs(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    fps: int = DEFAULT_FPS,
    n_frames: int | None = None,
    apply_highpass: bool = True,
) -> np.ndarray:
    """Per-frame Silero speech probability on the project 100 fps grid.

    Returns the raw per-window probabilities resampled onto the frame grid,
    not Silero's post-processed `get_speech_timestamps` output. Comparing at
    the probability level keeps either side's smoothing and hysteresis out of
    the measurement.

    Rate mapping. Silero emits one probability per 512-sample window at 16 kHz,
    so its native grid is 31.25 fps with window k covering [k * 32, (k + 1) * 32)
    ms. Those values are placed at window centers and **linearly interpolated**
    onto the 100 fps frame centers. Interpolation rather than nearest because
    nearest would quantize every threshold crossing to a 32 ms block edge and
    add a sawtooth bias of up to 16 ms to the boundary statistics; a linear ramp
    between two window centers puts a 0.5 crossing at the window boundary, which
    is the best available estimate.

    Interpolation does not create resolution. Silero cannot localize a boundary
    better than its own 32 ms window, so its onsets and offsets carry an
    inherent uncertainty of about +/- 16 ms, which is +/- 1.6 frames on this
    grid. Any boundary bias smaller than that is not meaningful.

    The model is recurrent over windows, so windows are fed sequentially with
    the state reset at the start of each clip. Passing them as a batch would
    give every window a fresh state and produce systematically low probabilities.
    """
    if sr not in SILERO_WINDOW:
        raise ValueError(
            f"Silero supports 8000 and 16000 Hz, got {sr}. Resample with "
            "loader.read_audio before calling."
        )

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"expected mono 1D audio, got shape {x.shape}")

    if apply_highpass:
        x = highpass(x, sr)

    if n_frames is None:
        n_frames = int(round(len(x) / sr * fps))
    n_frames = max(0, int(n_frames))
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    window = SILERO_WINDOW[sr]
    n_windows = max(1, int(np.ceil(len(x) / window)))
    padded = np.pad(x, (0, n_windows * window - len(x)))

    model = load_silero()
    model.reset_states()
    chunks = torch.from_numpy(padded).reshape(n_windows, window)
    with torch.no_grad():
        probs = np.array([float(model(chunks[i], sr)) for i in range(n_windows)],
                         dtype=np.float64)

    window_centers = (np.arange(n_windows) + 0.5) * window / sr
    frame_centers = (np.arange(n_frames) + 0.5) / fps
    return np.interp(frame_centers, window_centers, probs).astype(np.float32)


# --- comparison -----------------------------------------------------------


def _boundaries(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset and offset frame indices of a 0/1 array."""
    padded = np.concatenate(([0], np.asarray(flags).astype(np.int8), [0]))
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _collar_mask(ref: np.ndarray, tol_frames: int) -> np.ndarray:
    """True for frames that should be scored, False inside the boundary collar.

    The collar sits around *reference* boundaries. Excluding frames near them
    removes disagreement that is only about exactly where a transition falls,
    leaving disagreement about whether a region is speech at all.
    """
    keep = np.ones(len(ref), dtype=bool)
    if tol_frames <= 0:
        return keep
    onsets, offsets = _boundaries(ref)
    for index in np.concatenate([onsets, offsets]):
        lo = max(0, int(index) - tol_frames)
        hi = min(len(ref), int(index) + tol_frames)
        keep[lo:hi] = False
    return keep


def _boundary_bias(ref: np.ndarray, hyp: np.ndarray, fps: int, max_dist_frames: int = 50) -> dict:
    """Signed offset of Silero boundaries against reference boundaries, in ms.

    Positive means Silero is late: it starts speech after the reference does,
    or ends it after. Each reference boundary is matched to the nearest
    hypothesis boundary of the same kind within `max_dist_frames`. Unmatched
    reference boundaries are counted rather than scored, since a missing
    boundary is a region disagreement, not a timing one.
    """
    out = {}
    deltas_all = []
    for name, index in (("onset", 0), ("offset", 1)):
        ref_b = _boundaries(ref)[index]
        hyp_b = _boundaries(hyp)[index]
        deltas = []
        unmatched = 0
        for r in ref_b:
            if hyp_b.size == 0:
                unmatched += 1
                continue
            nearest = hyp_b[int(np.argmin(np.abs(hyp_b - r)))]
            if abs(int(nearest) - int(r)) <= max_dist_frames:
                deltas.append((int(nearest) - int(r)) * 1000.0 / fps)
            else:
                unmatched += 1
        deltas_all.extend(deltas)
        out[f"{name}_bias_ms_mean"] = float(np.mean(deltas)) if deltas else None
        out[f"{name}_bias_ms_median"] = float(np.median(deltas)) if deltas else None
        out[f"n_{name}s_matched"] = len(deltas)
        out[f"n_{name}s_unmatched"] = int(unmatched)

    out["bias_ms_mean"] = float(np.mean(deltas_all)) if deltas_all else None
    out["bias_ms_median"] = float(np.median(deltas_all)) if deltas_all else None
    return out


def _scores(ref: np.ndarray, hyp: np.ndarray) -> dict:
    """Accuracy, speech F1, speech IoU, and Cohen's kappa on a frame mask."""
    n = len(ref)
    if n == 0:
        return {"n_frames": 0, "accuracy": None, "f1": None, "iou": None, "kappa": None}

    tp = int(np.sum(ref & hyp))
    fp = int(np.sum(~ref & hyp))
    fn = int(np.sum(ref & ~hyp))
    tn = int(np.sum(~ref & ~hyp))

    accuracy = (tp + tn) / n
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else None

    # Cohen's kappa: agreement corrected for what chance alone would give
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else None

    return {
        "n_frames": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float(accuracy),
        "f1": float(f1) if f1 is not None else None,
        "iou": float(iou) if iou is not None else None,
        "kappa": float(kappa) if kappa is not None else None,
    }


def agreement(
    ref_labels: np.ndarray,
    silero_probs: np.ndarray,
    threshold: float = 0.5,
    tol_frames: int = 0,
    fps: int = DEFAULT_FPS,
) -> dict:
    """Frame agreement between reference labels and thresholded Silero probs.

    Reports the metrics twice: over every frame, and again with a collar of
    `tol_frames` around each reference boundary excluded. The gap between the
    two says how much of the disagreement is only about boundary placement,
    which is expected given that the two methods have different time
    resolutions.
    """
    ref = np.asarray(ref_labels).astype(bool)
    hyp = np.asarray(silero_probs) >= threshold
    if len(ref) != len(hyp):
        raise ValueError(f"length mismatch: ref {len(ref)}, silero {len(hyp)}")

    keep = _collar_mask(ref, tol_frames)
    out = {
        "threshold": float(threshold),
        "tol_frames": int(tol_frames),
        "fps": int(fps),
        "ref_speech_fraction": float(ref.mean()) if len(ref) else None,
        "silero_speech_fraction": float(hyp.mean()) if len(hyp) else None,
        "no_collar": _scores(ref, hyp),
        "with_collar": _scores(ref[keep], hyp[keep]),
        "n_frames_excluded_by_collar": int((~keep).sum()),
    }
    out.update(_boundary_bias(ref, hyp, fps))
    return out


def disagreement_breakdown(
    ref_labels: np.ndarray,
    silero_probs: np.ndarray,
    threshold: float = 0.5,
    tol_frames: int = 2,
    fps: int = DEFAULT_FPS,
) -> dict:
    """Split disagreement into boundary-only and wholesale region disagreement.

    A disagreement run is boundary-only when every frame in it sits inside the
    collar around a reference boundary: the two sides agree that a transition
    happens there and differ only on exactly where. Any run reaching beyond the
    collar is a region disagreement, where one side calls a stretch speech and
    the other calls it silence. The second kind is the one worth reading, since
    it is a candidate label error rather than a method difference.
    """
    ref = np.asarray(ref_labels).astype(bool)
    hyp = np.asarray(silero_probs) >= threshold
    inside_collar = ~_collar_mask(ref, tol_frames)

    wrong = ref != hyp
    onsets, offsets = _boundaries(wrong.astype(np.int8))

    boundary_runs, region_runs = [], []
    for start, end in zip(onsets, offsets):
        run = {
            "start_s": float(start) / fps,
            "end_s": float(end) / fps,
            "n_frames": int(end - start),
            # ref says speech and Silero disagrees means Silero missed speech
            "kind": "silero_missed_speech" if bool(ref[start]) else "silero_extra_speech",
        }
        if inside_collar[start:end].all():
            boundary_runs.append(run)
        else:
            region_runs.append(run)

    return {
        "tol_frames": int(tol_frames),
        "n_boundary_runs": len(boundary_runs),
        "n_region_runs": len(region_runs),
        "boundary_frames": int(sum(r["n_frames"] for r in boundary_runs)),
        "region_frames": int(sum(r["n_frames"] for r in region_runs)),
        "region_runs": region_runs,
    }
