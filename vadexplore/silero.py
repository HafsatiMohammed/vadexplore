"""Silero VAD wrapper and label-agreement metrics. Needs network on first run."""

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
    """Load Silero VAD from the torch.hub cache, fetching it if needed."""
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
            f"could not load {SILERO_REPO} from torch.hub: "
            f"{type(exc).__name__}: {exc}"
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
    """Per-frame Silero speech probability on the 100 fps grid.

    Silero runs at 31.25 fps; values are linearly interpolated onto frame centres,
    which does not create resolution: its boundaries carry about +/- 16 ms.
    Windows are fed sequentially with state reset per clip, never batched.
    """
    if sr not in SILERO_WINDOW:
        raise ValueError(f"Silero supports 8000 and 16000 Hz, got {sr}")

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


def _boundaries(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset and offset frame indices of a 0/1 array."""
    padded = np.concatenate(([0], np.asarray(flags).astype(np.int8), [0]))
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _collar_mask(ref: np.ndarray, tol_frames: int) -> np.ndarray:
    """True for frames that should be scored, False inside the collar.

    The collar sits around reference boundaries.
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
    """Signed offset of Silero boundaries against the reference, in ms.

    Positive means Silero is late. Unmatched reference boundaries are counted,
    not scored.
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

    Reported with and without a tol_frames collar; the gap between the two is how
    much of the disagreement is only boundary placement.
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
    """Split disagreement into boundary-only and region runs.

    A run inside the collar is boundary-only; one reaching past it is a region
    disagreement, which is the candidate label error.
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
