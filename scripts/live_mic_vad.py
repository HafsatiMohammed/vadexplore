"""Real-time microphone VAD with a scrolling display.

Runs the trained streaming causal model on live audio through the exact
training front-end and the streaming inference path. Nothing is reimplemented:
the high-pass, the log-mel, the feature statistics, the model, and the
frame-by-frame session are the same objects the offline evaluation uses.

Threading. The sounddevice callback does one thing: copy the block into a
queue and return. It never touches the model, the feature stack, or
matplotlib. A consumer driven by FuncAnimation drains the queue, advances the
front-end and the model, and redraws. Doing feature work inside the audio
callback is how you get dropouts, and doing matplotlib work there is how you
get a crash.

    python scripts/live_mic_vad.py
    python scripts/live_mic_vad.py --run runs/<name> --window-seconds 8
    python scripts/live_mic_vad.py --input-wav clip.wav --no-plot   # verification

sounddevice, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from vadexplore.config import DataConfig
from vadexplore.data import load_feature_stats
from vadexplore.features import logmel
from vadexplore.labels import DEFAULT_FPS
from vadexplore.model import StreamingVADSession
from vadexplore.preprocess import highpass
from vadexplore.train import load_checkpoint, resolve_device

DEFAULT_RUN = "runs/sweep_past_window/attn_pw_1s"
DEFAULT_STATS = "splits/feature_stats.json"
TARGET_SR = 16000

# The zero-phase high-pass runs forward and backward, so a block filtered in
# isolation has edge transients. Filtering a window with this much margin on
# each side and keeping only the interior reproduces the offline filter
# bit-exactly on this corpus (measured: 100 ms gives max abs diff 0.0, 50 ms
# gives 2e-9). The margin is real latency and is counted as such below.
FILTER_MARGIN_MS = 100.0

SPEECH_COLOR = "#2b6cb0"
BRIDGE_COLOR = "#dd6b20"
PROB_COLOR = "#38a169"
THRESHOLD_COLOR = "#c53030"


def _resolve(path) -> Path:
    return Path(os.path.expanduser(str(path)))


# --- causal post-processing ----------------------------------------------


class CausalPostprocessor:
    """Streaming-safe subset of the offline post-processing.

    Trailing median smoothing looks only at past frames, so it adds no latency,
    but it is not the same operator as the centered median used offline: a
    trailing window reacts to a transition one half-window late. That is the
    price of causality and it is declared rather than hidden.

    Hangover is naturally causal. It fires after an offset is observed and
    extends forward, so it needs no future frames and adds no latency. The
    post-processing sweep chose hangover alone as the best setting, which is
    convenient: the operation that helped most is the one that streams for
    free.

    Minimum speech duration is deliberately not implemented here. Deciding that
    a segment is long enough requires waiting for it to end, which is a delay
    of the full minimum duration. It stays an offline-only operation.
    """

    def __init__(self, threshold: float, smooth_frames: int = 1,
                 hangover_frames: int = 0):
        self.threshold = threshold
        self.smooth_frames = max(1, int(smooth_frames))
        self.hangover_frames = max(0, int(hangover_frames))
        self._history: list[float] = []
        self._countdown = 0

    def step(self, probability: float) -> tuple[float, bool]:
        """One frame in, (smoothed probability, decision) out."""
        self._history.append(float(probability))
        if len(self._history) > self.smooth_frames:
            self._history.pop(0)
        smoothed = float(np.median(self._history))

        raw = smoothed >= self.threshold
        if raw:
            self._countdown = self.hangover_frames
            return smoothed, True
        if self._countdown > 0:
            self._countdown -= 1
            return smoothed, True
        return smoothed, False


# --- the consumer ---------------------------------------------------------


class LiveVAD:
    """Front-end plus streaming model, advanced in batches of new audio."""

    def __init__(self, model, data_config: DataConfig, stats, threshold: float,
                 device, smooth_frames: int = 1, hangover_frames: int = 0,
                 filter_margin_ms: float = FILTER_MARGIN_MS):
        self.model = model
        self.config = data_config
        self.mean, self.std = stats
        self.device = device
        self.fps = data_config.fps
        self.hop = int(round(data_config.sample_rate * data_config.hop_ms / 1000))
        self.win = int(round(data_config.sample_rate * data_config.win_ms / 1000))
        self.margin = int(round(data_config.sample_rate * filter_margin_ms / 1000))

        self.session = StreamingVADSession(model, device=device)
        self.post = CausalPostprocessor(threshold, smooth_frames, hangover_frames)

        self._raw = np.zeros(0, dtype=np.float32)
        self._raw_offset = 0      # absolute sample index of self._raw[0]
        self._next_frame = 0      # next frame index to feed the model

        # rolling outputs, indexed by absolute frame. mel_frames holds the
        # unnormalized log-mel, because the display should show the signal, not
        # the whitened tensor the model consumes.
        self.mel_frames: list[np.ndarray] = []
        self.probabilities: dict[int, float] = {}
        self.decisions: dict[int, bool] = {}
        self.smoothed: dict[int, float] = {}

    # --- the front-end, matching training exactly ---

    def _finalizable_frames(self) -> int:
        """How many frames have enough filtered audio behind and ahead of them."""
        available = self._raw_offset + len(self._raw)
        usable = available - self.margin           # leave the filter its right margin
        if usable < self.win:
            return 0
        return max(0, (usable - self.win) // self.hop + 1)

    def _features(self, first: int, last: int) -> np.ndarray:
        """Normalized log-mel for absolute frames [first, last).

        The signal is filtered over a window with `margin` on both sides and
        only the interior is used, so each frame sees the same filtered samples
        the offline path would have produced for it.

        Normalization uses the SAVED TRAINING STATISTICS. Computing mean and
        variance from the live audio instead would silently rescale the input
        the model was trained on, and the failure is quiet: the model keeps
        producing plausible-looking probabilities that are simply wrong.
        """
        start = first * self.hop
        stop = (last - 1) * self.hop + self.win

        lo = max(0, start - self.margin)
        hi = min(self._raw_offset + len(self._raw), stop + self.margin)
        window = self._raw[lo - self._raw_offset: hi - self._raw_offset]

        filtered = highpass(window, self.config.sample_rate,
                            self.config.highpass_hz, self.config.highpass_order)
        segment = filtered[start - lo: stop - lo]

        mel = logmel(segment, sr=self.config.sample_rate, n_mels=self.config.n_mels,
                     win_ms=self.config.win_ms, hop_ms=self.config.hop_ms,
                     n_frames=last - first).T
        normalized = ((mel - self.mean) / self.std).astype(np.float32)
        return mel.astype(np.float32), normalized

    def push_audio(self, block: np.ndarray) -> list[int]:
        """Add mono 16 kHz audio. Returns the absolute frame indices emitted."""
        self._raw = np.concatenate([self._raw, np.asarray(block, dtype=np.float32)])

        limit = self._finalizable_frames()
        emitted: list[int] = []
        if limit > self._next_frame:
            raw_mel, features = self._features(self._next_frame, limit)
            for i, frame in enumerate(features):
                self.mel_frames.append(raw_mel[i])
                for index, logit in self.session.push(torch.from_numpy(frame)):
                    probability = float(torch.sigmoid(torch.tensor(logit)))
                    self.probabilities[index] = probability
                    smoothed, decision = self.post.step(probability)
                    self.smoothed[index] = smoothed
                    self.decisions[index] = decision
                    emitted.append(index)
            self._next_frame = limit

        # trim consumed audio, keeping enough history for the next window
        keep_from = max(0, self._next_frame * self.hop - self.margin - self.win)
        if keep_from > self._raw_offset:
            self._raw = self._raw[keep_from - self._raw_offset:]
            self._raw_offset = keep_from
        return emitted

    @property
    def n_frames_in(self) -> int:
        return self._next_frame


# --- latency accounting ---------------------------------------------------


def latency_report(block_ms: float, lookahead_frames: int, fps: int,
                   filter_margin_ms: float, hangover_frames: int,
                   smooth_frames: int) -> dict:
    """Every source of delay between a sound and its displayed decision."""
    window_tail_ms = 25.0 - 1000.0 / fps      # the 25 ms window past the 10 ms hop
    lookahead_ms = lookahead_frames * 1000.0 / fps
    smoothing_ms = (smooth_frames - 1) * 1000.0 / fps   # trailing, so it lags
    total = block_ms + filter_margin_ms + window_tail_ms + lookahead_ms + smoothing_ms
    return {
        "audio_block_ms": block_ms,
        "highpass_margin_ms": filter_margin_ms,
        "analysis_window_tail_ms": window_tail_ms,
        "model_lookahead_ms": lookahead_ms,
        "trailing_smoothing_ms": smoothing_ms,
        "hangover_ms": hangover_frames * 1000.0 / fps,
        "hangover_adds_latency": False,
        "total_ms": total,
        "note": ("hangover extends a decision forward in time, so it costs no "
                 "latency; minimum speech duration would cost its full duration "
                 "and is left offline"),
    }


# --- display --------------------------------------------------------------


def build_display(live: LiveVAD, window_seconds: float, threshold: float,
                  title: str, redraw_ms: int):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec

    n_view = int(round(window_seconds * live.fps))
    n_mels = live.config.n_mels

    fig = plt.figure(figsize=(12, 7))
    grid = GridSpec(3, 2, figure=fig, height_ratios=[3.0, 1.4, 0.5],
                    width_ratios=[1, 0.014], hspace=0.14, wspace=0.015)
    mel_ax = fig.add_subplot(grid[0, 0])
    prob_ax = fig.add_subplot(grid[1, 0], sharex=mel_ax)
    band_ax = fig.add_subplot(grid[2, 0], sharex=mel_ax)

    mel_buffer = np.full((n_mels, n_view), -12.0, dtype=np.float32)
    prob_buffer = np.full(n_view, np.nan, dtype=np.float32)
    band_buffer = np.zeros((1, n_view), dtype=np.float32)
    times = np.arange(n_view) / live.fps - window_seconds

    image = mel_ax.imshow(mel_buffer, origin="lower", aspect="auto",
                          interpolation="nearest", cmap="magma",
                          extent=(-window_seconds, 0.0, 0.0, n_mels), vmin=-12, vmax=4)
    mel_ax.set_ylabel("mel bin", fontsize=9)
    mel_ax.tick_params(labelsize=8, labelbottom=False)
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 1]))
    bar.set_label("log mel energy", fontsize=7)
    bar.ax.tick_params(labelsize=6)

    (prob_line,) = prob_ax.plot(times, prob_buffer, color=PROB_COLOR, linewidth=1.6)
    prob_ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2)
    prob_ax.annotate(f"threshold {threshold:.3f}", (-window_seconds, threshold),
                     xytext=(6, 4), textcoords="offset points", fontsize=8,
                     color=THRESHOLD_COLOR)
    prob_ax.set_ylim(-0.03, 1.03)
    prob_ax.set_ylabel("P(speech)", fontsize=9)
    prob_ax.tick_params(labelsize=8, labelbottom=False)
    for side in ("top", "right"):
        prob_ax.spines[side].set_visible(False)

    band = band_ax.imshow(band_buffer, origin="lower", aspect="auto",
                          interpolation="nearest",
                          cmap=ListedColormap(["#ffffff", SPEECH_COLOR]),
                          extent=(-window_seconds, 0.0, 0.0, 1.0), vmin=0, vmax=1)
    band_ax.set_yticks([])
    band_ax.set_ylabel("speech", rotation=0, ha="right", va="center", fontsize=9,
                       labelpad=8)
    band_ax.set_xlabel("seconds behind now", fontsize=9)
    band_ax.tick_params(labelsize=8)
    for side in ("top", "right", "left"):
        band_ax.spines[side].set_visible(False)

    fig.suptitle(title, fontsize=10.5, y=0.985)

    state = {"drawn": 0}

    def update(_):
        newest = live.n_frames_in
        if newest == state["drawn"]:
            return image, prob_line, band
        state["drawn"] = newest

        first = max(0, newest - n_view)
        count = newest - first
        # roll the buffers left by the number of new frames, then fill the edge
        if count > 0:
            mel = np.stack(live.mel_frames[first:newest], axis=1)
            mel_buffer[:, n_view - count:] = mel[:, -count:]
            mel_buffer[:, :n_view - count] = -12.0

            probs = np.array([live.smoothed.get(i, np.nan) for i in range(first, newest)])
            prob_buffer[n_view - count:] = probs[-count:]
            prob_buffer[:n_view - count] = np.nan

            flags = np.array([1.0 if live.decisions.get(i) else 0.0
                              for i in range(first, newest)])
            band_buffer[0, n_view - count:] = flags[-count:]
            band_buffer[0, :n_view - count] = 0.0

        image.set_data(mel_buffer)
        prob_line.set_ydata(prob_buffer)
        band.set_data(band_buffer)
        return image, prob_line, band

    animation = FuncAnimation(fig, update, interval=redraw_ms,
                              blit=False, cache_frame_data=False)
    return fig, animation


# --- main -----------------------------------------------------------------


def load_threshold(run_dir: Path, target_fa: float, fallback: float = 0.5) -> tuple:
    """Read the validation-chosen operating point from the evaluation report."""
    for split in ("test", "val"):
        path = run_dir / f"eval_{split}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        primary = report.get("primary_convention") or report["trained_on_convention"]
        for point in report["conventions"][primary]["operating_points"]:
            if abs(point["target_fa_per_hour"] - target_fa) < 1e-9:
                return float(point["threshold"]), f"{path.name} ({point['threshold_chosen_on']})"
    return fallback, "fallback default, no eval report found"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-fa-per-hour", type=float, default=100.0,
                        dest="target_fa")
    parser.add_argument("--window-seconds", type=float, default=6.0,
                        dest="window_seconds")
    parser.add_argument("--block-ms", type=float, default=30.0, dest="block_ms")
    parser.add_argument("--redraw-ms", type=int, default=40, dest="redraw_ms",
                        help="animation interval; 40 ms is 25 fps")
    parser.add_argument("--smooth-ms", type=float, default=0.0, dest="smooth_ms",
                        help="trailing median on the posterior, causal")
    parser.add_argument("--hangover-ms", type=float, default=100.0,
                        dest="hangover_ms",
                        help="causal, costs no latency; 100 ms won the offline sweep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--record", default=None, help="write the captured audio here")
    parser.add_argument("--log", default=None, help="write per-frame output here")
    parser.add_argument("--input-wav", default=None, dest="input_wav",
                        help="replay a file instead of the microphone, for verification")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot")
    args = parser.parse_args(argv)

    run_dir = _resolve(args.run)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(run_dir / "best.pt", device=device)
    data_config = DataConfig(**payload["data_config"])

    if payload["model_config"]["temporal"] != "causal_attn":
        print(f"error: {payload.get('run_name')} uses the "
              f"{payload['model_config']['temporal']} core, which is not streamable.\n"
              "  Point --run at a causal_attn checkpoint.", file=sys.stderr)
        return 2

    stats = load_feature_stats(args.stats)
    threshold, threshold_source = (
        (args.threshold, "command line") if args.threshold is not None
        else load_threshold(run_dir, args.target_fa))

    fps = data_config.fps
    smooth_frames = max(1, int(round(args.smooth_ms * fps / 1000)))
    hangover_frames = max(0, int(round(args.hangover_ms * fps / 1000)))

    live = LiveVAD(model, data_config, stats, threshold, device,
                   smooth_frames=smooth_frames, hangover_frames=hangover_frames)
    latency = latency_report(args.block_ms, live.session.emission_delay_frames, fps,
                             FILTER_MARGIN_MS, hangover_frames, smooth_frames)

    past = model.effective_past_window_frames
    print(f"live VAD: {payload.get('run_name')} ({payload['model_config']['temporal']})")
    print(f"  device          {device}")
    print(f"  past window     {past if past is not None else 'unbounded'} frames"
          + (f" ({past / fps:g} s effective)" if past is not None else ""))
    print(f"  lookahead       {live.session.emission_delay_frames} frames "
          f"({latency['model_lookahead_ms']:.0f} ms)")
    print(f"  threshold       {threshold:.4f}  (from {threshold_source})")
    print(f"  post-processing trailing median {args.smooth_ms:g} ms, "
          f"causal hangover {args.hangover_ms:g} ms")
    print(f"                  minimum speech duration is offline only, it would cost "
          f"its full duration in latency")
    print(f"  front-end       {data_config.highpass_hz:g} Hz zero-phase high-pass, "
          f"{data_config.n_mels} log-mel, saved training statistics")
    print(f"\n  end-to-end latency {latency['total_ms']:.0f} ms")
    for key in ("audio_block_ms", "highpass_margin_ms", "analysis_window_tail_ms",
                "model_lookahead_ms", "trailing_smoothing_ms"):
        print(f"    {key.replace('_', ' '):26s} {latency[key]:6.1f} ms")
    print()

    captured: list[np.ndarray] = []
    blocks: queue.Queue = queue.Queue()
    stop = threading.Event()

    if args.input_wav:
        import soundfile as sf
        audio, file_sr = sf.read(_resolve(args.input_wav), dtype="float32",
                                 always_2d=True)
        audio = audio.mean(axis=1)
        if file_sr != TARGET_SR:
            from vadexplore.loader import _resample
            audio = _resample(audio.astype(np.float64), file_sr, TARGET_SR).astype(np.float32)
        step = int(TARGET_SR * args.block_ms / 1000)
        for i in range(0, len(audio), step):
            blocks.put(audio[i:i + step].copy())
        print(f"  replaying {args.input_wav} ({len(audio) / TARGET_SR:.2f} s, "
              f"{blocks.qsize()} blocks)\n")
        stream = None
    else:
        import sounddevice as sd

        def callback(indata, frames_count, time_info, status):
            """Audio thread. Copy and return. No features, no model, no drawing."""
            if status:
                print(f"  audio status: {status}", file=sys.stderr)
            blocks.put(indata[:, 0].copy())

        stream = sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                                blocksize=int(TARGET_SR * args.block_ms / 1000),
                                callback=callback)

    def drain() -> None:
        while True:
            try:
                block = blocks.get_nowait()
            except queue.Empty:
                return
            if args.record:
                captured.append(block)
            live.push_audio(block)

    def finish() -> None:
        if args.record:
            import soundfile as sf
            path = _resolve(args.record)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), np.concatenate(captured) if captured
                     else np.zeros(0, np.float32), TARGET_SR)
            print(f"  wrote {path}")
        if args.log:
            path = _resolve(args.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "run": payload.get("run_name"),
                "threshold": threshold,
                "threshold_source": threshold_source,
                "fps": fps,
                "latency": latency,
                "frames": [
                    {"frame": i, "t_s": i / fps,
                     "probability": live.probabilities[i],
                     "smoothed": live.smoothed[i],
                     "decision": bool(live.decisions[i])}
                    for i in sorted(live.probabilities)
                ],
            }, indent=2))
            print(f"  wrote {path}")

    try:
        if stream is not None:
            stream.start()
        if args.no_plot:
            while not stop.is_set():
                drain()
                if args.input_wav and blocks.empty():
                    break
                time.sleep(0.005)
            emitted = live.session.finish()
            for index, logit in emitted:
                probability = float(torch.sigmoid(torch.tensor(logit)))
                live.probabilities[index] = probability
                smoothed, decision = live.post.step(probability)
                live.smoothed[index] = smoothed
                live.decisions[index] = decision
            speech = sum(1 for v in live.decisions.values() if v)
            print(f"  {len(live.probabilities)} frames, {speech} called speech "
                  f"({speech / max(len(live.decisions), 1) * 100:.1f}%)")
        else:
            import matplotlib.pyplot as plt
            title = (f"{payload.get('run_name')}   live   "
                     f"{latency['total_ms']:.0f} ms end to end   "
                     f"threshold {threshold:.3f}")
            fig, animation = build_display(live, args.window_seconds, threshold,
                                           title, args.redraw_ms)
            timer = fig.canvas.new_timer(interval=10)
            timer.add_callback(drain)
            timer.start()
            print("  showing the live view. Ctrl-C or close the window to stop.\n")
            plt.show()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        stop.set()
        if stream is not None:
            stream.stop()
            stream.close()
        finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
