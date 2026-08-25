"""Frame-level VAD model: shared frontend, swappable temporal core, shared head.

The point of the split is that a run differs from another run in exactly one
place. The convolutional frontend and the per-frame head are identical objects
for both cores, so any accuracy gap between `bigru` and `causal_attn` is
attributable to the temporal block and nothing else.

This model is not autoregressive: the output at frame t is never fed back as
input. That has a consequence worth stating early, because it decides how the
bounded-past attention is used. Once a past window is imposed, a batch forward
pass with the windowed causal mask computes exactly what a frame-by-frame loop
would compute, so evaluation runs through the ordinary masked forward and the
streaming loop in `StreamingVADSession` is a verification and measurement tool,
not a required path. The load-bearing part of the window is that it applies
during training, so the model learns on the same amount of history it will have
at deployment.

Input is the collate output from `vadexplore.data`: features (B, T, n_mels),
a boolean mask (B, T) that is True on real frames, and lengths (B,). Output is
per-frame speech logits (B, T), with no sigmoid applied, so the caller pairs it
with `binary_cross_entropy_with_logits`.

torch only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vadexplore.config import DEFAULT_MODEL_CONFIG, ModelConfig

TEMPORAL_CORES = ("bigru", "causal_attn")


class MaskedBatchNorm2d(nn.Module):
    """BatchNorm2d whose statistics ignore padded frames.

    Plain BatchNorm would fold the padding zeros into the batch mean and
    variance, so a real frame's output would depend on how much padding
    happened to sit beside it in the batch. That is precisely the thing the
    masking is supposed to prevent, and it is invisible at eval time because
    running statistics are frozen, which makes it an easy bug to ship.

    Statistics still depend on the other clips in the batch. That is ordinary
    BatchNorm behavior, not a padding leak.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x is (B, C, T, F); mask is (B, T) True on real frames
        weights = mask[:, None, :, None].to(x.dtype)

        if self.training:
            count = weights.sum() * x.shape[3]
            total = (x * weights).sum(dim=(0, 2, 3))
            mean = total / count.clamp(min=1.0)
            centered = (x - mean[None, :, None, None]) * weights
            var = (centered ** 2).sum(dim=(0, 2, 3)) / count.clamp(min=1.0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean, var = self.running_mean, self.running_var

        normalized = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ConvFrontend(nn.Module):
    """Conv blocks over the time-frequency map, time resolution preserved.

    Every convolution is stride 1 in time and pooling only ever touches the mel
    axis, so the output has exactly as many frames as the input. Padded frames
    are zeroed on the way in and on the way out, which together with the
    masked normalization keeps padding out of every real frame's receptive
    field.

    With `causal_frontend` the time padding goes entirely on the left, so a
    frame's features depend only on itself and earlier frames. This is what
    makes the causal cores causal end to end: with symmetric padding a strictly
    causal attention stack would still read `len(conv_channels)` frames of
    future through the convolutions. It costs the offline BiGRU almost nothing,
    since the recurrence supplies future context anyway.
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.causal = config.causal_frontend
        self.kernel = 3

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = 1
        mel = config.n_mels
        for out_channels in config.conv_channels:
            # time padding is applied by hand in forward, so that the causal
            # variant can pad on the left only; the mel axis is padded here
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel,
                                        stride=1, padding=(0, 1)))
            self.norms.append(MaskedBatchNorm2d(out_channels))
            in_channels = out_channels
            mel = mel // 2  # pooling over the mel axis only

        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.project = nn.Linear(in_channels * mel, config.d_model)
        self.out_dim = config.d_model

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = features * mask[:, :, None].to(features.dtype)
        x = x.unsqueeze(1)  # (B, 1, T, n_mels)

        frame_mask = mask[:, None, :, None].to(x.dtype)
        total_pad = self.kernel - 1
        time_pad = (total_pad, 0) if self.causal else (total_pad // 2, total_pad - total_pad // 2)
        for conv, norm in zip(self.convs, self.norms):
            x = F.pad(x, (0, 0, time_pad[0], time_pad[1]))
            x = conv(x)
            x = norm(x, mask)
            x = F.relu(x)
            x = self.pool(x)
            # Re-zero the padding after every block. The conv bias and the norm
            # bias leave a nonzero constant on padded positions, and the next
            # block's convolution would read it. Run alone the same clip gets
            # zero-padding from the convolution itself, so without this the two
            # paths disagree at the final real frames.
            x = x * frame_mask

        batch, channels, frames, mel = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, frames, channels * mel)
        x = self.project(x)
        return x * mask[:, :, None].to(x.dtype)


class BiGRUCore(nn.Module):
    """Recurrent core. Bidirectional by default, unidirectional when causal.

    Sequences are packed, so the recurrence literally never steps over a padded
    frame and the backward direction starts at each clip's true final frame
    rather than at the end of the batch.
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.bidirectional = not config.causal
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        directions = 2 if self.bidirectional else 1
        raw_dim = config.gru_hidden * directions
        # keep the core's output at d_model so the head is shared unchanged
        self.project = (nn.Identity() if raw_dim == config.d_model
                        else nn.Linear(raw_dim, config.d_model))
        self.out_dim = config.d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # pack_padded_sequence requires the lengths on CPU even when x is on MPS or CUDA
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.detach().cpu().to(torch.int64), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.shape[1])
        return self.project(out) * mask[:, :, None].to(out.dtype)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal encoding, computed on the fly for any length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        inverse = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / d_model))
        self.register_buffer("inverse_frequency", inverse)

    def at(self, index: int, device, dtype) -> torch.Tensor:
        """Encoding for one absolute frame index, shape (d_model,).

        Streaming needs the same vector the batch path would have placed at
        this position, so both call into the same formula.
        """
        angles = torch.tensor([float(index)], device=device) * self.inverse_frequency.to(device)
        encoding = torch.zeros(self.d_model, device=device, dtype=dtype)
        encoding[0::2] = torch.sin(angles).to(dtype)
        encoding[1::2] = torch.cos(angles).to(dtype)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None]
        angles = position * self.inverse_frequency[None, :].to(x.device)
        encoding = torch.zeros(x.shape[1], self.d_model, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        return x + encoding[None, :, :]


def causal_attention_mask(
    n_frames: int,
    lookahead: int,
    device,
    past_window: int | None = None,
) -> torch.Tensor:
    """True where attention is forbidden. Query i may read keys in
    [i - past_window, i + lookahead].

    `lookahead == 0` is strictly causal. A positive value admits a bounded
    window of future frames per layer, which is the accuracy against latency
    knob.

    `past_window` adds the lower bound. None leaves the past unbounded, which
    is the original behavior and is fine for batch evaluation over finite
    clips, but it cannot stream indefinitely at constant memory because the
    key and value cache grows with the stream.

    Both budgets compose over depth. See `VADModel.lookahead_frames` and
    `VADModel.effective_past_window_frames` for the end-to-end figures, which
    are these values times the number of attention layers.
    """
    index = torch.arange(n_frames, device=device)
    delta = index[None, :] - index[:, None]  # key index minus query index
    blocked = delta > lookahead
    if past_window is not None:
        blocked = blocked | (delta < -int(past_window))
    return blocked


class CausalAttentionCore(nn.Module):
    """Masked self-attention stack, streaming-capable by construction.

    Each layer combines two masks. The causal mask stops a frame seeing further
    ahead than its lookahead budget and, when `past_window_frames` is set,
    further back than its past budget. The key padding mask stops any frame
    attending to padded positions. Both are needed: the causal mask alone would
    still let a late real frame read padding, and the padding mask alone would
    leak the future.

    `past_window_frames` is applied here, in the single forward path that both
    training and evaluation use. That is the point of it: a model trained with
    every frame attending to the whole clip has learned to rely on a history
    length it will not have behind a deployment window, and its accuracy drops
    when the window is imposed later. Train and deploy must use the same value.

    A bounded past also makes `StreamingVADSession` possible, but that is a
    secondary benefit. Since the model is not autoregressive, this masked
    forward already equals the frame-by-frame result.
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.lookahead = int(config.lookahead_frames)
        self.past_window = (None if config.past_window_frames is None
                            else int(config.past_window_frames))
        self.positional = SinusoidalPositionalEncoding(config.d_model)

        self.attention = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        self.feed_forward = nn.ModuleList()
        self.ff_norms = nn.ModuleList()
        hidden = config.d_model * config.attn_ff_ratio
        for _ in range(config.attn_layers):
            self.attention.append(nn.MultiheadAttention(
                config.d_model, config.attn_heads,
                dropout=config.attn_dropout, batch_first=True))
            self.attention_norms.append(nn.LayerNorm(config.d_model))
            self.feed_forward.append(nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.ReLU(),
                nn.Dropout(config.attn_dropout),
                nn.Linear(hidden, config.d_model),
            ))
            self.ff_norms.append(nn.LayerNorm(config.d_model))
        self.dropout = nn.Dropout(config.attn_dropout)
        self.out_dim = config.d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pad_mask = ~mask  # True marks a key that must not be attended to
        attn_mask = causal_attention_mask(
            x.shape[1], self.lookahead, x.device, self.past_window)

        x = self.positional(x)
        x = x * mask[:, :, None].to(x.dtype)

        for attention, attn_norm, feed_forward, ff_norm in zip(
                self.attention, self.attention_norms, self.feed_forward, self.ff_norms):
            # pre-norm residual blocks, which train more stably at this depth
            normed = attn_norm(x)
            attended, _ = attention(
                normed, normed, normed,
                attn_mask=attn_mask,
                key_padding_mask=pad_mask,
                need_weights=False,
            )
            # a fully masked query would come back as NaN; padded queries are
            # discarded anyway, so neutralize them before they poison the residual
            attended = torch.nan_to_num(attended, nan=0.0)
            x = x + self.dropout(attended)
            x = x + feed_forward(ff_norm(x))
            x = x * mask[:, :, None].to(x.dtype)

        return x


class FrameHead(nn.Module):
    """LayerNorm then a linear projection to one logit per frame."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x)).squeeze(-1)


def build_core(config: ModelConfig) -> nn.Module:
    if config.temporal == "bigru":
        return BiGRUCore(config)
    if config.temporal == "causal_attn":
        return CausalAttentionCore(config)
    raise ValueError(f"temporal must be one of {TEMPORAL_CORES}, got {config.temporal!r}")


class VADModel(nn.Module):
    """Frame-level VAD: shared frontend, selected temporal core, shared head."""

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG):
        super().__init__()
        if config.temporal not in TEMPORAL_CORES:
            raise ValueError(f"temporal must be one of {TEMPORAL_CORES}, got {config.temporal!r}")
        self.config = config
        self.frontend = ConvFrontend(config)
        self.core = build_core(config)
        self.head = FrameHead(self.core.out_dim)

    @property
    def lookahead_frames(self) -> int:
        """Frames of future a real frame's output can depend on.

        The frontend contributes `len(conv_channels)` frames unless it is
        causal, and a bidirectional core contributes an unbounded amount.
        """
        frontend = 0 if self.config.causal_frontend else len(self.config.conv_channels)
        if self.config.temporal == "bigru":
            return frontend if self.config.causal else -1  # -1 means unbounded
        # Attention layers compose: layer 2 reads layer 1 outputs up to
        # t + lookahead, and each of those already read inputs up to
        # t + 2 * lookahead. The end-to-end latency is therefore per-layer
        # lookahead times depth, which is what a deployment budget must use.
        return frontend + int(self.config.lookahead_frames) * int(self.config.attn_layers)

    @property
    def effective_past_window_frames(self) -> int | None:
        """End-to-end past context in frames, or None when unbounded.

        Composes over depth exactly as the lookahead does. Layer 2 reads layer 1
        outputs back to t - W, and each of those already read inputs back to
        t - 2W, so the true context is the per-layer window times the number of
        attention layers. Quote this figure, not the per-layer one, when
        stating how much history the model uses or how much memory a stream
        needs.

        None for the BiGRU core: a recurrence carries unbounded history by
        construction, and `past_window_frames` is an attention-only setting
        that it ignores.
        """
        if self.config.temporal != "causal_attn":
            return None
        if self.config.past_window_frames is None:
            return None
        return int(self.config.past_window_frames) * int(self.config.attn_layers)

    @property
    def frontend_context_frames(self) -> int:
        """Past frames the convolutional frontend needs for one output frame."""
        return len(self.config.conv_channels) * (self.frontend.kernel - 1)

    @property
    def is_causal(self) -> bool:
        """True when no real frame's output depends on any later frame."""
        if not self.config.causal_frontend:
            return False
        if self.config.temporal == "bigru":
            return bool(self.config.causal)
        return self.config.lookahead_frames == 0

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits (B, T), aligned frame for frame with the labels."""
        if features.dim() != 3:
            raise ValueError(f"expected (batch, frames, n_mels), got {tuple(features.shape)}")

        batch, frames, _ = features.shape
        if mask is None:
            mask = torch.ones(batch, frames, dtype=torch.bool, device=features.device)
        if lengths is None:
            lengths = mask.sum(dim=1)

        x = self.frontend(features, mask)
        x = self.core(x, mask, lengths)
        logits = self.head(x)
        return logits * mask.to(logits.dtype)

    def forward_batch(self, batch: dict) -> torch.Tensor:
        """Convenience wrapper over the collate output."""
        return self(batch["features"], batch["mask"], batch["lengths"])


# --- reporting ------------------------------------------------------------


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_report(config: ModelConfig = DEFAULT_MODEL_CONFIG) -> dict:
    """Parameter counts split into the shared parts and the swappable core."""
    model = VADModel(config)
    return {
        "temporal": config.temporal,
        "frontend": count_parameters(model.frontend),
        "core": count_parameters(model.core),
        "head": count_parameters(model.head),
        "total": count_parameters(model),
    }


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross entropy over real frames only.

    Padded positions carry the ignore value rather than a class, so they are
    dropped before the loss instead of being weighted to zero afterwards.
    """
    valid = mask.reshape(-1)
    flat_logits = logits.reshape(-1)[valid]
    flat_labels = labels.reshape(-1)[valid].to(flat_logits.dtype)
    return F.binary_cross_entropy_with_logits(
        flat_logits, flat_labels, pos_weight=pos_weight)


# --- streaming inference --------------------------------------------------


def _project_qkv(attention: nn.MultiheadAttention, x: torch.Tensor):
    """Split one frame through a MultiheadAttention's packed input projection.

    Returns q, k, v each shaped (n_heads, head_dim). Reimplemented here because
    `nn.MultiheadAttention` has no incremental interface; the weights are the
    module's own, so the arithmetic stays identical to the batch path.
    """
    embed = attention.embed_dim
    heads = attention.num_heads
    head_dim = embed // heads

    projected = torch.nn.functional.linear(x, attention.in_proj_weight, attention.in_proj_bias)
    q, k, v = projected.split(embed, dim=-1)
    return (q.view(heads, head_dim), k.view(heads, head_dim), v.view(heads, head_dim))


def _attend(attention: nn.MultiheadAttention, q, keys, values) -> torch.Tensor:
    """Single-query attention over a cache. q is (H, D); keys/values are (H, N, D)."""
    head_dim = q.shape[-1]
    scores = (keys * q[:, None, :]).sum(-1) / (head_dim ** 0.5)  # (H, N)
    weights = torch.softmax(scores, dim=-1)
    context = (weights[:, :, None] * values).sum(dim=1)  # (H, D)
    return attention.out_proj(context.reshape(-1))


class StreamingVADSession:
    """Frame-by-frame inference with a fixed-size key and value cache.

    Not required to produce streaming-faithful metrics. Because the model is
    not autoregressive, the masked batch forward with the same
    `past_window_frames` and `lookahead_frames` computes numerically identical
    per-frame logits, and that is what every reported evaluation number uses.
    This class exists to prove that equivalence and to measure real per-frame
    latency and constant memory, not to serve them.

    Only the `causal_attn` core streams. The BiGRU core is bidirectional in its
    default form and, even when made unidirectional, carries unbounded state,
    so a windowed cache means nothing for it.

    Memory. Per attention layer the cache holds at most
    `past_window_frames + lookahead_frames + 1` entries, each one key and one
    value of `d_model` floats, so the whole cache is

        (past_window + lookahead + 1) * attn_layers * n_heads * head_dim * 2

    floats, plus a frontend buffer of `frontend_context_frames + 1` input
    frames. All of that is independent of how long the stream runs. With
    `past_window_frames=None` the cache grows without bound, which is exactly
    the failure the window exists to prevent; the session still runs so the
    two can be compared, but it is not deployable.

    Latency. A frame is emitted once its lookahead budget has been filled at
    every layer, so the emission delay is `lookahead_frames * attn_layers`
    frames. At the end of a stream, `finish()` releases the tail frames using
    whatever future actually exists, which is what the batch path does too
    since its padding mask blocks keys past the final frame.
    """

    def __init__(self, model: "VADModel", device=None):
        if model.config.temporal != "causal_attn":
            raise ValueError(
                "streaming requires the causal_attn core; "
                f"got temporal={model.config.temporal!r}. The BiGRU core is "
                "bidirectional by default and keeps unbounded recurrent state, "
                "so past_window_frames does not apply to it."
            )
        if model.training:
            raise ValueError("call model.eval() before streaming; dropout and batch "
                             "statistics must be frozen for streaming to match batching")

        self.model = model
        self.core = model.core
        self.device = device or next(model.parameters()).device
        self.n_layers = len(self.core.attention)
        self.lookahead = self.core.lookahead
        self.past_window = self.core.past_window

        self.frontend_context = model.frontend_context_frames
        self._buffer: list[torch.Tensor] = []   # recent raw input frames
        self._buffer_valid: list[bool] = []

        # per layer: absolute index -> representation, and -> (k, v)
        self._features: list[dict[int, torch.Tensor]] = [{} for _ in range(self.n_layers + 1)]
        self._cache: list[dict[int, tuple]] = [{} for _ in range(self.n_layers)]
        self._next_query = [0] * self.n_layers
        self._available = [-1] * (self.n_layers + 1)

        self._n_pushed = 0
        self._next_emit = 0
        self._finished = False

    # --- introspection ---

    @property
    def cached_entries(self) -> int:
        """Live key and value pairs across all layers, for the memory claim."""
        return sum(len(layer) for layer in self._cache)

    @property
    def emission_delay_frames(self) -> int:
        return self.lookahead * self.n_layers

    # --- driving the stream ---

    def push(self, frame: torch.Tensor) -> list[tuple[int, float]]:
        """Feed one frame of shape (n_mels,). Returns whatever became emittable.

        Not needed for streaming-faithful evaluation: the windowed batch
        forward is numerically identical for this non-autoregressive model.
        """
        if self._finished:
            raise RuntimeError("session already finished")
        frame = torch.as_tensor(frame, dtype=torch.float32, device=self.device).reshape(-1)

        self._buffer.append(frame)
        self._buffer_valid.append(True)
        keep = self.frontend_context + 1
        if len(self._buffer) > keep:
            self._buffer = self._buffer[-keep:]
            self._buffer_valid = self._buffer_valid[-keep:]

        index = self._n_pushed
        self._n_pushed += 1
        self._features[0][index] = self._frontend_frame(index)
        self._available[0] = index
        return self._advance()

    def finish(self) -> list[tuple[int, float]]:
        """Close the stream and release the tail frames still inside the delay."""
        self._finished = True
        return self._advance()

    def run(self, features: torch.Tensor) -> torch.Tensor:
        """Stream a whole (n_frames, n_mels) clip and return logits (n_frames,).

        Not needed for streaming-faithful evaluation: for this
        non-autoregressive model the windowed batch forward returns the same
        logits, so use it instead and keep this for equivalence checks and
        latency measurement.
        """
        emitted: list[tuple[int, float]] = []
        for i in range(features.shape[0]):
            emitted.extend(self.push(features[i]))
        emitted.extend(self.finish())

        logits = torch.zeros(features.shape[0], device=self.device)
        for index, value in emitted:
            logits[index] = value
        if len(emitted) != features.shape[0]:
            raise RuntimeError(f"streamed {len(emitted)} logits for "
                               f"{features.shape[0]} frames")
        return logits

    # --- internals ---

    def _frontend_frame(self, index: int) -> torch.Tensor:
        """Run the shared frontend over the rolling buffer, keep the last frame.

        The buffer carries `frontend_context` real past frames so the causal
        convolutions see the same history the batch path would give them. At
        the very start of a stream the missing history is marked invalid rather
        than zero-filled, because the frontend re-zeros masked positions between
        blocks and a valid-but-zero frame would pick up the conv and norm biases
        instead of staying at zero.
        """
        needed = self.frontend_context + 1
        pad = needed - len(self._buffer)
        frames = ([torch.zeros_like(self._buffer[0])] * pad) + self._buffer
        valid = ([False] * pad) + self._buffer_valid

        window = torch.stack(frames).unsqueeze(0)
        mask = torch.tensor(valid, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model.frontend(window, mask)[0, -1]
        return out + self.core.positional.at(index, self.device, out.dtype)

    def _key_range(self, layer: int, query: int) -> range:
        low = 0 if self.past_window is None else max(0, query - self.past_window)
        high = min(query + self.lookahead, self._available[layer])
        return range(low, high + 1)

    def _evict(self, layer: int) -> None:
        if self.past_window is None:
            return
        oldest = max(0, self._next_query[layer] - self.past_window)
        for store in (self._cache[layer], self._features[layer]):
            for index in [i for i in store if i < oldest]:
                del store[index]

    def _advance(self) -> list[tuple[int, float]]:
        with torch.no_grad():
            for layer in range(self.n_layers):
                attention = self.core.attention[layer]
                attn_norm = self.core.attention_norms[layer]
                feed_forward = self.core.feed_forward[layer]
                ff_norm = self.core.ff_norms[layer]

                while True:
                    query = self._next_query[layer]
                    if query > self._available[layer]:
                        break
                    if not self._finished and self._available[layer] < query + self.lookahead:
                        break

                    # project any newly available frames into the cache once
                    for index in self._key_range(layer, query):
                        if index not in self._cache[layer]:
                            normed = attn_norm(self._features[layer][index])
                            _, k, v = _project_qkv(attention, normed)
                            self._cache[layer][index] = (k, v)

                    indices = list(self._key_range(layer, query))
                    keys = torch.stack([self._cache[layer][i][0] for i in indices], dim=1)
                    values = torch.stack([self._cache[layer][i][1] for i in indices], dim=1)

                    own = self._features[layer][query]
                    q, _, _ = _project_qkv(attention, attn_norm(own))
                    attended = _attend(attention, q, keys, values)

                    hidden = own + attended
                    hidden = hidden + feed_forward(ff_norm(hidden))

                    self._features[layer + 1][query] = hidden
                    self._available[layer + 1] = query
                    self._next_query[layer] = query + 1
                    self._evict(layer)

            emitted = []
            while self._next_emit <= self._available[self.n_layers]:
                index = self._next_emit
                hidden = self._features[self.n_layers].pop(index)
                emitted.append((index, float(self.model.head(hidden[None, None, :])[0, 0])))
                self._next_emit += 1
        return emitted


def streaming_profile(config: ModelConfig = DEFAULT_MODEL_CONFIG, hop_ms: float = 10.0) -> dict:
    """Deployment profile: context, latency, and cache size for a config."""
    model = VADModel(config)
    layers = int(config.attn_layers)
    head_dim = config.d_model // config.attn_heads

    past = model.effective_past_window_frames
    lookahead = model.lookahead_frames
    streamable = config.temporal == "causal_attn"

    if streamable and config.past_window_frames is not None:
        entries = (int(config.past_window_frames) + int(config.lookahead_frames) + 1)
        cache_floats = entries * layers * int(config.attn_heads) * head_dim * 2
    else:
        cache_floats = None

    return {
        "temporal": config.temporal,
        "streamable": streamable,
        "per_layer_past_window_frames": config.past_window_frames,
        "effective_past_window_frames": past,
        "effective_past_window_ms": None if past is None else past * hop_ms,
        "per_layer_lookahead_frames": config.lookahead_frames,
        "effective_lookahead_frames": lookahead,
        "emission_latency_ms": None if lookahead < 0 else lookahead * hop_ms,
        "frontend_context_frames": model.frontend_context_frames,
        "kv_cache_floats": cache_floats,
        "kv_cache_kib": None if cache_floats is None else cache_floats * 4 / 1024,
        "methodology": _methodology_note(config),
    }


def _methodology_note(config: ModelConfig) -> list[str]:
    """The sentences that go into the deployment section of the report."""
    if config.temporal != "causal_attn":
        return [
            "The BiGRU core is bidirectional and carries unbounded recurrent state, "
            "so it is an offline model and past_window_frames does not apply to it.",
        ]

    note = [
        "Evaluation is done with the windowed batch forward, not with a "
        "frame-by-frame loop. This model is not autoregressive, meaning no output "
        "is fed back as input, so a batch forward carrying the same causal window "
        "computes numerically identical per-frame logits to true streaming. The "
        "streaming session is kept only to verify that equivalence and to measure "
        "per-frame latency and constant memory.",
    ]
    if config.past_window_frames is None:
        note.append(
            "The past window is unbounded, so this configuration is offline. Any "
            "deployment behind a finite window must be trained with that same "
            "window, since a model trained on unbounded history has learned to "
            "rely on context it would not have."
        )
    else:
        note.append(
            f"The training window equals the deployment window: "
            f"past_window_frames={config.past_window_frames} per layer is applied "
            f"in the same masked forward used for training, so the model learns to "
            f"decide from exactly the history it will have at inference. Training "
            f"unbounded and deploying windowed is a train and deploy mismatch and "
            f"costs accuracy."
        )
    return note


def print_streaming_profile(config: ModelConfig = DEFAULT_MODEL_CONFIG,
                            hop_ms: float = 10.0) -> dict:
    """Human-readable form of `streaming_profile`, for the deployment section."""
    profile = streaming_profile(config, hop_ms)
    print(f"streaming profile: {profile['temporal']}")
    if not profile["streamable"]:
        print("  not streamable: the BiGRU core is bidirectional and keeps unbounded state")
        print("  past_window_frames and lookahead_frames are attention-only and ignored here")
        print("  methodology")
        for paragraph in profile["methodology"]:
            for line in _wrap(paragraph, 74):
                print(f"    {line}")
        return profile

    past = profile["effective_past_window_frames"]
    print(f"  past window     per layer {profile['per_layer_past_window_frames']}, "
          f"effective {past if past is not None else 'unbounded'} frames"
          + (f" ({profile['effective_past_window_ms']:.0f} ms)" if past is not None else ""))
    print(f"  lookahead       per layer {profile['per_layer_lookahead_frames']}, "
          f"effective {profile['effective_lookahead_frames']} frames "
          f"({profile['emission_latency_ms']:.0f} ms emission latency)")
    print(f"  frontend needs  {profile['frontend_context_frames']} past frames")
    if profile["kv_cache_floats"] is None:
        print("  kv cache        unbounded, grows with the stream (set past_window_frames)")
    else:
        print(f"  kv cache        {profile['kv_cache_floats']:,} floats "
              f"({profile['kv_cache_kib']:.1f} KiB at fp32), constant in stream length")
    print("  methodology")
    for paragraph in profile["methodology"]:
        for line in _wrap(paragraph, 74):
            print(f"    {line}")
    return profile


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
