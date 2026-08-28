"""Config dataclasses for the data layer and the model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DataConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    win_ms: float = 25.0
    hop_ms: float = 10.0
    fps: int = 100

    # None disables the high-pass, which is the whole ablation
    highpass_hz: float | None = 80.0
    highpass_order: int = 2

    # committed at the elbow of the gap distribution, see DECISIONS.md
    bridge_gap_s: float = 0.10

    # padded label positions, matching the torch loss convention
    ignore_index: int = -100

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Frontend and head are shared; temporal is the only thing that varies."""

    n_mels: int = 40
    conv_channels: tuple = (32, 64, 64)
    d_model: int = 128

    # Left-pad the frontend convolutions in time so the shared frontend adds no
    # lookahead of its own. Without this a "strictly causal" core still sees
    # len(conv_channels) frames of future through the convolution stack, and
    # lookahead_frames would not mean what it says.
    causal_frontend: bool = True

    temporal: str = "bigru"  # "bigru" or "causal_attn"

    # bigru core
    gru_hidden: int = 64
    gru_layers: int = 2
    causal: bool = False  # True makes the recurrence unidirectional

    # causal_attn core
    attn_layers: int = 2
    attn_heads: int = 4
    attn_ff_ratio: int = 2
    attn_dropout: float = 0.1
    lookahead_frames: int = 0  # 0 is strictly causal

    # Bounded past for the attention core. None keeps the unbounded causal
    # behavior, which is fine for batch evaluation but cannot stream on an
    # indefinitely long input at constant memory. Attention-only: the BiGRU
    # core ignores it, see VADModel.effective_past_window_frames.
    past_window_frames: int | None = None

    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
