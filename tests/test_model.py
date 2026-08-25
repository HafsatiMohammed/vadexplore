"""Tests for the VAD model, run against both temporal cores.

The padding-invariance test is the important one: it is what proves the
packing and the attention masks actually keep padded frames out of every real
frame's output, and it is the failure that is invisible until evaluation
numbers quietly depend on batch composition.
"""

from __future__ import annotations

import pytest
import torch

from vadexplore.config import ModelConfig
from vadexplore.model import (
    TEMPORAL_CORES,
    VADModel,
    count_parameters,
    masked_bce_loss,
    parameter_report,
)

N_MELS = 40
PARAM_MIN, PARAM_MAX = 100_000, 400_000

# dropout off by default so a test compares the model, not the RNG
CORES = [
    pytest.param({"temporal": "bigru"}, id="bigru"),
    pytest.param({"temporal": "causal_attn"}, id="causal_attn"),
]
CAUSAL_CORES = [
    pytest.param({"temporal": "bigru", "causal": True}, id="bigru-causal"),
    pytest.param({"temporal": "causal_attn", "lookahead_frames": 0}, id="causal_attn"),
]


def make_model(overrides=None, **extra):
    settings = {"dropout": 0.0, "attn_dropout": 0.0}
    settings.update(overrides or {})
    settings.update(extra)
    model = VADModel(ModelConfig(**settings))
    model.eval()
    return model


def pad_batch(sequences):
    """Build a collate-shaped batch from a list of (n_frames, n_mels) tensors."""
    max_len = max(len(s) for s in sequences)
    features = torch.zeros(len(sequences), max_len, N_MELS)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, sequence in enumerate(sequences):
        features[i, : len(sequence)] = sequence
        mask[i, : len(sequence)] = True
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    return {"features": features, "mask": mask, "lengths": lengths,
            "stems": [f"clip-{i}" for i in range(len(sequences))]}


@pytest.fixture
def sequences():
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(n, N_MELS, generator=generator) for n in (63, 31, 17)]


# --- 1. forward shape -----------------------------------------------------


@pytest.mark.parametrize("overrides", CORES)
def test_forward_returns_logits_per_frame(overrides, sequences):
    model = make_model(overrides)
    batch = pad_batch(sequences)
    logits = model.forward_batch(batch)

    assert logits.shape == batch["features"].shape[:2] == (3, 63)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("overrides", CORES)
def test_model_emits_raw_logits_not_probabilities(overrides):
    """No squashing inside the model, so the caller owns the sigmoid.

    Checked structurally rather than by inspecting the output range. An
    untrained model's range depends on the initialization draw, and the head's
    LayerNorm makes the output scale-invariant anyway, so neither the range nor
    a scaling probe can distinguish a squashed head from a linear one.
    """
    model = make_model(overrides)
    assert not any(isinstance(m, (torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.Tanh))
                   for m in model.modules())
    assert isinstance(model.head.linear, torch.nn.Linear)
    assert model.head.linear.out_features == 1

    # the head ends on an unbounded affine map, so nothing clips the range
    last = list(model.head.children())[-1]
    assert isinstance(last, torch.nn.Linear)


@pytest.mark.parametrize("overrides", CORES)
def test_forward_without_mask_treats_everything_as_real(overrides):
    model = make_model(overrides)
    features = torch.randn(2, 25, N_MELS)
    assert model(features).shape == (2, 25)


@pytest.mark.parametrize("overrides", CORES)
@pytest.mark.parametrize("n_frames", [1, 2, 7, 128])
def test_frame_count_is_preserved_exactly(overrides, n_frames):
    model = make_model(overrides)
    batch = pad_batch([torch.randn(n_frames, N_MELS)])
    assert model.forward_batch(batch).shape == (1, n_frames)


def test_unknown_core_is_rejected():
    with pytest.raises(ValueError, match="temporal must be one of"):
        VADModel(ModelConfig(temporal="lstm"))


# --- 2. padding and masking, the critical test ----------------------------


@pytest.mark.parametrize("overrides", CORES)
def test_padded_frames_do_not_change_real_frame_logits(overrides, sequences):
    """A clip's logits must be the same alone as inside a padded batch."""
    model = make_model(overrides)
    short = sequences[2]

    alone = model.forward_batch(pad_batch([short]))[0, : len(short)]
    with_longer = model.forward_batch(pad_batch([short, sequences[0]]))[0, : len(short)]
    with_two = model.forward_batch(pad_batch([short, sequences[0], sequences[1]]))[0, : len(short)]

    assert torch.allclose(alone, with_longer, atol=1e-5)
    assert torch.allclose(alone, with_two, atol=1e-5)


@pytest.mark.parametrize("overrides", CORES)
def test_padding_content_does_not_leak(overrides, sequences):
    """Even garbage in the padded region must not move a real frame."""
    model = make_model(overrides)
    batch = pad_batch(sequences)
    clean = model.forward_batch(batch)

    poisoned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    poisoned["features"][~batch["mask"]] = 50.0
    dirty = model.forward_batch(poisoned)

    assert torch.allclose(clean[batch["mask"]], dirty[batch["mask"]], atol=1e-5)


@pytest.mark.parametrize("overrides", CORES)
def test_extra_padding_does_not_change_train_mode_statistics(overrides, sequences):
    """Masked normalization: padding must stay out of the batch statistics.

    Plain BatchNorm would fold the padded zeros into the batch mean and
    variance, and the leak would be invisible in eval mode because running
    statistics are frozen.
    """
    model = make_model(overrides)
    model.train()
    batch = pad_batch(sequences)

    extra = 40
    wider = torch.cat([batch["features"],
                       torch.zeros(len(sequences), extra, N_MELS)], dim=1)
    wider_mask = torch.cat([batch["mask"],
                            torch.zeros(len(sequences), extra, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        narrow = model(batch["features"], batch["mask"], batch["lengths"])
        widened = model(wider, wider_mask, batch["lengths"])

    assert torch.allclose(narrow[batch["mask"]],
                          widened[wider_mask], atol=1e-5)


@pytest.mark.parametrize("overrides", CORES)
def test_logits_are_zero_on_padded_positions(overrides, sequences):
    model = make_model(overrides)
    batch = pad_batch(sequences)
    logits = model.forward_batch(batch)
    assert (logits[~batch["mask"]] == 0).all()


# --- 3. causality ---------------------------------------------------------


@pytest.mark.parametrize("overrides", CAUSAL_CORES)
def test_causal_cores_ignore_the_future(overrides):
    """Rewriting the input after frame t must not move any output up to t."""
    model = make_model(overrides)
    assert model.is_causal

    generator = torch.Generator().manual_seed(1)
    features = torch.randn(1, 60, N_MELS, generator=generator)
    cut = 30

    before = model(features)
    altered = features.clone()
    altered[:, cut + 1:] = torch.randn(1, 60 - cut - 1, N_MELS, generator=generator) * 5
    after = model(altered)

    assert torch.allclose(before[:, : cut + 1], after[:, : cut + 1], atol=1e-5)
    # and the future itself really did change, so the test is not vacuous
    assert not torch.allclose(before[:, cut + 1:], after[:, cut + 1:], atol=1e-3)


def test_bidirectional_gru_does_look_at_the_future():
    """The default core must be genuinely bidirectional."""
    model = make_model({"temporal": "bigru", "causal": False})
    assert not model.is_causal

    generator = torch.Generator().manual_seed(2)
    features = torch.randn(1, 60, N_MELS, generator=generator)
    cut = 30

    before = model(features)
    altered = features.clone()
    altered[:, cut + 1:] = torch.randn(1, 60 - cut - 1, N_MELS, generator=generator) * 5
    after = model(altered)

    assert not torch.allclose(before[:, : cut + 1], after[:, : cut + 1], atol=1e-4)


@pytest.mark.parametrize("lookahead, layers", [(5, 2), (3, 2), (4, 1)])
def test_lookahead_widens_the_causal_window_by_exactly_n_frames(lookahead, layers):
    """End-to-end lookahead is per-layer lookahead times attention depth.

    Attention layers compose: layer 2 reads layer 1 outputs up to t + N, and
    those already read inputs up to t + 2N. Reporting the per-layer value as
    the latency would understate it by a factor of the depth.
    """
    model = make_model({"temporal": "causal_attn", "lookahead_frames": lookahead,
                        "attn_layers": layers})
    assert not model.is_causal
    effective = lookahead * layers
    assert model.lookahead_frames == effective

    generator = torch.Generator().manual_seed(3)
    features = torch.randn(1, 60, N_MELS, generator=generator)
    cut = 30

    before = model(features)
    altered = features.clone()
    altered[:, cut + 1:] = torch.randn(1, 60 - cut - 1, N_MELS, generator=generator) * 5
    after = model(altered)

    # The frontend is causal, so the model's total lookahead is exactly the
    # attention lookahead. Frame t reads up to t + lookahead, so changing
    # everything after `cut` must move frames above cut - lookahead and no
    # frame at or below it.
    safe = cut - effective
    assert torch.allclose(before[:, : safe + 1], after[:, : safe + 1], atol=1e-5)
    assert not torch.allclose(before[:, safe + 1: cut + 1],
                              after[:, safe + 1: cut + 1], atol=1e-4)


def test_non_causal_frontend_reintroduces_lookahead():
    """The frontend padding mode is what makes the causal claim hold."""
    model = make_model({"temporal": "causal_attn", "causal_frontend": False})
    assert not model.is_causal
    assert model.lookahead_frames == len(model.config.conv_channels)

    generator = torch.Generator().manual_seed(4)
    features = torch.randn(1, 60, N_MELS, generator=generator)
    cut = 30
    before = model(features)
    altered = features.clone()
    altered[:, cut + 1:] = torch.randn(1, 60 - cut - 1, N_MELS, generator=generator) * 5
    after = model(altered)

    # frame `cut` now reads future frames through the symmetric convolutions
    assert not torch.allclose(before[:, cut], after[:, cut], atol=1e-4)


def test_causal_attention_mask_shape_and_content():
    from vadexplore.model import causal_attention_mask
    mask = causal_attention_mask(4, 0, torch.device("cpu"))
    assert mask.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]
    widened = causal_attention_mask(4, 1, torch.device("cpu"))
    assert widened[0].tolist() == [False, False, True, True]


# --- 4. parameter counts --------------------------------------------------


@pytest.mark.parametrize("temporal", TEMPORAL_CORES)
def test_parameter_counts_are_in_the_comparable_range(temporal):
    report = parameter_report(ModelConfig(temporal=temporal))
    print(f"\n{temporal}: frontend {report['frontend']:,}  core {report['core']:,}  "
          f"head {report['head']:,}  total {report['total']:,}")
    assert PARAM_MIN <= report["total"] <= PARAM_MAX
    assert PARAM_MIN <= report["core"] <= PARAM_MAX


def test_frontend_and_head_are_identical_across_cores():
    """The only difference between runs must be the temporal block."""
    reports = {t: parameter_report(ModelConfig(temporal=t)) for t in TEMPORAL_CORES}
    assert len({r["frontend"] for r in reports.values()}) == 1
    assert len({r["head"] for r in reports.values()}) == 1

    shapes = {}
    for temporal in TEMPORAL_CORES:
        model = VADModel(ModelConfig(temporal=temporal))
        shapes[temporal] = (
            [tuple(p.shape) for p in model.frontend.parameters()],
            [tuple(p.shape) for p in model.head.parameters()],
        )
    assert shapes["bigru"] == shapes["causal_attn"]


def test_cores_are_within_a_factor_of_two_of_each_other():
    counts = [parameter_report(ModelConfig(temporal=t))["core"] for t in TEMPORAL_CORES]
    assert max(counts) / min(counts) <= 2.0


# --- 5. gradients ---------------------------------------------------------


@pytest.mark.parametrize("overrides", CORES)
def test_gradients_are_finite_after_masked_bce(overrides, sequences):
    model = VADModel(ModelConfig(**{"dropout": 0.0, "attn_dropout": 0.0, **overrides}))
    model.train()
    batch = pad_batch(sequences)
    labels = torch.zeros_like(batch["mask"], dtype=torch.long)
    labels[batch["mask"]] = torch.randint(0, 2, (int(batch["mask"].sum()),))

    logits = model.forward_batch(batch)
    loss = masked_bce_loss(logits, labels, batch["mask"])
    assert torch.isfinite(loss)
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("overrides", CORES)
def test_masked_loss_ignores_padded_positions(overrides, sequences):
    model = make_model(overrides)
    batch = pad_batch(sequences)
    labels = torch.full_like(batch["mask"], -100, dtype=torch.long)
    labels[batch["mask"]] = torch.randint(0, 2, (int(batch["mask"].sum()),))

    logits = model.forward_batch(batch)
    baseline = masked_bce_loss(logits, labels, batch["mask"])

    scrambled = logits.clone()
    scrambled[~batch["mask"]] = 99.0
    assert torch.allclose(baseline, masked_bce_loss(scrambled, labels, batch["mask"]))


# --- 6. devices -----------------------------------------------------------


def available_devices():
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("overrides", CORES)
@pytest.mark.parametrize("device", available_devices(), ids=lambda d: d.type)
def test_runs_on_available_devices(overrides, sequences, device):
    """Covers the pack_padded_sequence lengths-must-be-on-CPU requirement."""
    model = make_model(overrides).to(device)
    batch = pad_batch(sequences)
    logits = model(
        batch["features"].to(device),
        batch["mask"].to(device),
        batch["lengths"].to(device),
    )
    assert logits.shape == (3, 63)
    assert logits.device.type == device.type
    assert torch.isfinite(logits).all()


# --- integration with the data layer --------------------------------------


def test_accepts_a_real_collate_batch():
    """The model must consume `data.collate` output without any reshaping."""
    from vadexplore.data import collate

    items = [
        {"features": torch.randn(n, N_MELS),
         "labels": torch.randint(0, 2, (n,)),
         "stem": f"clip-{n}"}
        for n in (40, 25, 11)
    ]
    batch = collate(items)
    model = make_model({"temporal": "bigru"})
    logits = model.forward_batch(batch)

    assert logits.shape == batch["labels"].shape
    assert torch.isfinite(masked_bce_loss(logits, batch["labels"], batch["mask"]))


# --- 7. bounded past window -----------------------------------------------


def test_unbounded_window_reproduces_the_original_mask():
    """past_window_frames=None must be exactly the previous behavior."""
    from vadexplore.model import causal_attention_mask
    cpu = torch.device("cpu")
    for n_frames in (1, 5, 40):
        for lookahead in (0, 3):
            assert torch.equal(
                causal_attention_mask(n_frames, lookahead, cpu),
                causal_attention_mask(n_frames, lookahead, cpu, past_window=None),
            )


def test_unbounded_window_reproduces_the_original_logits(sequences):
    default = make_model({"temporal": "causal_attn"})
    explicit = make_model({"temporal": "causal_attn", "past_window_frames": None})
    explicit.load_state_dict(default.state_dict())

    batch = pad_batch(sequences)
    assert torch.allclose(default.forward_batch(batch), explicit.forward_batch(batch), atol=1e-7)


@pytest.mark.parametrize("window, lookahead", [(2, 0), (3, 1), (5, 2)])
def test_windowed_mask_admits_exactly_the_intended_keys(window, lookahead):
    from vadexplore.model import causal_attention_mask
    mask = causal_attention_mask(12, lookahead, torch.device("cpu"), past_window=window)
    for query in range(12):
        allowed = torch.nonzero(~mask[query]).flatten().tolist()
        expected = [k for k in range(12) if query - window <= k <= query + lookahead]
        assert allowed == expected


def test_window_bounds_the_receptive_field_in_the_batch_path():
    """A frame must not move when input outside its composed window changes."""
    window, layers = 3, 2
    model = make_model({"temporal": "causal_attn", "past_window_frames": window,
                        "attn_layers": layers})
    effective = model.effective_past_window_frames
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(1, 60, N_MELS, generator=generator)

    probe = 50
    altered = features.clone()
    # everything strictly older than the composed window, and older than the
    # frontend's own context, must be irrelevant to frame `probe`
    edge = probe - effective - model.frontend_context_frames
    altered[:, :edge] = torch.randn(1, edge, N_MELS, generator=generator) * 5

    before = model(features)
    after = model(altered)
    assert torch.allclose(before[0, probe], after[0, probe], atol=1e-5)
    assert not torch.allclose(before[0, :edge], after[0, :edge], atol=1e-3)


@pytest.mark.parametrize("window, layers", [(4, 2), (10, 2), (7, 3), (6, 1)])
def test_past_window_composes_over_depth(window, layers):
    """Mirrors the lookahead composition: end-to-end context is window x depth."""
    model = make_model({"temporal": "causal_attn", "past_window_frames": window,
                        "attn_layers": layers})
    assert model.effective_past_window_frames == window * layers


def test_unbounded_and_bigru_report_no_finite_window():
    assert make_model({"temporal": "causal_attn"}).effective_past_window_frames is None
    assert make_model({"temporal": "bigru"}).effective_past_window_frames is None


def test_window_seconds_helper():
    from vadexplore.model import streaming_profile
    profile = streaming_profile(ModelConfig(temporal="causal_attn",
                                            past_window_frames=50, attn_layers=2))
    assert profile["effective_past_window_frames"] == 100
    assert profile["effective_past_window_ms"] == pytest.approx(1000.0)


# --- 8. attention-only settings are a documented no-op for the BiGRU ------


def test_bigru_ignores_attention_only_settings(sequences):
    """Documented no-op: the settings change nothing for the recurrent core."""
    plain = make_model({"temporal": "bigru"})
    decorated = make_model({"temporal": "bigru", "past_window_frames": 4,
                            "lookahead_frames": 3})
    decorated.load_state_dict(plain.state_dict())

    batch = pad_batch(sequences)
    assert torch.allclose(plain.forward_batch(batch), decorated.forward_batch(batch), atol=1e-7)
    assert decorated.effective_past_window_frames is None


def test_streaming_a_bigru_fails_loudly():
    """The no-op is silent in the forward pass, so streaming must not be."""
    from vadexplore.model import StreamingVADSession
    model = make_model({"temporal": "bigru", "past_window_frames": 4})
    with pytest.raises(ValueError, match="requires the causal_attn core"):
        StreamingVADSession(model)


def test_streaming_requires_eval_mode():
    from vadexplore.model import StreamingVADSession
    model = make_model({"temporal": "causal_attn", "past_window_frames": 4})
    model.train()
    with pytest.raises(ValueError, match="model.eval"):
        StreamingVADSession(model)


# --- 9. streaming equals batching, the load-bearing test ------------------


@pytest.mark.parametrize("window", [None, 4, 8, 64])
@pytest.mark.parametrize("lookahead", [0, 2])
def test_streaming_matches_the_batch_forward(window, lookahead):
    """If frame-by-frame streaming and the masked batch pass disagree, one is wrong."""
    from vadexplore.model import StreamingVADSession

    model = make_model({"temporal": "causal_attn", "past_window_frames": window,
                        "lookahead_frames": lookahead})
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(37, N_MELS, generator=generator)

    with torch.no_grad():
        batched = model(features[None])[0]
    streamed = StreamingVADSession(model).run(features)

    assert streamed.shape == batched.shape
    assert torch.allclose(streamed, batched, atol=1e-5), \
        f"max difference {(streamed - batched).abs().max():.2e}"


def test_streaming_matches_batch_when_eviction_actually_happens():
    """A window far shorter than the clip, so the cache really does evict."""
    from vadexplore.model import StreamingVADSession

    window = 5
    model = make_model({"temporal": "causal_attn", "past_window_frames": window,
                        "lookahead_frames": 0})
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(120, N_MELS, generator=generator)

    session = StreamingVADSession(model)
    streamed = session.run(features)
    with torch.no_grad():
        batched = model(features[None])[0]

    assert torch.allclose(streamed, batched, atol=1e-5)
    # the cache must hold a window, not the whole 120-frame history
    assert session.cached_entries <= (window + 1) * model.config.attn_layers


def test_cold_start_frames_match_the_batch_forward():
    """The first `window` frames have less than a full window of history.

    The batch path is in the same position there, since its mask cannot reach
    below index zero, so the two must agree frame for frame from the very
    first one.
    """
    from vadexplore.model import StreamingVADSession

    window = 8
    model = make_model({"temporal": "causal_attn", "past_window_frames": window,
                        "lookahead_frames": 1})
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(30, N_MELS, generator=generator)

    streamed = StreamingVADSession(model).run(features)
    with torch.no_grad():
        batched = model(features[None])[0]

    cold = window * model.config.attn_layers
    assert torch.allclose(streamed[:cold], batched[:cold], atol=1e-5)
    assert torch.allclose(streamed[0], batched[0], atol=1e-5)
    assert torch.allclose(streamed, batched, atol=1e-5)


def test_streaming_tail_is_released_by_finish():
    """Frames still inside the lookahead delay must come out at end of stream."""
    from vadexplore.model import StreamingVADSession

    model = make_model({"temporal": "causal_attn", "past_window_frames": 6,
                        "lookahead_frames": 2, "attn_layers": 2})
    session = StreamingVADSession(model)
    assert session.emission_delay_frames == 4

    generator = torch.Generator().manual_seed(10)
    features = torch.randn(20, N_MELS, generator=generator)

    emitted = []
    for i in range(20):
        emitted.extend(session.push(features[i]))
    assert len(emitted) == 20 - session.emission_delay_frames

    emitted.extend(session.finish())
    assert [index for index, _ in emitted] == list(range(20))


def test_streaming_memory_is_constant_in_stream_length():
    """The whole point of the window: memory must not grow with the stream."""
    from vadexplore.model import StreamingVADSession

    window = 12
    model = make_model({"temporal": "causal_attn", "past_window_frames": window})
    session = StreamingVADSession(model)

    generator = torch.Generator().manual_seed(12)
    sizes = []
    for step in range(400):
        session.push(torch.randn(N_MELS, generator=generator))
        if step >= 100:
            sizes.append(session.cached_entries)

    assert len(set(sizes)) == 1, f"cache size drifted: {sorted(set(sizes))}"
    assert sizes[0] <= (window + 1) * model.config.attn_layers


def test_unbounded_streaming_memory_grows_and_says_so():
    """Documenting the failure mode the window exists to prevent."""
    from vadexplore.model import StreamingVADSession, streaming_profile

    model = make_model({"temporal": "causal_attn", "past_window_frames": None})
    session = StreamingVADSession(model)
    generator = torch.Generator().manual_seed(13)

    for _ in range(20):
        session.push(torch.randn(N_MELS, generator=generator))
    small = session.cached_entries
    for _ in range(80):
        session.push(torch.randn(N_MELS, generator=generator))
    assert session.cached_entries > small

    assert streaming_profile(model.config)["kv_cache_floats"] is None


# --- 10. streaming profile ------------------------------------------------


def test_streaming_profile_cache_formula():
    from vadexplore.model import streaming_profile

    config = ModelConfig(temporal="causal_attn", past_window_frames=50,
                         lookahead_frames=2, attn_layers=2, attn_heads=4, d_model=128)
    profile = streaming_profile(config)

    head_dim = config.d_model // config.attn_heads
    entries = config.past_window_frames + config.lookahead_frames + 1
    expected = entries * config.attn_layers * config.attn_heads * head_dim * 2

    assert profile["kv_cache_floats"] == expected
    assert profile["streamable"] is True
    assert profile["effective_lookahead_frames"] == 4
    assert profile["emission_latency_ms"] == pytest.approx(40.0)


def test_streaming_profile_marks_bigru_unstreamable(capsys):
    from vadexplore.model import print_streaming_profile
    profile = print_streaming_profile(ModelConfig(temporal="bigru"))
    assert profile["streamable"] is False
    assert "not streamable" in capsys.readouterr().out


# --- 11. the window constrains TRAINING, not only inference ---------------


def capture_attention_weights(model, features, mask=None, lengths=None):
    """Run a forward pass and return each attention layer's weight matrix.

    `nn.MultiheadAttention` does not expose its weights unless asked, so each
    module's forward is temporarily wrapped to request and record them. This
    reads the real weights used by the pass, rather than re-deriving the mask.
    """
    captured = []
    originals = [layer.forward for layer in model.core.attention]

    def wrap(original):
        def forward(query, key, value, **kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            out, weights = original(query, key, value, **kwargs)
            captured.append(weights.detach())
            return out, None
        return forward

    for layer, original in zip(model.core.attention, originals):
        layer.forward = wrap(original)
    try:
        model(features, mask, lengths)
    finally:
        for layer, original in zip(model.core.attention, originals):
            layer.forward = original
    return captured


@pytest.mark.parametrize("window, lookahead", [(4, 0), (6, 2), (10, 0)])
def test_training_forward_gives_zero_weight_outside_the_window(window, lookahead):
    """The window must bind during training, not only at inference.

    A model that trains with every frame attending to the whole clip has
    learned to rely on history it will not have behind a deployment window.
    This asserts the training forward pass really does starve each frame of
    anything older than t - window.
    """
    model = make_model({"temporal": "causal_attn", "past_window_frames": window,
                        "lookahead_frames": lookahead})
    model.train()  # the point of the test: training mode, not eval

    generator = torch.Generator().manual_seed(21)
    features = torch.randn(1, 45, N_MELS, generator=generator)
    layers = capture_attention_weights(model, features)

    assert len(layers) == model.config.attn_layers
    for weights in layers:
        weights = weights[0]  # (queries, keys)
        n_frames = weights.shape[0]
        for query in range(n_frames):
            too_old = weights[query, : max(0, query - window)]
            too_new = weights[query, query + lookahead + 1:]
            assert torch.all(too_old == 0), f"query {query} attended below its window"
            assert torch.all(too_new == 0), f"query {query} attended past its lookahead"
            # and the surviving row is still a distribution
            assert weights[query].sum() == pytest.approx(1.0, abs=1e-5)


def test_unbounded_training_forward_does_attend_far_back():
    """Control for the test above: without a window the far past is used."""
    model = make_model({"temporal": "causal_attn", "past_window_frames": None})
    model.train()

    generator = torch.Generator().manual_seed(22)
    features = torch.randn(1, 45, N_MELS, generator=generator)
    layers = capture_attention_weights(model, features)

    last_row = layers[0][0, -1]
    assert last_row[0] > 0, "unbounded attention should still reach frame 0"


def test_window_constrains_the_same_positions_in_train_and_eval_mode():
    """One forward path, so the window blocks the same keys in both modes.

    The attention values themselves legitimately differ: the frontend's
    BatchNorm uses batch statistics while training and running statistics
    while evaluating. What must not differ is which keys are reachable, which
    is the window.
    """
    window = 5
    model = make_model({"temporal": "causal_attn", "past_window_frames": window})
    generator = torch.Generator().manual_seed(23)
    features = torch.randn(1, 40, N_MELS, generator=generator)

    model.eval()
    evaluated = capture_attention_weights(model, features)
    model.train()
    trained = capture_attention_weights(model, features)

    for a, b in zip(evaluated, trained):
        assert torch.equal(a == 0, b == 0), "the reachable key set changed between modes"
        assert not torch.allclose(a, b, atol=1e-6), (
            "values identical, so this test is not actually exercising train mode")


# --- 12. the window changes what the model computes -----------------------


@pytest.mark.parametrize("window", [2, 5, 10])
def test_windowed_and_unbounded_forwards_disagree(window):
    """The window is not cosmetic: it changes the computed logits.

    Same weights, same input, only the mask differs. Frames deep enough into
    the clip to have more than `window` frames of past come out different.

    Implication: the training window and the deployment window must be the
    same value. A model trained unbounded and then deployed behind this window
    is being asked to decide from a history it never learned on, and these
    logits are the evidence that the two are not the same computation.
    """
    unbounded = make_model({"temporal": "causal_attn", "past_window_frames": None})
    windowed = make_model({"temporal": "causal_attn", "past_window_frames": window})
    windowed.load_state_dict(unbounded.state_dict())  # identical weights

    generator = torch.Generator().manual_seed(24)
    features = torch.randn(1, 60, N_MELS, generator=generator)

    with torch.no_grad():
        wide = unbounded(features)[0]
        narrow = windowed(features)[0]

    composed = windowed.effective_past_window_frames
    deep = composed + windowed.frontend_context_frames + 1
    difference = (wide[deep:] - narrow[deep:]).abs()

    assert difference.max() > 1e-3, (
        f"window {window} changed nothing, so the mask is not being applied: "
        f"max difference {difference.max():.2e}"
    )
    # the very first frames have no past to lose, so they must agree
    assert torch.allclose(wide[0], narrow[0], atol=1e-6)


def test_two_different_windows_also_disagree():
    """Not just windowed against unbounded: window size itself matters."""
    small = make_model({"temporal": "causal_attn", "past_window_frames": 3})
    large = make_model({"temporal": "causal_attn", "past_window_frames": 20})
    large.load_state_dict(small.state_dict())

    generator = torch.Generator().manual_seed(25)
    features = torch.randn(1, 60, N_MELS, generator=generator)
    with torch.no_grad():
        difference = (small(features)[0, 50:] - large(features)[0, 50:]).abs().max()
    assert difference > 1e-3


# --- 13. methodology reporting --------------------------------------------


def test_profile_states_the_evaluation_methodology(capsys):
    from vadexplore.model import print_streaming_profile

    profile = print_streaming_profile(
        ModelConfig(temporal="causal_attn", past_window_frames=50, lookahead_frames=2))
    text = " ".join(profile["methodology"]).lower()

    assert "not autoregressive" in text
    assert "windowed batch forward" in text
    assert "numerically identical" in text
    assert "training window equals the deployment window" in text

    printed = capsys.readouterr().out
    assert "methodology" in printed
    assert "not autoregressive" in printed


def test_profile_warns_when_unbounded_is_deployed_windowed():
    from vadexplore.model import streaming_profile
    text = " ".join(streaming_profile(
        ModelConfig(temporal="causal_attn", past_window_frames=None))["methodology"]).lower()
    assert "unbounded" in text and "trained with that same window" in text


def test_streaming_docstrings_disclaim_being_required():
    from vadexplore.model import StreamingVADSession
    for doc in (StreamingVADSession.__doc__,
                StreamingVADSession.run.__doc__,
                StreamingVADSession.push.__doc__):
        assert "not required" in doc.lower() or "not needed" in doc.lower()
