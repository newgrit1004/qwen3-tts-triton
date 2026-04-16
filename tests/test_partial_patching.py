"""Tests for partial layer patching (patch_range parameter)."""

import pytest
import torch
import torch.nn as nn

from qwen3_tts_triton.kernels.rms_norm import TritonRMSNorm
from qwen3_tts_triton.models.patching import (
    _get_layer_index,
    _should_patch,
    apply_triton_kernels,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

HIDDEN = 64  # small for fast tests
INTERMEDIATE = 128
EPS = 1e-6
NUM_LAYERS = 4


# ── Helpers: minimal model that mirrors Qwen3-TTS structure ──────────


class _FakeRMSNorm(nn.Module):
    """Minimal RMSNorm with a .weight attribute (detected by patching)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _FakeAttention(nn.Module):
    """Placeholder attention (not actually called, just needs to exist)."""

    def forward(self, **kwargs):  # type: ignore[override]
        return kwargs.get("hidden_states"), None


class _FakeMLP(nn.Module):
    """MLP with gate/up/down projections (detected by SwiGLU patching)."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _FakeDecoderLayer(nn.Module):
    """Decoder layer with 4 attributes that trigger fused norm patching."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.input_layernorm = _FakeRMSNorm(hidden)
        self.post_attention_layernorm = _FakeRMSNorm(hidden)
        self.self_attn = _FakeAttention()
        self.mlp = _FakeMLP(hidden, inter)


class _FakeModel(nn.Module):
    """Top-level model with .layers and a final .norm."""

    def __init__(self, n_layers: int, hidden: int, inter: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeDecoderLayer(hidden, inter) for _ in range(n_layers)]
        )
        self.norm = _FakeRMSNorm(hidden)  # final norm, outside layers


def _make_model() -> _FakeModel:
    model = _FakeModel(NUM_LAYERS, HIDDEN, INTERMEDIATE)
    return model.cuda()


# ── _get_layer_index tests ───────────────────────────────────────────


class TestGetLayerIndex:
    def test_layer_norm(self) -> None:
        assert _get_layer_index("layers.5.input_layernorm") == 5

    def test_nested_mlp(self) -> None:
        assert _get_layer_index("layers.27.mlp.gate_proj") == 27

    def test_final_norm_returns_none(self) -> None:
        assert _get_layer_index("norm") is None

    def test_model_prefix(self) -> None:
        assert _get_layer_index("model.layers.0.self_attn") == 0

    def test_no_layers_keyword(self) -> None:
        assert _get_layer_index("embeddings.weight") is None


# ── _should_patch tests ─────────────────────────────────────────────


class TestShouldPatch:
    def test_none_range_always_true(self) -> None:
        assert _should_patch("layers.5.mlp", None) is True
        assert _should_patch("norm", None) is True

    def test_in_range(self) -> None:
        assert _should_patch("layers.3.mlp", (0, 5)) is True

    def test_out_of_range(self) -> None:
        assert _should_patch("layers.5.mlp", (0, 5)) is False

    def test_final_norm_excluded_with_range(self) -> None:
        assert _should_patch("norm", (0, 5)) is False

    def test_boundary_start_inclusive(self) -> None:
        assert _should_patch("layers.0.mlp", (0, 2)) is True

    def test_boundary_end_exclusive(self) -> None:
        assert _should_patch("layers.2.mlp", (0, 2)) is False


# ── apply_triton_kernels with patch_range ────────────────────────────


def _count_triton_norms(model: nn.Module) -> int:
    """Count TritonRMSNorm instances in the model."""
    return sum(1 for m in model.modules() if isinstance(m, TritonRMSNorm))


def _count_original_norms(model: nn.Module) -> int:
    """Count _FakeRMSNorm instances (un-patched)."""
    return sum(1 for m in model.modules() if isinstance(m, _FakeRMSNorm))


def test_patch_range_none_patches_all() -> None:
    """patch_range=None patches every layer + final norm (default)."""
    model = _make_model()
    apply_triton_kernels(model, enable_fused_norm=False)

    # 4 layers * 2 norms + 1 final = 9 total norms, all should be Triton
    assert _count_triton_norms(model) == NUM_LAYERS * 2 + 1
    assert _count_original_norms(model) == 0


def test_partial_patch_only_patches_specified_layers() -> None:
    """patch_range=(0, 2) patches layers 0-1 only."""
    model = _make_model()
    apply_triton_kernels(model, enable_fused_norm=False, patch_range=(0, 2))

    # Layers 0,1: 2 norms each = 4 patched
    assert _count_triton_norms(model) == 4
    # Layers 2,3: 2 norms each = 4 un-patched + 1 final norm = 5
    assert _count_original_norms(model) == 5


def test_patch_range_final_norm_not_patched() -> None:
    """Final norm (outside layers) is not patched when patch_range is set."""
    model = _make_model()
    apply_triton_kernels(model, enable_fused_norm=False, patch_range=(0, NUM_LAYERS))

    # All layer norms patched, but final norm should still be original
    assert isinstance(model.norm, _FakeRMSNorm)


def test_patch_range_fused_norm_respects_range() -> None:
    """Fused norm+residual forward is only applied to layers in range."""
    model = _make_model()

    # Save original forward refs for layers 2,3
    orig_fwd_2 = type(model.layers[2]).forward
    orig_fwd_3 = type(model.layers[3]).forward

    apply_triton_kernels(model, enable_fused_norm=True, patch_range=(0, 2))

    # Layers 0,1 should have patched forwards (different from class default)
    assert model.layers[0].forward.__func__ is not orig_fwd_2  # type: ignore[attr-defined]
    assert model.layers[1].forward.__func__ is not orig_fwd_2  # type: ignore[attr-defined]
    # Layers 2,3 should retain original forward
    assert type(model.layers[2]).forward is orig_fwd_2
    assert type(model.layers[3]).forward is orig_fwd_3


def test_patch_range_validation_start_ge_end() -> None:
    """patch_range with start >= end raises ValueError."""
    model = _make_model()
    with pytest.raises(ValueError, match="0 <= start < end"):
        apply_triton_kernels(model, patch_range=(5, 5))


def test_patch_range_validation_negative() -> None:
    """patch_range with negative start raises ValueError."""
    model = _make_model()
    with pytest.raises(ValueError, match="0 <= start < end"):
        apply_triton_kernels(model, patch_range=(-1, 5))


def test_patch_range_validation_reversed() -> None:
    """patch_range with start > end raises ValueError."""
    model = _make_model()
    with pytest.raises(ValueError, match="0 <= start < end"):
        apply_triton_kernels(model, patch_range=(10, 3))
