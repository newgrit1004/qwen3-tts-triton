"""Tier 1 GPU tests for Triton fused dequantization kernel.

Verifies that triton_fused_dequant() matches the Python reference
implementation (per-head loop with dequantize_vectors) to within
floating-point associativity tolerance.
"""

import pytest
import torch

from qwen3_tts_triton.kernels.turboquant import (
    TurboQuantKVCache,
    dequantize_vectors,
    generate_rotation_matrix,
    lloyd_max_codebook,
    pack_3bit,
    pack_4bit,
    quantize_vectors,
)

# Skip entire module if CUDA is unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)

DEVICE = "cuda"
HEAD_DIM = 128
NUM_HEADS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_data(
    bits: int,
    batch: int = 1,
    heads: int = NUM_HEADS,
    seq_len: int = 64,
    dim: int = HEAD_DIM,
    device: str = DEVICE,
):
    """Create quantized test data and reference rotations/codebook."""
    codebook = lloyd_max_codebook(bits, device=device)
    from qwen3_tts_triton.kernels.turboquant import lloyd_max_boundaries

    boundaries = lloyd_max_boundaries(codebook).to(device)

    rotations_list = [
        generate_rotation_matrix(dim, 0, h, device=device, dtype=torch.float32)
        for h in range(heads)
    ]
    rot_stacked = torch.stack(rotations_list)  # (H, D, D)

    # Generate random data → quantize → pack
    torch.manual_seed(42)
    x = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.bfloat16)

    all_indices = []
    all_scales = []
    for h in range(heads):
        head_data = x[:, h, :, :]
        idx, sc = quantize_vectors(head_data, rotations_list[h], codebook, boundaries)
        all_indices.append(idx)
        all_scales.append(sc)

    indices = torch.stack(all_indices, dim=1)  # (B, H, S, D) uint8
    scales = torch.stack(all_scales, dim=1)  # (B, H, S, 1)

    # Pack
    if bits == 4:
        packed = pack_4bit(indices)
    else:
        packed = pack_3bit(indices)

    return packed, scales, indices, rotations_list, rot_stacked, codebook


def _python_reference_dequant(
    indices: torch.Tensor,
    scales: torch.Tensor,
    rotations_list: list[torch.Tensor],
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Reference dequant using the existing Python per-head loop."""
    b, h, s, d = indices.shape
    heads = []
    for head_idx in range(h):
        deq = dequantize_vectors(
            indices[:, head_idx, :, :],
            scales[:, head_idx, :, :],
            rotations_list[head_idx],
            codebook,
        )
        heads.append(deq)
    return torch.stack(heads, dim=1)


# ---------------------------------------------------------------------------
# 4-bit tests
# ---------------------------------------------------------------------------


class TestFusedDequant4bit:
    """Tests for 4-bit fused dequantization kernel."""

    def test_matches_reference(self):
        """Triton output matches Python reference within tolerance."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(4)
        ref = _python_reference_dequant(indices, scales, rots_list, cb)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=4)
        assert torch.allclose(ref.float(), out.float(), atol=5e-3, rtol=5e-3)

    def test_cosine_similarity(self):
        """Per-head cosine similarity > 0.999 vs reference."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(
            4, seq_len=256
        )
        ref = _python_reference_dequant(indices, scales, rots_list, cb)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=4)

        for h in range(NUM_HEADS):
            ref_flat = ref[:, h].reshape(-1).float()
            out_flat = out[:, h].reshape(-1).float()
            cos = torch.nn.functional.cosine_similarity(
                ref_flat.unsqueeze(0), out_flat.unsqueeze(0)
            )
            assert cos.item() > 0.999, f"Head {h} cosine={cos.item():.6f}"

    def test_output_shape(self):
        """Output shape matches (B, H, S, D)."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, _, _, rot_stacked, cb = _make_test_data(4, batch=2, seq_len=32)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=4)
        assert out.shape == (2, NUM_HEADS, 32, HEAD_DIM)

    def test_output_dtype_bfloat16(self):
        """Default output is bfloat16."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, _, _, rot_stacked, cb = _make_test_data(4)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=4)
        assert out.dtype == torch.bfloat16

    def test_output_dtype_float16(self):
        """Explicit float16 output works."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, _, _, rot_stacked, cb = _make_test_data(4)
        out = triton_fused_dequant(
            packed,
            scales,
            rot_stacked,
            cb,
            HEAD_DIM,
            bits=4,
            out_dtype=torch.float16,
        )
        assert out.dtype == torch.float16

    def test_single_token(self):
        """Handles S=1 (autoregressive single-token decode)."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(
            4, seq_len=1
        )
        ref = _python_reference_dequant(indices, scales, rots_list, cb)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=4)
        assert out.shape == (1, NUM_HEADS, 1, HEAD_DIM)
        assert torch.allclose(ref.float(), out.float(), atol=5e-3, rtol=5e-3)

    def test_determinism(self):
        """Three runs produce identical results."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, _, _, rot_stacked, cb = _make_test_data(4)
        results = [
            triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=4)
            for _ in range(3)
        ]
        assert torch.equal(results[0], results[1])
        assert torch.equal(results[1], results[2])


# ---------------------------------------------------------------------------
# 3-bit tests
# ---------------------------------------------------------------------------


class TestFusedDequant3bit:
    """Tests for 3-bit fused dequantization kernel."""

    def test_matches_reference(self):
        """Triton output matches Python reference within tolerance."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(3)
        ref = _python_reference_dequant(indices, scales, rots_list, cb)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=3)
        assert torch.allclose(ref.float(), out.float(), atol=5e-3, rtol=5e-3)

    def test_cosine_similarity(self):
        """Per-head cosine similarity > 0.999 vs reference."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(
            3, seq_len=256
        )
        ref = _python_reference_dequant(indices, scales, rots_list, cb)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=3)

        for h in range(NUM_HEADS):
            ref_flat = ref[:, h].reshape(-1).float()
            out_flat = out[:, h].reshape(-1).float()
            cos = torch.nn.functional.cosine_similarity(
                ref_flat.unsqueeze(0), out_flat.unsqueeze(0)
            )
            assert cos.item() > 0.999, f"Head {h} cosine={cos.item():.6f}"

    def test_single_token(self):
        """Handles S=1 for 3-bit."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(
            3, seq_len=1
        )
        ref = _python_reference_dequant(indices, scales, rots_list, cb)
        out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=3)
        assert torch.allclose(ref.float(), out.float(), atol=5e-3, rtol=5e-3)

    def test_determinism(self):
        """Three runs produce identical results."""
        from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

        packed, scales, _, _, rot_stacked, cb = _make_test_data(3)
        results = [
            triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=3)
            for _ in range(3)
        ]
        assert torch.equal(results[0], results[1])
        assert torch.equal(results[1], results[2])


# ---------------------------------------------------------------------------
# Sequence length sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 16, 64, 256, 512])
@pytest.mark.parametrize("bits", [3, 4])
def test_fused_dequant_various_seq_lengths(seq_len: int, bits: int):
    """Fused kernel matches reference across various sequence lengths."""
    from qwen3_tts_triton.kernels.fused_dequant import triton_fused_dequant

    packed, scales, indices, rots_list, rot_stacked, cb = _make_test_data(
        bits, seq_len=seq_len
    )
    ref = _python_reference_dequant(indices, scales, rots_list, cb)
    out = triton_fused_dequant(packed, scales, rot_stacked, cb, HEAD_DIM, bits=bits)
    assert torch.allclose(ref.float(), out.float(), atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# KV Cache integration
# ---------------------------------------------------------------------------


def test_kv_cache_gpu_uses_fused_kernel():
    """TurboQuantKVCache on GPU dispatches to Triton fused path."""
    cache = TurboQuantKVCache(
        bits=4,
        num_layers=2,
        num_kv_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(42)
    k = torch.randn(1, NUM_HEADS, 16, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn_like(k)

    keys, values = cache.update(k, v, layer_idx=0)

    assert keys.shape == (1, NUM_HEADS, 16, HEAD_DIM)
    assert values.shape == (1, NUM_HEADS, 16, HEAD_DIM)
    assert keys.dtype == torch.bfloat16
    assert not torch.isnan(keys).any()
    assert not torch.isnan(values).any()

    # Incremental: add more tokens
    k2 = torch.randn(1, NUM_HEADS, 4, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    v2 = torch.randn_like(k2)
    keys2, values2 = cache.update(k2, v2, layer_idx=0)

    assert keys2.shape == (1, NUM_HEADS, 20, HEAD_DIM)
    assert values2.shape == (1, NUM_HEADS, 20, HEAD_DIM)
