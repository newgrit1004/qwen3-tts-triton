"""Tier 1: TurboQuant kernel accuracy tests.

Validates Lloyd-Max codebook, rotation matrices, quantize/dequantize
round-trip fidelity, and TurboQuantKVCache correctness.
"""

import pytest
import torch
import torch.nn.functional as F

from qwen3_tts_triton.kernels.turboquant import (
    TurboQuantKVCache,
    dequantize_vectors,
    generate_rotation_matrix,
    lloyd_max_boundaries,
    lloyd_max_codebook,
    pack_3bit,
    pack_4bit,
    quantize_vectors,
    unpack_3bit,
    unpack_4bit,
)

# ---------------------------------------------------------------------------
# Lloyd-Max Codebook Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_max_codebook_length(bits: int) -> None:
    """Codebook has exactly 2**bits centroids."""
    cb = lloyd_max_codebook(bits)
    assert cb.shape == (2**bits,)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_max_codebook_sorted(bits: int) -> None:
    """Centroids are strictly sorted ascending."""
    cb = lloyd_max_codebook(bits)
    diffs = cb[1:] - cb[:-1]
    assert (diffs > 0).all(), "Codebook not strictly ascending"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_max_codebook_symmetric(bits: int) -> None:
    """Codebook is symmetric around zero for N(0,1)."""
    cb = lloyd_max_codebook(bits)
    n = cb.shape[0]
    for i in range(n // 2):
        assert abs(cb[i].item() + cb[n - 1 - i].item()) < 1e-5, (
            f"Asymmetric: {cb[i].item()} vs {cb[n - 1 - i].item()}"
        )


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_max_codebook_deterministic(bits: int) -> None:
    """Same bits always produces same codebook."""
    cb1 = lloyd_max_codebook(bits, device="cpu")
    cb2 = lloyd_max_codebook(bits, device="cpu")
    assert torch.equal(cb1, cb2)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_max_boundaries_count(bits: int) -> None:
    """Boundaries has 2**bits - 1 elements."""
    cb = lloyd_max_codebook(bits)
    bd = lloyd_max_boundaries(cb)
    assert bd.shape == (2**bits - 1,)


# ---------------------------------------------------------------------------
# Rotation Matrix Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [64, 128])
def test_rotation_matrix_orthogonality(dim: int) -> None:
    """R^T @ R should equal identity."""
    r = generate_rotation_matrix(dim, layer_idx=0, head_idx=0)
    eye = r.T @ r
    identity = torch.eye(dim)
    assert torch.allclose(eye, identity, atol=1e-5), (
        f"Max deviation from identity: {(eye - identity).abs().max().item()}"
    )


def test_rotation_matrix_determinism() -> None:
    """Same (dim, layer, head, seed) produces identical matrix."""
    r1 = generate_rotation_matrix(128, 5, 3, seed=42)
    r2 = generate_rotation_matrix(128, 5, 3, seed=42)
    assert torch.equal(r1, r2)


def test_rotation_matrix_uniqueness() -> None:
    """Different (layer, head) produces different matrices."""
    r1 = generate_rotation_matrix(128, 0, 0)
    r2 = generate_rotation_matrix(128, 0, 1)
    assert not torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# Quantize / Dequantize Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize_dequantize_roundtrip(bits: int, dtype: torch.dtype) -> None:
    """Quantize then dequantize should approximate original."""
    torch.manual_seed(123)
    dim = 128
    x = torch.randn(4, dim, dtype=dtype)  # 4 vectors

    rot = generate_rotation_matrix(dim, 0, 0)
    cb = lloyd_max_codebook(bits)
    bd = lloyd_max_boundaries(cb)

    indices, scales = quantize_vectors(x, rot, cb, bd)
    x_hat = dequantize_vectors(indices, scales, rot, cb)

    # Check shapes
    assert indices.shape == (4, dim)
    assert indices.dtype == torch.uint8
    assert scales.shape == (4, 1)
    assert x_hat.shape == x.shape

    # Cosine similarity per vector
    x_f = x.float()
    x_hat_f = x_hat.float()
    for i in range(4):
        cos = F.cosine_similarity(x_f[i].unsqueeze(0), x_hat_f[i].unsqueeze(0)).item()
        min_cos = 0.98 if bits == 4 else 0.95
        assert cos > min_cos, f"Vector {i}: cos_sim={cos:.4f} < {min_cos}"


@pytest.mark.parametrize("bits", [3, 4])
def test_quantize_indices_range(bits: int) -> None:
    """Indices should be in [0, 2**bits)."""
    torch.manual_seed(42)
    x = torch.randn(8, 128)
    rot = generate_rotation_matrix(128, 0, 0)
    cb = lloyd_max_codebook(bits)
    bd = lloyd_max_boundaries(cb)

    indices, _ = quantize_vectors(x, rot, cb, bd)
    assert indices.min() >= 0
    assert indices.max() < 2**bits


@pytest.mark.parametrize("bits", [3, 4])
def test_quantize_preserves_inner_product(bits: int) -> None:
    """PolarQuant should preserve Q@K^T inner products."""
    torch.manual_seed(99)
    dim = 128
    q = torch.randn(1, dim)
    k = torch.randn(16, dim)  # 16 key vectors

    rot = generate_rotation_matrix(dim, 0, 0)
    cb = lloyd_max_codebook(bits)
    bd = lloyd_max_boundaries(cb)

    # Original attention logits
    logits_orig = (q @ k.T).squeeze(0)

    # Quantized key attention logits
    k_idx, k_sc = quantize_vectors(k, rot, cb, bd)
    k_hat = dequantize_vectors(k_idx, k_sc, rot, cb)
    logits_quant = (q @ k_hat.T).squeeze(0)

    cos = F.cosine_similarity(
        logits_orig.unsqueeze(0), logits_quant.unsqueeze(0)
    ).item()
    min_cos = 0.99 if bits == 4 else 0.97
    assert cos > min_cos, f"Inner product cosine sim: {cos:.4f} < {min_cos}"


def test_quantize_preserves_norm_direction() -> None:
    """Dequantized vector should have similar norm and direction."""
    torch.manual_seed(7)
    x = torch.randn(1, 128) * 5.0  # larger scale
    rot = generate_rotation_matrix(128, 0, 0)
    cb = lloyd_max_codebook(4)
    bd = lloyd_max_boundaries(cb)

    _, scales = quantize_vectors(x, rot, cb, bd)
    # Scale should capture the original norm
    orig_norm = x.norm().item()
    scale_val = scales.item()
    assert abs(orig_norm - scale_val) / orig_norm < 0.01, (
        f"Norm mismatch: original={orig_norm:.4f}, scale={scale_val:.4f}"
    )


# ---------------------------------------------------------------------------
# TurboQuantKVCache Tests
# ---------------------------------------------------------------------------


def test_kv_cache_basic() -> None:
    """TurboQuantKVCache stores and retrieves correctly."""
    cache = TurboQuantKVCache(
        bits=4, num_layers=2, num_kv_heads=2, head_dim=64, device="cpu"
    )

    k = torch.randn(1, 2, 4, 64)  # B=1, H=2, S=4, D=64
    v = torch.randn(1, 2, 4, 64)

    keys, values = cache.update(k, v, layer_idx=0)

    assert keys.shape == (1, 2, 4, 64)
    assert values.shape == (1, 2, 4, 64)
    assert cache.get_seq_length(0) == 4
    assert cache.get_seq_length(1) == 0  # layer 1 untouched


def test_kv_cache_incremental() -> None:
    """Sequential updates accumulate correctly (autoregressive)."""
    cache = TurboQuantKVCache(
        bits=4, num_layers=2, num_kv_heads=2, head_dim=64, device="cpu"
    )

    # First token batch
    k1 = torch.randn(1, 2, 3, 64)
    v1 = torch.randn(1, 2, 3, 64)
    cache.update(k1, v1, layer_idx=0)
    assert cache.get_seq_length(0) == 3

    # Second token
    k2 = torch.randn(1, 2, 1, 64)
    v2 = torch.randn(1, 2, 1, 64)
    keys, values = cache.update(k2, v2, layer_idx=0)

    assert keys.shape == (1, 2, 4, 64)  # accumulated
    assert values.shape == (1, 2, 4, 64)
    assert cache.get_seq_length(0) == 4


def test_kv_cache_cosine_fidelity() -> None:
    """Cached-then-retrieved KV should be close to original."""
    torch.manual_seed(42)
    cache = TurboQuantKVCache(
        bits=4, num_layers=1, num_kv_heads=4, head_dim=128, device="cpu"
    )

    k = torch.randn(1, 4, 8, 128)
    v = torch.randn(1, 4, 8, 128)

    keys_out, values_out = cache.update(k, v, layer_idx=0)

    k_cos = F.cosine_similarity(
        k.flatten().unsqueeze(0), keys_out.flatten().unsqueeze(0)
    ).item()
    v_cos = F.cosine_similarity(
        v.flatten().unsqueeze(0), values_out.flatten().unsqueeze(0)
    ).item()

    assert k_cos > 0.97, f"Key fidelity: {k_cos:.4f}"
    assert v_cos > 0.97, f"Value fidelity: {v_cos:.4f}"


def test_kv_cache_memory_savings() -> None:
    """Compressed cache should use less memory than uncompressed."""
    cache = TurboQuantKVCache(
        bits=4, num_layers=2, num_kv_heads=4, head_dim=128, device="cpu"
    )

    k = torch.randn(1, 4, 32, 128)
    v = torch.randn(1, 4, 32, 128)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    stats = cache.get_memory_stats()
    ratio = float(stats["compression_ratio"])
    assert ratio > 1.5, f"Compression ratio too low: {ratio}"
    assert float(stats["compressed_mb"]) < float(stats["uncompressed_mb"])


def test_kv_cache_gqa() -> None:
    """Works with GQA (kv_heads < q_heads)."""
    cache = TurboQuantKVCache(
        bits=4,
        num_layers=1,
        num_kv_heads=8,  # Qwen3-TTS: 8 KV heads, 16 Q heads
        head_dim=128,
        device="cpu",
    )

    k = torch.randn(1, 8, 10, 128)
    v = torch.randn(1, 8, 10, 128)
    keys, values = cache.update(k, v, layer_idx=0)

    assert keys.shape == (1, 8, 10, 128)
    assert values.shape == (1, 8, 10, 128)


def test_kv_cache_reset() -> None:
    """Reset clears all state."""
    cache = TurboQuantKVCache(
        bits=4, num_layers=2, num_kv_heads=2, head_dim=64, device="cpu"
    )

    k = torch.randn(1, 2, 4, 64)
    v = torch.randn(1, 2, 4, 64)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(0) == 4

    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert not cache.is_initialized


@pytest.mark.parametrize("bits", [3, 4])
def test_kv_cache_bits_config(bits: int) -> None:
    """Cache respects bit-width configuration."""
    cache = TurboQuantKVCache(
        bits=bits, num_layers=1, num_kv_heads=2, head_dim=64, device="cpu"
    )
    assert cache.bits == bits

    k = torch.randn(1, 2, 4, 64)
    v = torch.randn(1, 2, 4, 64)
    keys, values = cache.update(k, v, layer_idx=0)

    # 3-bit should have lower fidelity than 4-bit
    assert keys.shape == (1, 2, 4, 64)


@pytest.mark.parametrize("seq_len", [1, 16, 64, 256])
def test_kv_cache_various_seq_lengths(seq_len: int) -> None:
    """Quantization works across different sequence lengths."""
    torch.manual_seed(42)
    cache = TurboQuantKVCache(
        bits=4, num_layers=1, num_kv_heads=2, head_dim=128, device="cpu"
    )

    k = torch.randn(1, 2, seq_len, 128)
    v = torch.randn(1, 2, seq_len, 128)
    keys, values = cache.update(k, v, layer_idx=0)

    assert keys.shape == (1, 2, seq_len, 128)
    assert cache.get_seq_length(0) == seq_len


# ---------------------------------------------------------------------------
# Bit-Packing Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_pack_unpack_4bit_roundtrip(dim: int) -> None:
    """4-bit pack/unpack roundtrip preserves all values."""
    indices = torch.randint(0, 16, (8, dim), dtype=torch.uint8)
    packed = pack_4bit(indices)
    assert packed.shape == (8, dim // 2)
    unpacked = unpack_4bit(packed, dim)
    assert torch.equal(indices, unpacked)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_pack_unpack_3bit_roundtrip(dim: int) -> None:
    """3-bit pack/unpack roundtrip preserves all values."""
    indices = torch.randint(0, 8, (8, dim), dtype=torch.uint8)
    packed = pack_3bit(indices)
    assert packed.shape == (8, dim * 3 // 8)
    unpacked = unpack_3bit(packed, dim)
    assert torch.equal(indices, unpacked)


def test_pack_4bit_size_reduction() -> None:
    """4-bit packing halves the storage size."""
    indices = torch.randint(0, 16, (1, 8, 32, 128), dtype=torch.uint8)
    packed = pack_4bit(indices)
    assert packed.nelement() == indices.nelement() // 2


def test_pack_3bit_size_reduction() -> None:
    """3-bit packing reduces storage to 3/8 of original."""
    indices = torch.randint(0, 8, (1, 8, 32, 128), dtype=torch.uint8)
    packed = pack_3bit(indices)
    assert packed.nelement() == indices.nelement() * 3 // 8


def test_pack_4bit_batch_shapes() -> None:
    """4-bit packing handles arbitrary batch dimensions."""
    indices = torch.randint(0, 16, (2, 4, 10, 128), dtype=torch.uint8)
    packed = pack_4bit(indices)
    assert packed.shape == (2, 4, 10, 64)
    unpacked = unpack_4bit(packed, 128)
    assert torch.equal(indices, unpacked)


def test_pack_3bit_batch_shapes() -> None:
    """3-bit packing handles arbitrary batch dimensions."""
    indices = torch.randint(0, 8, (2, 4, 10, 128), dtype=torch.uint8)
    packed = pack_3bit(indices)
    assert packed.shape == (2, 4, 10, 48)
    unpacked = unpack_3bit(packed, 128)
    assert torch.equal(indices, unpacked)


@pytest.mark.parametrize("bits", [3, 4])
def test_kv_cache_packed_storage(bits: int) -> None:
    """KV cache stores bit-packed indices internally."""
    cache = TurboQuantKVCache(
        bits=bits, num_layers=1, num_kv_heads=2, head_dim=128, device="cpu"
    )
    k = torch.randn(1, 2, 8, 128)
    v = torch.randn(1, 2, 8, 128)
    cache.update(k, v, layer_idx=0)

    # Check internal packed storage shape
    stored = cache._key_indices[0]
    assert isinstance(stored, torch.Tensor)
    if bits == 4:
        assert stored.shape[-1] == 64  # 128 // 2
    elif bits == 3:
        assert stored.shape[-1] == 48  # 128 * 3 // 8


@pytest.mark.parametrize("bits", [3, 4])
def test_kv_cache_compression_ratio_with_packing(bits: int) -> None:
    """Bit-packed cache achieves expected compression ratios."""
    cache = TurboQuantKVCache(
        bits=bits, num_layers=2, num_kv_heads=4, head_dim=128, device="cpu"
    )
    k = torch.randn(1, 4, 64, 128)
    v = torch.randn(1, 4, 64, 128)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    stats = cache.get_memory_stats()
    # 4-bit packed: ~3.8x, 3-bit packed: ~5.1x
    ratio = float(stats["compression_ratio"])
    if bits == 4:
        assert ratio > 3.5, f"4-bit ratio {ratio} < 3.5"
    else:
        assert ratio > 4.5, f"3-bit ratio {ratio} < 4.5"


@pytest.mark.parametrize("bits", [3, 4])
def test_kv_cache_packed_fidelity(bits: int) -> None:
    """Bit-packed cache maintains quantization fidelity."""
    torch.manual_seed(42)
    cache = TurboQuantKVCache(
        bits=bits, num_layers=1, num_kv_heads=4, head_dim=128, device="cpu"
    )
    k = torch.randn(1, 4, 16, 128)
    v = torch.randn(1, 4, 16, 128)

    keys_out, values_out = cache.update(k, v, layer_idx=0)

    k_cos = F.cosine_similarity(
        k.flatten().unsqueeze(0), keys_out.flatten().unsqueeze(0)
    ).item()
    v_cos = F.cosine_similarity(
        v.flatten().unsqueeze(0), values_out.flatten().unsqueeze(0)
    ).item()

    min_cos = 0.97 if bits == 4 else 0.94
    assert k_cos > min_cos, f"Key fidelity: {k_cos:.4f} < {min_cos}"
    assert v_cos > min_cos, f"Value fidelity: {v_cos:.4f} < {min_cos}"
