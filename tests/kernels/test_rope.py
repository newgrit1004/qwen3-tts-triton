"""Tests for Fused M-RoPE Triton kernel."""

import pytest
import torch

from qwen3_tts_triton.kernels.rope import triton_mrope_forward

pytestmark = pytest.mark.gpu

MROPE_SECTION = [24, 20, 20]
N_Q_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128


def torch_mrope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for interleaved M-RoPE.

    Args:
        q: (bsz, n_q_head, seq_len, head_dim)
        k: (bsz, n_kv_head, seq_len, head_dim)
        cos: (3, bsz, seq_len, head_dim)
        sin: (3, bsz, seq_len, head_dim)
        mrope_section: [sec_t, sec_h, sec_w], sum = head_dim/2

    Returns:
        Rotated (q, k) in original shapes.
    """
    sec_t, sec_h, _sec_w = mrope_section
    half_dim = q.shape[-1] // 2
    orig_dtype = q.dtype

    q = q.float()
    k = k.float()
    cos = cos.float()
    sin = sin.float()

    # Merge cos/sin from 3 dimensions based on section boundaries
    merged_cos = torch.cat(
        [
            cos[0, :, :, :sec_t],
            cos[1, :, :, sec_t : sec_t + sec_h],
            cos[2, :, :, sec_t + sec_h : half_dim],
        ],
        dim=-1,
    ).unsqueeze(1)  # (bsz, 1, seq_len, half_dim)

    merged_sin = torch.cat(
        [
            sin[0, :, :, :sec_t],
            sin[1, :, :, sec_t : sec_t + sec_h],
            sin[2, :, :, sec_t + sec_h : half_dim],
        ],
        dim=-1,
    ).unsqueeze(1)

    # Interleaved rotation: pairs (x_{2i}, x_{2i+1})
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    new_q_even = q_even * merged_cos - q_odd * merged_sin
    new_q_odd = q_odd * merged_cos + q_even * merged_sin
    new_q = torch.stack([new_q_even, new_q_odd], dim=-1).flatten(-2)

    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    new_k_even = k_even * merged_cos - k_odd * merged_sin
    new_k_odd = k_odd * merged_cos + k_even * merged_sin
    new_k = torch.stack([new_k_even, new_k_odd], dim=-1).flatten(-2)

    return new_q.to(orig_dtype), new_k.to(orig_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mrope_triton_vs_torch(
    batch_size: int, seq_len: int, dtype: torch.dtype
) -> None:
    """Triton M-RoPE must match PyTorch reference within tolerance."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(
        batch_size, N_Q_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, N_KV_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype
    )
    cos = torch.randn(
        3, batch_size, seq_len, HEAD_DIM, device=device, dtype=torch.float32
    )
    sin = torch.randn(
        3, batch_size, seq_len, HEAD_DIM, device=device, dtype=torch.float32
    )

    q_ref, k_ref = q.clone(), k.clone()

    q_triton, k_triton = triton_mrope_forward(q, k, cos, sin, MROPE_SECTION)
    q_torch, k_torch = torch_mrope_reference(q_ref, k_ref, cos, sin, MROPE_SECTION)

    q_diff = (q_triton - q_torch).abs().max().item()
    k_diff = (k_triton - k_torch).abs().max().item()

    # bfloat16 has 7-bit mantissa (vs 10-bit for float16) -> wider tolerance
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3

    assert not q_triton.isnan().any(), "NaN in q output"
    assert not k_triton.isnan().any(), "NaN in k output"
    assert torch.allclose(
        q_triton, q_torch, atol=atol, rtol=rtol
    ), f"Q mismatch: max diff = {q_diff}"
    assert torch.allclose(
        k_triton, k_torch, atol=atol, rtol=rtol
    ), f"K mismatch: max diff = {k_diff}"


# --- New tests: in-place, sections, edge cases, determinism ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrope_inplace_mutation() -> None:
    """triton_mrope_forward returns rotated tensors different from input."""
    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn(1, N_Q_HEADS, 128, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, N_KV_HEADS, 128, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos = torch.randn(3, 1, 128, HEAD_DIM, device=device, dtype=torch.float32)
    sin = torch.randn(3, 1, 128, HEAD_DIM, device=device, dtype=torch.float32)

    q_before = q.clone()
    k_before = k.clone()

    q_out, k_out = triton_mrope_forward(q, k, cos, sin, MROPE_SECTION)

    assert not torch.equal(q_out, q_before), "Q unchanged after rotation"
    assert not torch.equal(k_out, k_before), "K unchanged after rotation"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mrope_section",
    [[32, 16, 16], [20, 22, 22], [10, 27, 27]],
)
def test_mrope_alternative_sections(mrope_section: list[int]) -> None:
    """M-RoPE works with different section splits summing to head_dim/2."""
    torch.manual_seed(42)
    device = "cuda"
    seq_len = 64

    dtype = torch.bfloat16
    q = torch.randn(1, N_Q_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    cos = torch.randn(3, 1, seq_len, HEAD_DIM, device=device, dtype=torch.float32)
    sin = torch.randn(3, 1, seq_len, HEAD_DIM, device=device, dtype=torch.float32)

    q_ref, k_ref = q.clone(), k.clone()
    q_triton, k_triton = triton_mrope_forward(q, k, cos, sin, mrope_section)
    q_torch, k_torch = torch_mrope_reference(q_ref, k_ref, cos, sin, mrope_section)

    assert not q_triton.isnan().any(), f"NaN in q, sections={mrope_section}"
    assert torch.allclose(
        q_triton, q_torch, atol=1e-2, rtol=1e-2
    ), f"Q mismatch with sections={mrope_section}"
    assert torch.allclose(
        k_triton, k_torch, atol=1e-2, rtol=1e-2
    ), f"K mismatch with sections={mrope_section}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrope_single_head() -> None:
    """M-RoPE handles minimal head count (n_heads=1)."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(1, 1, 64, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 64, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos = torch.randn(3, 1, 64, HEAD_DIM, device=device, dtype=torch.float32)
    sin = torch.randn(3, 1, 64, HEAD_DIM, device=device, dtype=torch.float32)

    q_ref, k_ref = q.clone(), k.clone()
    q_triton, k_triton = triton_mrope_forward(q, k, cos, sin, MROPE_SECTION)
    q_torch, k_torch = torch_mrope_reference(q_ref, k_ref, cos, sin, MROPE_SECTION)

    assert torch.allclose(q_triton, q_torch, atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_triton, k_torch, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrope_seq_len_one() -> None:
    """M-RoPE handles seq_len=1 edge case."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(1, N_Q_HEADS, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos = torch.randn(3, 1, 1, HEAD_DIM, device=device, dtype=torch.float32)
    sin = torch.randn(3, 1, 1, HEAD_DIM, device=device, dtype=torch.float32)

    q_ref, k_ref = q.clone(), k.clone()
    q_triton, k_triton = triton_mrope_forward(q, k, cos, sin, MROPE_SECTION)
    q_torch, k_torch = torch_mrope_reference(q_ref, k_ref, cos, sin, MROPE_SECTION)

    assert q_triton.shape == (1, N_Q_HEADS, 1, HEAD_DIM)
    assert torch.allclose(q_triton, q_torch, atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_triton, k_torch, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrope_determinism() -> None:
    """Identical inputs produce bitwise identical outputs across 3 runs."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(1, N_Q_HEADS, 64, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, N_KV_HEADS, 64, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos = torch.randn(3, 1, 64, HEAD_DIM, device=device, dtype=torch.float32)
    sin = torch.randn(3, 1, 64, HEAD_DIM, device=device, dtype=torch.float32)

    results = [
        triton_mrope_forward(
            q.clone(), k.clone(), cos.clone(), sin.clone(), MROPE_SECTION
        )
        for _ in range(3)
    ]
    assert torch.equal(results[0][0], results[1][0]), "Q: Run 1 != Run 2"
    assert torch.equal(results[1][0], results[2][0]), "Q: Run 2 != Run 3"
    assert torch.equal(results[0][1], results[1][1]), "K: Run 1 != Run 2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [63, 127, 255])
def test_mrope_non_power_of_2_seq(seq_len: int) -> None:
    """M-RoPE handles non-power-of-2 sequence lengths."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(
        1, N_Q_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        1, N_KV_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cos = torch.randn(3, 1, seq_len, HEAD_DIM, device=device, dtype=torch.float32)
    sin = torch.randn(3, 1, seq_len, HEAD_DIM, device=device, dtype=torch.float32)

    q_ref, k_ref = q.clone(), k.clone()
    q_triton, k_triton = triton_mrope_forward(q, k, cos, sin, MROPE_SECTION)
    q_torch, k_torch = torch_mrope_reference(q_ref, k_ref, cos, sin, MROPE_SECTION)

    assert not q_triton.isnan().any(), f"NaN in q, seq_len={seq_len}"
    assert torch.allclose(
        q_triton, q_torch, atol=1e-2, rtol=1e-2
    ), f"Q mismatch, seq_len={seq_len}"
    assert torch.allclose(
        k_triton, k_torch, atol=1e-2, rtol=1e-2
    ), f"K mismatch, seq_len={seq_len}"
