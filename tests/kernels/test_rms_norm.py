"""Tests for the fused RMSNorm Triton kernel."""

import pytest
import torch

from qwen3_tts_triton.kernels.rms_norm import TritonRMSNorm, triton_rms_norm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Half-precision kernels accumulate rounding errors on large tensors.
# bfloat16 has only 7-bit mantissa → max abs diff ~0.03 is normal.
_ATOL = 0.05
_RTOL = 0.05


def _pytorch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch RMSNorm (llama casting mode: fp32 variance only)."""
    x_fp32 = x.float()
    rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 / rms).to(x.dtype) * weight


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 2048),
        (2, 512, 2048),
        (4, 1024, 2048),
        (1, 1, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_function(shape: tuple, dtype: torch.dtype) -> None:
    """triton_rms_norm output matches PyTorch reference within tolerance."""
    hidden_size = shape[-1]
    torch.manual_seed(42)

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    ref = _pytorch_rms_norm(x, weight)
    out = triton_rms_norm(x, weight, eps=1e-6)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    assert out.dtype == ref.dtype, f"Dtype mismatch: {out.dtype} vs {ref.dtype}"
    assert not out.isnan().any(), f"NaN in output, shape={shape}, dtype={dtype}"
    assert not out.isinf().any(), f"Inf in output, shape={shape}, dtype={dtype}"
    assert torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL), (
        f"Max abs diff: {(out - ref).abs().max().item():.6f}, "
        f"shape={shape}, dtype={dtype}"
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 2048),
        (2, 512, 2048),
        (4, 1024, 2048),
        (1, 1, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_module(shape: tuple, dtype: torch.dtype) -> None:
    """TritonRMSNorm module output matches PyTorch reference within tolerance."""
    hidden_size = shape[-1]
    torch.manual_seed(42)

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    module = TritonRMSNorm(hidden_size=hidden_size).to(dtype=dtype, device="cuda")
    weight = module.weight.detach()

    ref = _pytorch_rms_norm(x, weight)
    out = module(x)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    assert out.dtype == ref.dtype, f"Dtype mismatch: {out.dtype} vs {ref.dtype}"
    assert not out.isnan().any(), f"NaN in module output, shape={shape}"
    assert not out.isinf().any(), f"Inf in module output, shape={shape}"
    assert torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL), (
        f"Max abs diff: {(out - ref).abs().max().item():.6f}, "
        f"shape={shape}, dtype={dtype}"
    )


# --- New tests: edge cases, determinism, numerical stability ---


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_2d_input(dtype: torch.dtype) -> None:
    """triton_rms_norm handles 2D (N, H) inputs correctly."""
    torch.manual_seed(42)
    x = torch.randn(128, 2048, dtype=dtype, device="cuda")
    weight = torch.randn(2048, dtype=dtype, device="cuda")

    ref = _pytorch_rms_norm(x, weight)
    out = triton_rms_norm(x, weight, eps=1e-6)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_non_contiguous(dtype: torch.dtype) -> None:
    """triton_rms_norm handles non-contiguous inputs via .contiguous()."""
    torch.manual_seed(42)
    # Create non-contiguous tensor via transpose
    x = torch.randn(2048, 128, dtype=dtype, device="cuda").T  # (128, 2048)
    assert not x.is_contiguous()
    weight = torch.randn(2048, dtype=dtype, device="cuda")

    ref = _pytorch_rms_norm(x.contiguous(), weight)
    out = triton_rms_norm(x, weight, eps=1e-6)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL)


def test_triton_rms_norm_float32() -> None:
    """triton_rms_norm works with float32 dtype."""
    torch.manual_seed(42)
    x = torch.randn(2, 128, 2048, dtype=torch.float32, device="cuda")
    weight = torch.randn(2048, dtype=torch.float32, device="cuda")

    ref = _pytorch_rms_norm(x, weight)
    out = triton_rms_norm(x, weight, eps=1e-6)

    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_triton_rms_norm_single_token() -> None:
    """triton_rms_norm handles minimal sequence length (1,1,H)."""
    torch.manual_seed(42)
    x = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(2048, dtype=torch.bfloat16, device="cuda")

    ref = _pytorch_rms_norm(x, weight)
    out = triton_rms_norm(x, weight, eps=1e-6)

    assert out.shape == (1, 1, 2048)
    assert torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL)


def test_triton_rms_norm_determinism() -> None:
    """Identical inputs produce bitwise identical outputs across 3 runs."""
    torch.manual_seed(42)
    x = torch.randn(2, 128, 2048, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(2048, dtype=torch.bfloat16, device="cuda")

    results = [triton_rms_norm(x.clone(), weight.clone()) for _ in range(3)]
    assert torch.equal(results[0], results[1]), "Run 1 != Run 2"
    assert torch.equal(results[1], results[2]), "Run 2 != Run 3"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_numerical_stability(dtype: torch.dtype) -> None:
    """Kernel produces no NaN/Inf for extreme input values."""
    torch.manual_seed(42)
    hidden = 2048
    weight = torch.ones(hidden, dtype=dtype, device="cuda")

    # Near-zero inputs
    x_small = torch.full((1, 32, hidden), 1e-7, dtype=dtype, device="cuda")
    out_small = triton_rms_norm(x_small, weight)
    assert not out_small.isnan().any(), "NaN with near-zero input"
    assert not out_small.isinf().any(), "Inf with near-zero input"

    # Large-magnitude inputs (50% of max)
    x_large = torch.full(
        (1, 32, hidden), torch.finfo(dtype).max * 0.5, dtype=dtype, device="cuda"
    )
    out_large = triton_rms_norm(x_large, weight)
    assert not out_large.isnan().any(), "NaN with large input"

    # Mixed-scale inputs
    x_mixed = torch.randn(1, 32, hidden, dtype=dtype, device="cuda")
    x_mixed[:, :, : hidden // 2] *= 1000
    x_mixed[:, :, hidden // 2 :] *= 0.001
    out_mixed = triton_rms_norm(x_mixed, weight)
    assert not out_mixed.isnan().any(), "NaN with mixed-scale input"
    assert not out_mixed.isinf().any(), "Inf with mixed-scale input"
