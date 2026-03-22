"""Tests for the fused RMSNorm + Residual Add Triton kernel."""

import pytest
import torch

from qwen3_tts_triton.kernels.fused_norm_residual import (
    TritonFusedAddRMSNorm,
    triton_fused_add_rms_norm,
)

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
HIDDEN_SIZE = 2048
EPS = 1e-6


def _pytorch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Reference RMSNorm in PyTorch (llama casting mode).

    Llama mode: cast to fp32 for variance, normalize, cast back,
    then multiply by weight in the original dtype.
    """
    x_fp32 = x.float()
    rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 / rms).to(x.dtype) * weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, HIDDEN_SIZE),
        (2, 512, HIDDEN_SIZE),
        (4, 1024, HIDDEN_SIZE),
    ],
)
def test_fused_add_rms_norm(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Triton fused-add-rmsnorm matches the two-step PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=dtype)

    # PyTorch reference (fp32 addition to match kernel precision)
    ref_residual_fp32 = x.float() + residual.float()
    ref_residual = ref_residual_fp32.to(dtype)
    rms = torch.sqrt(ref_residual_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ref_output = (ref_residual_fp32 / rms).to(dtype) * weight

    # Triton kernel
    tri_output, tri_residual = triton_fused_add_rms_norm(x, residual, weight, EPS)

    assert tri_residual.shape == ref_residual.shape
    assert tri_residual.dtype == ref_residual.dtype
    assert torch.allclose(
        tri_residual, ref_residual, atol=1e-3, rtol=1e-3
    ), f"Residual max abs diff: {(tri_residual - ref_residual).abs().max().item():.6f}"
    assert torch.allclose(tri_output, ref_output, atol=1e-2, rtol=1e-2), (
        f"Output max abs diff: {(tri_output - ref_output).abs().max().item():.6f}, "
        f"shape={shape}, dtype={dtype}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_module_wrapper() -> None:
    """TritonFusedAddRMSNorm module produces correct results."""
    torch.manual_seed(42)
    shape = (2, 512, HIDDEN_SIZE)
    dtype = torch.float16

    module = TritonFusedAddRMSNorm(HIDDEN_SIZE, EPS).to(device=DEVICE, dtype=dtype)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(shape, device=DEVICE, dtype=dtype)

    tri_output, tri_residual = module(x, residual)

    ref_residual_fp32 = x.float() + residual.float()
    ref_residual = ref_residual_fp32.to(dtype)
    rms = torch.sqrt(ref_residual_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ref_output = (ref_residual_fp32 / rms).to(dtype) * module.weight.data

    assert torch.allclose(tri_residual, ref_residual, atol=1e-3, rtol=1e-3)
    assert torch.allclose(
        tri_output, ref_output, atol=1e-2, rtol=1e-2
    ), f"Max abs diff: {(tri_output - ref_output).abs().max().item():.6f}"


# --- New tests: bfloat16 module, 2D, non-contiguous, determinism, stability ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_module_wrapper_bfloat16() -> None:
    """TritonFusedAddRMSNorm module works with bfloat16."""
    torch.manual_seed(42)
    shape = (2, 512, HIDDEN_SIZE)
    dtype = torch.bfloat16

    module = TritonFusedAddRMSNorm(HIDDEN_SIZE, EPS).to(device=DEVICE, dtype=dtype)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(shape, device=DEVICE, dtype=dtype)

    tri_output, tri_residual = module(x, residual)

    ref_residual_fp32 = x.float() + residual.float()
    ref_residual = ref_residual_fp32.to(dtype)
    rms = torch.sqrt(ref_residual_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ref_output = (ref_residual_fp32 / rms).to(dtype) * module.weight.data

    assert not tri_output.isnan().any(), "NaN in bfloat16 module output"
    assert torch.allclose(tri_residual, ref_residual, atol=1e-3, rtol=1e-3)
    assert torch.allclose(
        tri_output, ref_output, atol=1e-2, rtol=1e-2
    ), f"Max abs diff: {(tri_output - ref_output).abs().max().item():.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_2d_input(dtype: torch.dtype) -> None:
    """Fused add+RMSNorm handles 2D (N, H) inputs."""
    torch.manual_seed(42)
    x = torch.randn(128, HIDDEN_SIZE, device=DEVICE, dtype=dtype)
    residual = torch.randn(128, HIDDEN_SIZE, device=DEVICE, dtype=dtype)
    weight = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=dtype)

    tri_output, tri_residual = triton_fused_add_rms_norm(x, residual, weight, EPS)

    ref_residual_fp32 = x.float() + residual.float()
    ref_residual = ref_residual_fp32.to(dtype)
    rms = torch.sqrt(ref_residual_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ref_output = (ref_residual_fp32 / rms).to(dtype) * weight

    assert tri_output.shape == (128, HIDDEN_SIZE)
    assert torch.allclose(tri_residual, ref_residual, atol=1e-3, rtol=1e-3)
    assert torch.allclose(tri_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_add_rms_norm_non_contiguous() -> None:
    """Fused add+RMSNorm handles non-contiguous inputs."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    # Create non-contiguous via transpose
    x = torch.randn(HIDDEN_SIZE, 128, device=DEVICE, dtype=dtype).T
    residual = torch.randn(HIDDEN_SIZE, 128, device=DEVICE, dtype=dtype).T
    assert not x.is_contiguous()
    weight = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=dtype)

    tri_output, tri_residual = triton_fused_add_rms_norm(x, residual, weight, EPS)

    x_c = x.contiguous()
    r_c = residual.contiguous()
    ref_fp32 = x_c.float() + r_c.float()
    ref_residual = ref_fp32.to(dtype)
    rms = torch.sqrt(ref_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ref_output = (ref_fp32 / rms).to(dtype) * weight

    assert torch.allclose(tri_residual, ref_residual, atol=1e-3, rtol=1e-3)
    assert torch.allclose(tri_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_add_rms_norm_determinism() -> None:
    """Identical inputs produce bitwise identical outputs across 3 runs."""
    torch.manual_seed(42)
    shape = (2, 128, HIDDEN_SIZE)
    dtype = torch.bfloat16
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=dtype)

    results = [
        triton_fused_add_rms_norm(x.clone(), residual.clone(), weight.clone(), EPS)
        for _ in range(3)
    ]
    assert torch.equal(results[0][0], results[1][0]), "Output: Run 1 != Run 2"
    assert torch.equal(results[1][0], results[2][0]), "Output: Run 2 != Run 3"
    assert torch.equal(results[0][1], results[1][1]), "Residual: Run 1 != Run 2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_numerical_stability(dtype: torch.dtype) -> None:
    """Kernel produces no NaN/Inf for extreme input values."""
    shape = (1, 32, HIDDEN_SIZE)
    weight = torch.ones(HIDDEN_SIZE, device=DEVICE, dtype=dtype)

    # Near-zero inputs
    x = torch.full(shape, 1e-7, device=DEVICE, dtype=dtype)
    r = torch.full(shape, 1e-7, device=DEVICE, dtype=dtype)
    out, res = triton_fused_add_rms_norm(x, r, weight, EPS)
    assert not out.isnan().any(), "NaN with near-zero input"
    assert not out.isinf().any(), "Inf with near-zero input"

    # Mixed-scale inputs
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    r = torch.randn(shape, device=DEVICE, dtype=dtype)
    x[:, :, : HIDDEN_SIZE // 2] *= 1000
    r[:, :, HIDDEN_SIZE // 2 :] *= 0.001
    out, res = triton_fused_add_rms_norm(x, r, weight, EPS)
    assert not out.isnan().any(), "NaN with mixed-scale input"
    assert not out.isinf().any(), "Inf with mixed-scale input"
