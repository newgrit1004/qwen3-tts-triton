"""Tests for fused SwiGLU Triton kernel.

Compares Triton kernel output against PyTorch reference:
    expected = torch.nn.functional.silu(gate) * up
"""

import pytest
import torch
import torch.nn.functional as F

from qwen3_tts_triton.kernels.swiglu import TritonSwiGLU, triton_swiglu_forward

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

SHAPES = [
    (1, 128, 6144),
    (2, 512, 6144),
    (4, 1024, 3072),
]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_swiglu_forward(shape, dtype):
    """Verify triton_swiglu_forward matches silu(gate)*up within tolerance."""
    gate = torch.randn(*shape, dtype=dtype, device="cuda")
    up = torch.randn(*shape, dtype=dtype, device="cuda")

    expected = F.silu(gate) * up
    actual = triton_swiglu_forward(gate, up)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(
        actual, expected, atol=1e-3, rtol=1e-3
    ), f"shape={shape} dtype={dtype} max_diff={max_diff:.6f}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_swiglu_module(shape, dtype):
    """Verify TritonSwiGLU module output matches silu(gate)*up within tolerance."""
    gate = torch.randn(*shape, dtype=dtype, device="cuda")
    up = torch.randn(*shape, dtype=dtype, device="cuda")

    module = TritonSwiGLU()
    expected = F.silu(gate) * up
    actual = module(gate, up)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert not actual.isnan().any(), f"NaN in module output, shape={shape}"
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(
        actual, expected, atol=1e-3, rtol=1e-3
    ), f"shape={shape} dtype={dtype} max_diff={max_diff:.6f}"


# --- New tests: validation, edge cases, determinism, stability ---


def test_swiglu_shape_mismatch_raises() -> None:
    """triton_swiglu_forward raises ValueError on shape mismatch."""
    gate = torch.randn(2, 128, 6144, dtype=torch.float16, device="cuda")
    up = torch.randn(2, 128, 3072, dtype=torch.float16, device="cuda")

    with pytest.raises(ValueError, match="same shape"):
        triton_swiglu_forward(gate, up)


def test_swiglu_cpu_tensor_raises() -> None:
    """triton_swiglu_forward raises ValueError for CPU tensors."""
    gate = torch.randn(2, 128, 6144, dtype=torch.float16)
    up = torch.randn(2, 128, 6144, dtype=torch.float16)

    with pytest.raises(ValueError, match="CUDA"):
        triton_swiglu_forward(gate, up)


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_2d_input(dtype) -> None:
    """triton_swiglu_forward handles 2D (N, intermediate) inputs."""
    gate = torch.randn(128, 6144, dtype=dtype, device="cuda")
    up = torch.randn(128, 6144, dtype=dtype, device="cuda")

    expected = F.silu(gate) * up
    actual = triton_swiglu_forward(gate, up)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_non_contiguous(dtype) -> None:
    """triton_swiglu_forward handles non-contiguous inputs."""
    gate = torch.randn(6144, 128, dtype=dtype, device="cuda").T  # (128, 6144)
    up = torch.randn(6144, 128, dtype=dtype, device="cuda").T
    assert not gate.is_contiguous()

    expected = F.silu(gate.contiguous()) * up.contiguous()
    actual = triton_swiglu_forward(gate, up)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_swiglu_determinism() -> None:
    """Identical inputs produce bitwise identical outputs across 3 runs."""
    torch.manual_seed(42)
    gate = torch.randn(2, 128, 6144, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(2, 128, 6144, dtype=torch.bfloat16, device="cuda")

    results = [triton_swiglu_forward(gate.clone(), up.clone()) for _ in range(3)]
    assert torch.equal(results[0], results[1]), "Run 1 != Run 2"
    assert torch.equal(results[1], results[2]), "Run 2 != Run 3"


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_numerical_stability(dtype) -> None:
    """Kernel produces no NaN/Inf for extreme input values."""
    size = (1, 32, 6144)

    # Near-zero inputs
    gate = torch.full(size, 1e-7, dtype=dtype, device="cuda")
    up = torch.full(size, 1e-7, dtype=dtype, device="cuda")
    out = triton_swiglu_forward(gate, up)
    assert not out.isnan().any(), "NaN with near-zero input"
    assert not out.isinf().any(), "Inf with near-zero input"

    # Large positive inputs (silu saturation region)
    gate = torch.full(size, 10.0, dtype=dtype, device="cuda")
    up = torch.ones(size, dtype=dtype, device="cuda")
    out = triton_swiglu_forward(gate, up)
    assert not out.isnan().any(), "NaN with large gate"
    assert not out.isinf().any(), "Inf with large gate"

    # Large negative inputs (silu near zero)
    gate = torch.full(size, -10.0, dtype=dtype, device="cuda")
    up = torch.ones(size, dtype=dtype, device="cuda")
    out = triton_swiglu_forward(gate, up)
    assert not out.isnan().any(), "NaN with large negative gate"
    assert out.abs().max().item() < 1.0, "Large negative gate should give ~0"
