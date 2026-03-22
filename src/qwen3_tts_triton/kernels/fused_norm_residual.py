"""Fused RMSNorm + Residual Add Triton kernel (forward-only, inference optimized)."""

import torch
import triton
import triton.language as tl

from qwen3_tts_triton.kernels.utils import calculate_settings

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _fused_add_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused residual-add + RMSNorm forward kernel (llama casting mode).

    Computes in a single pass:
        s_i = x_i + r_i          (residual add)
        y_i = (s_i / RMS(s)) * w_i   (RMSNorm)

    Llama casting mode: S is cast to fp32 for variance computation only,
    then cast back before weight multiplication.
    Single HBM roundtrip: load X, R once, store Y and S.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    R_row = tl.load(R_ptr + row_idx * R_row_stride + col_offsets, mask=mask, other=0.0)
    S_row_dtype = X_row.dtype

    # Residual add in fp32 to preserve precision
    X_fp32 = X_row.to(tl.float32)
    R_fp32 = R_row.to(tl.float32)
    S_fp32 = X_fp32 + R_fp32

    # Store updated residual (cast back to original dtype)
    S_row = S_fp32.to(S_row_dtype)
    tl.store(S_ptr + row_idx * S_row_stride + col_offsets, S_row, mask=mask)

    # Load weight
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    # Variance + normalize in fp32 (reuse S_fp32, no extra cast)
    mean_square = tl.sum(S_fp32 * S_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # Normalize in fp32, cast back before weight scaling
    S_norm = (S_fp32 * rstd).to(S_row_dtype)
    Y_row = S_norm * W_row

    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def triton_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused residual-add + RMSNorm using a Triton kernel.

    Computes:
        residual = x + residual
        output = RMSNorm(residual)

    Args:
        x: Input tensor of shape (B, T, H) or (N, H).
        residual: Residual tensor, same shape as x.
        weight: Weight tensor of shape (H,).
        eps: Epsilon for numerical stability. Default: 1e-6.

    Returns:
        Tuple of (normalized_output, updated_residual), both same shape as x.
    """
    shape = x.shape
    n_cols = shape[-1]

    x = x.contiguous().view(-1, n_cols)
    residual = residual.contiguous().view(-1, n_cols)
    weight = weight.contiguous()
    n_rows = x.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y = torch.empty_like(x)
    s = torch.empty_like(x)

    _fused_add_rms_norm_forward_kernel[(n_rows,)](
        y,
        y.stride(0),
        s,
        s.stride(0),
        x,
        x.stride(0),
        residual,
        residual.stride(0),
        weight,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y.view(*shape), s.view(*shape)


class TritonFusedAddRMSNorm(torch.nn.Module):
    """Fused residual-add + RMSNorm backed by a Triton forward kernel.

    Computes ``residual = x + residual`` then ``output = RMSNorm(residual)``
    in a single GPU kernel launch. Optimized for inference with fp32 variance
    accumulation (llama casting mode).

    Args:
        hidden_size: Size of the last dimension.
        eps: Epsilon added to the variance for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize TritonFusedAddRMSNorm.

        Args:
            hidden_size: Size of the last dimension.
            eps: Epsilon added to the variance for numerical stability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual-add + RMSNorm in a single kernel launch.

        Computes::

            residual = x + residual
            output   = RMSNorm(residual)

        Args:
            x: Input tensor of shape (B, T, H) or (N, H).
            residual: Residual tensor of the same shape as x.

        Returns:
            Tuple of (normalized_output, updated_residual), both with the
            same shape and dtype as x.
        """
        return triton_fused_add_rms_norm(x, residual, self.weight, self.eps)

    def extra_repr(self) -> str:
        """Return a string of extra representation parameters.

        Returns:
            A string describing hidden_size and eps values.
        """
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
