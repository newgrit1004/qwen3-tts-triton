"""Fused RMSNorm Triton kernel (forward-only, inference optimized)."""

import torch
import triton
import triton.language as tl

from qwen3_tts_triton.kernels.utils import calculate_settings

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward kernel with llama casting mode.

    y_i = (x_i / RMS(x)) * w_i,  RMS = sqrt(sum(x_i^2) / N)

    Llama casting mode: X is cast to fp32 for variance computation,
    then cast back to the original dtype before weight multiplication.
    Single HBM roundtrip: load X once, compute in SRAM, store Y.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load X row and weight from HBM into SRAM
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    X_row_dtype = X_row.dtype

    # Llama mode: cast to fp32 for variance computation only
    X_row_fp32 = X_row.to(tl.float32)
    mean_square = tl.sum(X_row_fp32 * X_row_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # Normalize in fp32, cast back to original dtype before scaling
    X_norm = (X_row_fp32 * rstd).to(X_row_dtype)
    Y_row = X_norm * W_row

    # Store Y to HBM
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm using a fused Triton kernel.

    Args:
        x: Input tensor of shape (B, T, H) or (N, H).
        weight: Weight tensor of shape (H,).
        eps: Epsilon for numerical stability. Default: 1e-6.

    Returns:
        Normalized tensor of the same shape and dtype as x.
    """
    shape = x.shape
    n_cols = shape[-1]

    x = x.contiguous().view(-1, n_cols)
    weight = weight.contiguous()
    n_rows = x.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y = torch.empty_like(x)

    _rms_norm_forward_kernel[(n_rows,)](
        y,
        y.stride(0),
        x,
        x.stride(0),
        weight,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y.view(*shape)


class TritonRMSNorm(torch.nn.Module):
    """RMSNorm drop-in replacement backed by a fused Triton forward kernel.

    Optimized for inference: single HBM roundtrip with fp32 variance
    accumulation (llama casting mode). Supports float16 and bfloat16.

    Args:
        hidden_size: Size of the last dimension (e.g. 2048 for Qwen3-TTS).
        eps: Epsilon added to the variance for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize TritonRMSNorm.

        Args:
            hidden_size: Size of the last dimension (e.g. 2048 for Qwen3-TTS).
            eps: Epsilon added to the variance for numerical stability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape (B, T, H) or (N, H).

        Returns:
            Normalized tensor of the same shape and dtype as x.
        """
        return triton_rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        """Return a string of extra representation parameters.

        Returns:
            A string describing hidden_size and eps values.
        """
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
