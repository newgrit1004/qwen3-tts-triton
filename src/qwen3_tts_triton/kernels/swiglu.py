"""Fused SwiGLU Triton kernel for inference (forward-only).

Computes: output = silu(gate) * up
where silu(x) = x * sigmoid(x)

Optimized for Qwen3-TTS Talker MLP intermediate_size=6144.
sigmoid is computed in float32 for numerical stability.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from qwen3_tts_triton.kernels.utils import calculate_settings


@triton.jit
def _swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU forward kernel: output = silu(gate) * up.

    Each program instance processes one row of the 2-D view of gate/up.
    sigmoid is computed in float32 for numerical stability; the result is
    cast back to the original dtype before the element-wise multiply.
    """
    program_id = tl.program_id(0).to(tl.int64)

    gate_ptr += program_id * stride
    up_ptr += program_id * stride
    out_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load gate in float32 for numerically stable sigmoid
    gate = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + col_offsets, mask=mask, other=0)

    # silu(gate) = gate * sigmoid(gate), all in float32
    silu_gate = gate * tl.sigmoid(gate)

    # Cast back to original dtype, then fused multiply
    out = silu_gate.cast(up.dtype) * up
    tl.store(out_ptr + col_offsets, out, mask=mask)


def triton_swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward pass using Triton kernel.

    Args:
        gate: Gate tensor of shape (..., intermediate_size). Must be on CUDA.
        up: Up-projection tensor of same shape as gate. Must be on CUDA.

    Returns:
        Output tensor of same shape and dtype as inputs.
    """
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("Tensors must be on CUDA")

    gate = gate.contiguous()
    up = up.contiguous()

    ori_shape = gate.shape
    n_cols = ori_shape[-1]

    gate_2d = gate.view(-1, n_cols)
    up_2d = up.view(-1, n_cols)
    out_2d = torch.empty_like(gate_2d)
    n_rows = gate_2d.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        gate_2d,
        up_2d,
        out_2d,
        out_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out_2d.view(*ori_shape)


class TritonSwiGLU(nn.Module):
    """SwiGLU activation using fused Triton kernel (inference-only).

    Computes: output = silu(gate) * up

    Drop-in replacement for the MLP activation in Qwen3-TTS Talker.
    No backward pass — optimized for inference.
    """

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Apply fused SwiGLU activation.

        Args:
            gate: Gate tensor of shape (..., intermediate_size). Must be on CUDA.
            up: Up-projection tensor of the same shape as gate. Must be on CUDA.

        Returns:
            Output tensor of the same shape and dtype as the inputs,
            computed as ``silu(gate) * up``.
        """
        return triton_swiglu_forward(gate, up)
