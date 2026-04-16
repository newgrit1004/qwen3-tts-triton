"""Triton kernels for Qwen3-TTS optimization.

This package exposes four fused Triton kernels designed for inference-time
acceleration of the Qwen3-TTS 1.7B Talker transformer:

- :class:`TritonRMSNorm` / :func:`triton_rms_norm`: fused RMSNorm (single
  HBM roundtrip, llama fp32 casting mode).
- :class:`TritonSwiGLU` / :func:`triton_swiglu_forward`: fused SwiGLU
  activation (``silu(gate) * up``).
- :func:`triton_mrope_forward`: fused Multi-dimensional Rotary Position
  Embedding (M-RoPE) with interleaved rotation and 3-section splitting.
- :class:`TritonFusedAddRMSNorm` / :func:`triton_fused_add_rms_norm`: fused
  residual-add + RMSNorm in one kernel launch.

All kernels are forward-only and optimised for CUDA devices (RTX 5090 /
sm_120, CUDA 12.8).
"""

from qwen3_tts_triton.kernels.fused_dequant import (
    triton_fused_dequant,
    triton_fused_quant,
)
from qwen3_tts_triton.kernels.fused_norm_residual import (
    TritonFusedAddRMSNorm,
    triton_fused_add_rms_norm,
)
from qwen3_tts_triton.kernels.rms_norm import TritonRMSNorm, triton_rms_norm
from qwen3_tts_triton.kernels.rope import triton_mrope_forward
from qwen3_tts_triton.kernels.swiglu import TritonSwiGLU, triton_swiglu_forward
from qwen3_tts_triton.kernels.turboquant import (
    TurboQuantKVCache,
    dequantize_vectors,
    generate_rotation_matrix,
    lloyd_max_codebook,
    pack_3bit,
    pack_4bit,
    quantize_vectors,
    unpack_3bit,
    unpack_4bit,
)

__all__ = [
    "triton_rms_norm",
    "TritonRMSNorm",
    "triton_swiglu_forward",
    "TritonSwiGLU",
    "triton_mrope_forward",
    "triton_fused_add_rms_norm",
    "TritonFusedAddRMSNorm",
    "TurboQuantKVCache",
    "quantize_vectors",
    "dequantize_vectors",
    "lloyd_max_codebook",
    "generate_rotation_matrix",
    "pack_4bit",
    "unpack_4bit",
    "pack_3bit",
    "unpack_3bit",
    "triton_fused_dequant",
    "triton_fused_quant",
]
