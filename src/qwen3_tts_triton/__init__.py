"""Qwen3-TTS Triton kernel fusion for inference acceleration.

This package provides Triton-fused GPU kernels (RMSNorm, SwiGLU, M-RoPE,
Fused Norm+Residual) and model runner classes that transparently patch
Qwen3-TTS 1.7B for faster inference on CUDA-capable hardware.

Typical usage::

    from qwen3_tts_triton import TritonRunner

    runner = TritonRunner()
    runner.load_model()
    result = runner.generate(text="Hello", language="English", speaker="vivian")
    runner.unload_model()
"""

import warnings

__version__ = "0.1.0"


def _check_torch() -> None:
    """Verify PyTorch with CUDA support is installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "qwen3-tts-triton requires PyTorch with CUDA support. "
            "Install it first:\n"
            "  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. qwen3-tts-triton requires a CUDA-capable GPU.",
            stacklevel=2,
        )


_check_torch()

from qwen3_tts_triton.kernels import (  # noqa: E402
    TritonFusedAddRMSNorm,
    TritonRMSNorm,
    TritonSwiGLU,
    triton_fused_add_rms_norm,
    triton_mrope_forward,
    triton_rms_norm,
    triton_swiglu_forward,
)
from qwen3_tts_triton.models import (  # noqa: E402
    BaseRunner,
    FasterRunner,
    TritonFasterRunner,
    TritonRunner,
    apply_triton_kernels,
    get_runner_class,
)

__all__ = [
    "__version__",
    # Kernels
    "TritonRMSNorm",
    "TritonSwiGLU",
    "TritonFusedAddRMSNorm",
    "triton_rms_norm",
    "triton_swiglu_forward",
    "triton_mrope_forward",
    "triton_fused_add_rms_norm",
    # Models
    "BaseRunner",
    "FasterRunner",
    "TritonRunner",
    "TritonFasterRunner",
    "apply_triton_kernels",
    "get_runner_class",
]
