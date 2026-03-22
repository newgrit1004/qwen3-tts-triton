"""Triton-optimized Qwen3-TTS runner."""

import logging

from qwen3_tts_triton.models.base_runner import BaseRunner
from qwen3_tts_triton.models.patching import apply_triton_kernels

logger = logging.getLogger(__name__)


class TritonRunner(BaseRunner):
    """BaseRunner with Triton kernel patching applied after model load.

    Replaces RMSNorm, SwiGLU, Norm+Residual, and M-RoPE ops with
    fused Triton kernels across all talker and code predictor layers.
    """

    def load_model(self) -> None:
        """Load model then apply Triton kernel patches."""
        super().load_model()
        apply_triton_kernels(self.model)
        logger.info("TritonRunner ready (Triton kernels applied).")
