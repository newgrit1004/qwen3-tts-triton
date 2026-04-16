"""Triton-optimized Qwen3-TTS runner."""

import logging

from qwen3_tts_triton.models.base_runner import BaseRunner
from qwen3_tts_triton.models.patching import apply_triton_kernels

logger = logging.getLogger(__name__)


class TritonRunner(BaseRunner):
    """BaseRunner with Triton kernel patching applied after model load.

    Replaces RMSNorm, SwiGLU, Norm+Residual ops with fused Triton kernels
    across all talker and code predictor layers.

    Args:
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch.  Defaults to ``(0, 24)`` — keeps last 4 layers
            in PyTorch for pronunciation accuracy.  ``None`` patches all.
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string (``"bf16"``, ``"fp16"``, ``"fp32"``).
        enable_turboquant: Enable TurboQuant KV cache quantization.
        tq_bits: TurboQuant bit-width (3 or 4).
    """

    def __init__(
        self,
        patch_range: tuple[int, int] | None = (0, 24),
        device: str = "cuda",
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype: str = "bf16",
        enable_turboquant: bool = False,
        tq_bits: int = 4,
    ) -> None:
        super().__init__(
            device=device,
            model_id=model_id,
            dtype=dtype,
            enable_turboquant=enable_turboquant,
            tq_bits=tq_bits,
        )
        self.patch_range = patch_range

    def load_model(self) -> None:
        """Load model then apply Triton kernel patches."""
        super().load_model()
        apply_triton_kernels(
            self.model,
            patch_range=self.patch_range,
        )
        logger.info("TritonRunner ready (Triton kernels applied).")
