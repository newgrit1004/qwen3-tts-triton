"""Hybrid runner: faster-qwen3-tts + Triton kernel patching."""

import logging

from qwen3_tts_triton.models.faster_runner import FasterRunner
from qwen3_tts_triton.models.patching import apply_triton_kernels, find_patchable_model

logger = logging.getLogger(__name__)


class TritonFasterRunner(FasterRunner):
    """FasterRunner with Triton kernel patches on internal model.

    Triton patches are applied BEFORE first generate() so that
    CUDA Graph capture includes the fused kernels.

    Args:
        enable_fused_norm: Enable fused norm+residual kernel. Default True.
        device: Target device (default: "cuda").
    """

    def __init__(
        self,
        enable_fused_norm: bool = True,
        device: str = "cuda",
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype: str = "bf16",
    ) -> None:
        """Initialize TritonFasterRunner.

        Args:
            enable_fused_norm: When ``True``, apply the fused residual-add +
                RMSNorm Triton kernel to every decoder layer in addition to
                the SwiGLU and RMSNorm patches.  Defaults to ``True``.
            device: PyTorch device string for model placement (e.g.
                ``"cuda"`` or ``"cuda:0"``).
            model_id: HuggingFace model ID.
            dtype: Model dtype string (``"bf16"``, ``"fp16"``, ``"fp32"``).
        """
        super().__init__(device=device, model_id=model_id, dtype=dtype)
        self.enable_fused_norm = enable_fused_norm

    def load_model(self) -> None:
        """Load faster-qwen3-tts model then apply Triton kernel patches."""
        super().load_model()  # FasterQwen3TTS.from_pretrained()
        # self.model = FasterQwen3TTS, self.model.model = Qwen3TTSModel wrapper
        internal = find_patchable_model(self.model.model)
        apply_triton_kernels(internal, enable_fused_norm=self.enable_fused_norm)
        logger.info("TritonFasterRunner ready (Triton kernels -> CUDA Graph)")
