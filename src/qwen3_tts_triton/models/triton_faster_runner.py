"""Hybrid runner: faster-qwen3-tts + Triton kernel patching."""

import logging
from typing import Any

import torch

from qwen3_tts_triton.models.faster_runner import FasterRunner
from qwen3_tts_triton.models.patching import apply_triton_kernels, find_patchable_model

logger = logging.getLogger(__name__)


class TritonFasterRunner(FasterRunner):
    """FasterRunner with Triton kernel patches on internal model.

    Triton patches are applied BEFORE first generate() so that
    CUDA Graph capture includes the fused kernels.

    Args:
        enable_fused_norm: Enable fused norm+residual kernel. Default True.
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch.  Defaults to ``(0, 24)`` — keeps last 4 layers
            in PyTorch for pronunciation accuracy.  ``None`` patches all.
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string (``"bf16"``, ``"fp16"``, ``"fp32"``).
    """

    def __init__(
        self,
        enable_fused_norm: bool = True,
        patch_range: tuple[int, int] | None = (0, 24),
        device: str = "cuda",
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype: str = "bf16",
        enable_turboquant: bool = False,
        tq_bits: int = 4,
    ) -> None:
        super().__init__(device=device, model_id=model_id, dtype=dtype)
        self.enable_fused_norm = enable_fused_norm
        self.enable_turboquant = enable_turboquant
        self.tq_bits = tq_bits
        self.patch_range = patch_range

    def load_model(self) -> None:
        """Load faster-qwen3-tts model then apply Triton kernel patches."""
        super().load_model()  # FasterQwen3TTS.from_pretrained()
        # self.model = FasterQwen3TTS, self.model.model = Qwen3TTSModel wrapper
        internal = find_patchable_model(self.model.model)
        apply_triton_kernels(
            internal,
            enable_fused_norm=self.enable_fused_norm,
            patch_range=self.patch_range,
        )

        if self.enable_turboquant:
            self._init_turboquant_hybrid()

        logger.info("TritonFasterRunner ready (Triton kernels -> CUDA Graph)")

    def _init_turboquant_hybrid(self) -> None:
        """Create TurboQuantKVCache for the hybrid runner.

        FasterRunner wraps the model differently — the talker is nested
        inside ``self.model.model`` (FasterQwen3TTS → Qwen3TTSModel).
        """
        from qwen3_tts_triton.kernels.turboquant import TurboQuantKVCache

        internal = find_patchable_model(self.model.model)
        talker = getattr(internal, "talker", internal)
        config = getattr(talker, "config")

        self._tq_cache = TurboQuantKVCache(
            bits=self.tq_bits,
            num_layers=getattr(config, "num_hidden_layers"),
            num_kv_heads=getattr(config, "num_key_value_heads"),
            head_dim=getattr(config, "hidden_size")
            // getattr(config, "num_attention_heads"),
            device=self.device,
            dtype=torch.bfloat16,
        )

        original_generate = getattr(talker, "generate")
        tq_cache = self._tq_cache

        def _tq_generate(*args: Any, **kwargs: Any) -> Any:
            tq_cache.reset()
            kwargs.setdefault("past_key_values", tq_cache)
            return original_generate(*args, **kwargs)

        setattr(talker, "generate", _tq_generate)
        logger.info("TurboQuant %d-bit KV cache injected (hybrid)", self.tq_bits)
