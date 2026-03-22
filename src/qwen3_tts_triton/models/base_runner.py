"""Base Qwen3-TTS runner using HuggingFace transformers."""

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SAMPLE_RATE = 24000

# Map short language codes to full names for generate_custom_voice()
_LANG_MAP: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
}

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 2048

# dtype string to torch dtype mapping
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = dtype.lower().replace("bfloat16", "bf16").replace("float16", "fp16")
    key = key.replace("float32", "fp32")
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}'. Use: bf16, fp16, fp32")
    return _DTYPE_MAP[key]


def _to_numpy(audio: Any) -> np.ndarray:
    """Convert audio output to numpy array."""
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().float().numpy()
    return audio


class BaseRunner:
    """Load and run Qwen3-TTS from HuggingFace transformers.

    Args:
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID or local path.
        dtype: Model dtype ("bf16", "fp16", "fp32").
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = DEFAULT_MODEL_ID,
        dtype: str | torch.dtype = "bf16",
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.dtype = _resolve_dtype(dtype)
        self._tts: Any = None
        self._clone_tts: Any = None

    def load_model(self) -> None:
        """Download and load model + processor onto device."""
        from qwen_tts import Qwen3TTSModel

        logger.info("Loading %s ...", self.model_id)
        torch.cuda.reset_peak_memory_stats()

        self._tts = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=self.dtype,
        )

        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info("Model loaded. VRAM: %.2f GB", vram_gb)

    @property
    def model(self) -> Any:
        """Internal model (for patching)."""
        return self._tts.model if self._tts else None

    @property
    def processor(self) -> Any:
        """Processor/tokenizer."""
        return self._tts.processor if self._tts else None

    def _check_loaded(self) -> None:
        """Raise if model not loaded."""
        if self._tts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def generate(
        self,
        text: str,
        language: str = "English",
        speaker: str = "vivian",
        *,
        instruct: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> dict:
        """Generate speech from text using custom voice.

        Args:
            text: Input text to synthesize.
            language: Language name (e.g. "English", "Chinese").
            speaker: Speaker name for custom voice.
            instruct: Optional style instruction.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty (>1.0).
            max_new_tokens: Maximum tokens to generate.
            greedy: If True, disable sampling (deterministic).

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        torch.cuda.reset_peak_memory_stats()

        lang_name = _LANG_MAP.get(language, language)

        kwargs: dict = {
            "text": text,
            "language": lang_name,
            "speaker": speaker,
            "temperature": temperature,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy,
        }
        if instruct is not None:
            kwargs["instruct"] = instruct

        start = time.perf_counter()
        wavs, sr = self._tts.generate_custom_voice(**kwargs)
        elapsed = time.perf_counter() - start

        return {
            "audio": _to_numpy(wavs),
            "sample_rate": sr,
            "time_s": elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def generate_streaming(
        self,
        text: str,
        language: str = "English",
        speaker: str = "vivian",
        *,
        instruct: str | None = None,
        chunk_size: int = 12,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """Generate speech in streaming mode (chunk by chunk).

        Subclasses should override this. BaseRunner does not support streaming.

        Yields:
            (audio_chunk, sample_rate, timing_dict) per chunk.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming generation."
        )

    def _load_clone_model(self) -> Any:
        """Lazy-load the Base model for voice cloning.

        Voice cloning requires ``Qwen3-TTS-Base``, not ``CustomVoice``.
        The Base model is loaded on first call and cached in ``_clone_tts``.
        """
        if self._clone_tts is not None:
            return self._clone_tts

        from qwen_tts import Qwen3TTSModel

        logger.info("Loading clone model %s ...", CLONE_MODEL_ID)
        self._clone_tts = Qwen3TTSModel.from_pretrained(
            CLONE_MODEL_ID,
            device_map=self.device,
            dtype=self.dtype,
        )
        logger.info("Clone model loaded.")
        return self._clone_tts

    def generate_voice_clone(
        self,
        text: str,
        language: str = "English",
        ref_audio: str | Path = "",
        ref_text: str = "",
        *,
        xvec_only: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> dict:
        """Generate speech by cloning a reference voice.

        Uses the Base model (not CustomVoice) which supports voice cloning.

        Args:
            text: Input text to synthesize.
            language: Language name (e.g. "English", "Chinese").
            ref_audio: Path to reference audio file.
            ref_text: Transcription of the reference audio.
            xvec_only: Use only x-vector (no ICL acoustic context).
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty (>1.0).
            max_new_tokens: Maximum tokens to generate.
            greedy: If True, disable sampling (deterministic).

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        clone_tts = self._load_clone_model()
        torch.cuda.reset_peak_memory_stats()

        kwargs: dict[str, Any] = {
            "text": text,
            "language": language,
            "ref_audio": str(ref_audio),
            "x_vector_only_mode": xvec_only,
            "temperature": temperature,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        }
        if ref_text:
            kwargs["ref_text"] = ref_text
        if greedy:
            kwargs["temperature"] = 0.0
            kwargs["top_k"] = 1

        start = time.perf_counter()
        wavs, sr = clone_tts.generate_voice_clone(**kwargs)
        elapsed = time.perf_counter() - start

        return {
            "audio": _to_numpy(wavs),
            "sample_rate": sr,
            "time_s": elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str = "English",
        ref_audio: str | Path = "",
        ref_text: str = "",
        *,
        chunk_size: int = 12,
        xvec_only: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """Stream voice-cloned speech chunk by chunk.

        Subclasses should override this.

        Yields:
            (audio_chunk, sample_rate, timing_dict) per chunk.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming voice cloning."
        )

    def generate_voice_design(
        self,
        text: str,
        instruct: str = "",
        language: str = "English",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> dict:
        """Generate speech with a designed voice from instructions.

        Subclasses should override this.

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support voice design."
        )

    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str = "",
        language: str = "English",
        *,
        chunk_size: int = 12,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """Stream voice-designed speech chunk by chunk.

        Subclasses should override this.

        Yields:
            (audio_chunk, sample_rate, timing_dict) per chunk.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming voice design."
        )

    def list_speakers(self) -> list[str]:
        """List available speaker IDs from model config.

        Returns:
            List of speaker name strings.
        """
        self._check_loaded()
        try:
            spk_id = self._tts.model.config.talker_config.spk_id
            return sorted(spk_id.keys()) if isinstance(spk_id, dict) else []
        except AttributeError:
            return []

    def unload_model(self) -> None:
        """Free model from GPU memory."""
        del self._tts
        self._tts = None
        if self._clone_tts is not None:
            del self._clone_tts
            self._clone_tts = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")
