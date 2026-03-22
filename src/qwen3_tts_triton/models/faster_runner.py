"""Wrapper around faster-qwen3-tts (optional dependency)."""

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qwen3_tts_triton.models.base_runner import (
    CLONE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    _resolve_dtype,
    _to_numpy,
)

logger = logging.getLogger(__name__)

# faster-qwen3-tts requires full language names, not short codes
_LANG_MAP: dict[str, str] = {
    "ko": "korean",
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
}


def _sampling_kwargs(
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    greedy: bool,
) -> dict:
    """Build a sampling keyword-arguments dict for faster-qwen3-tts generate calls.

    Args:
        temperature: Softmax temperature for sampling (higher = more random).
        top_k: Number of top-probability tokens to keep during sampling.
        repetition_penalty: Multiplicative penalty applied to already-generated
            token logits; values above 1.0 discourage repetition.
        max_new_tokens: Hard cap on the number of new tokens to generate.
        greedy: When ``True``, sets ``do_sample=False`` for deterministic
            greedy decoding regardless of temperature/top_k.

    Returns:
        Dict of keyword arguments ready to be unpacked into a generate call.
    """
    return {
        "temperature": temperature,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
    }


class FasterRunner:
    """Qwen3-TTS runner using the faster-qwen3-tts package.

    Exposes all faster-qwen3-tts APIs: custom voice, voice cloning,
    voice design, and streaming variants for each.

    Install the optional dependency first:
        uv add --optional faster faster-qwen3-tts

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
        self.model: Any = None
        self._clone_model: Any = None
        self._available = True

        try:
            import faster_qwen3_tts  # noqa: F401
        except ImportError:
            self._available = False
            logger.warning(
                "faster-qwen3-tts not installed. "
                "Install with: uv add --optional faster faster-qwen3-tts"
            )

    def load_model(self) -> None:
        """Load faster-qwen3-tts model."""
        if not self._available:
            raise ImportError(
                "faster-qwen3-tts is not installed. "
                "Install with: uv add --optional faster faster-qwen3-tts"
            )

        from faster_qwen3_tts import FasterQwen3TTS

        logger.info("Loading faster-qwen3-tts %s ...", self.model_id)
        torch.cuda.reset_peak_memory_stats()

        self.model = FasterQwen3TTS.from_pretrained(
            self.model_id,
            device=self.device,
            dtype=self.dtype,
        )

        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info("Model loaded. VRAM: %.2f GB", vram_gb)

    def _check_loaded(self) -> None:
        """Raise RuntimeError if the model has not been loaded yet.

        Raises:
            RuntimeError: If ``load_model()`` has not been called.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def _lang(self, language: str) -> str:
        """Resolve a short language code to the full name expected by faster-qwen3-tts.

        Args:
            language: ISO 639-1 code (e.g. ``"en"``, ``"zh"``) or a full
                language name that is passed through unchanged.

        Returns:
            Full lowercase language name (e.g. ``"english"``, ``"chinese"``).
        """
        return _LANG_MAP.get(language, language)

    def _result(
        self,
        wavs: list,
        sr: int | None,
        elapsed: float,
    ) -> dict:
        """Build the standard result dictionary returned by all generate methods.

        Args:
            wavs: Raw audio output from the model (list or tensor).
            sr: Sample rate reported by the model, or ``None`` to fall back to
                ``model.sample_rate`` / ``DEFAULT_SAMPLE_RATE``.
            elapsed: Wall-clock generation time in seconds.

        Returns:
            Dict with keys ``audio`` (np.ndarray), ``sample_rate`` (int),
            ``time_s`` (float), and ``peak_vram_gb`` (float).
        """
        return {
            "audio": _to_numpy(wavs),
            "sample_rate": sr
            or getattr(
                self.model,
                "sample_rate",
                DEFAULT_SAMPLE_RATE,
            ),
            "time_s": elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    # ------------------------------------------------------------------
    # Custom Voice (non-streaming / streaming)
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        language: str = "en",
        speaker: str = "vivian",
        *,
        instruct: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> dict:
        """Generate speech from text using custom voice API.

        Args:
            text: Input text to synthesize.
            language: Language code (e.g. "en", "zh") or full name.
            speaker: Speaker name for voice cloning.
            instruct: Optional style instruction.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty (>1.0).
            max_new_tokens: Maximum tokens to generate.
            greedy: If True, disable sampling.

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        torch.cuda.reset_peak_memory_stats()

        kwargs = {
            "text": text,
            "language": self._lang(language),
            "speaker": speaker,
            **_sampling_kwargs(
                temperature,
                top_k,
                repetition_penalty,
                max_new_tokens,
                greedy,
            ),
        }
        if instruct is not None:
            kwargs["instruct"] = instruct

        start = time.perf_counter()
        wavs, sr = self.model.generate_custom_voice(**kwargs)
        elapsed = time.perf_counter() - start
        return self._result(wavs, sr, elapsed)

    def generate_streaming(
        self,
        text: str,
        language: str = "en",
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
        """Stream custom-voice speech chunk by chunk.

        Args:
            text: Input text to synthesize.
            language: Language code (e.g. ``"en"``) or full name.
            speaker: Speaker name for custom voice.
            instruct: Optional style instruction.
            chunk_size: Number of codec tokens per yielded audio chunk.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty (>1.0).
            max_new_tokens: Maximum tokens to generate.
            greedy: If ``True``, disable sampling.

        Yields:
            Tuple of ``(audio_chunk, sample_rate, timing_dict)`` for each
            decoded chunk.
        """
        self._check_loaded()
        kwargs = {
            "text": text,
            "language": self._lang(language),
            "speaker": speaker,
            "chunk_size": chunk_size,
            **_sampling_kwargs(
                temperature,
                top_k,
                repetition_penalty,
                max_new_tokens,
                greedy,
            ),
        }
        if instruct is not None:
            kwargs["instruct"] = instruct
        gen = self.model.generate_custom_voice_streaming(**kwargs)
        yield from gen

    # ------------------------------------------------------------------
    # Voice Cloning (non-streaming / streaming)
    # ------------------------------------------------------------------

    def _load_clone_model(self) -> Any:
        """Lazy-load a FasterQwen3TTS with Base model for voice cloning."""
        if self._clone_model is not None:
            return self._clone_model

        from faster_qwen3_tts import FasterQwen3TTS

        logger.info("Loading clone model %s ...", CLONE_MODEL_ID)
        self._clone_model = FasterQwen3TTS.from_pretrained(
            CLONE_MODEL_ID,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info("Clone model loaded.")
        return self._clone_model

    def generate_voice_clone(
        self,
        text: str,
        language: str = "en",
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
            text: Text to synthesize.
            language: Language code or full name.
            ref_audio: Path to reference audio WAV.
            ref_text: Transcription of the reference audio.
            xvec_only: Use only x-vector (no ICL acoustic context).
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty.
            max_new_tokens: Maximum tokens.
            greedy: Disable sampling.

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        clone_model = self._load_clone_model()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        wavs, sr = clone_model.generate_voice_clone(
            text=text,
            language=self._lang(language),
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            xvec_only=xvec_only,
            **_sampling_kwargs(
                temperature,
                top_k,
                repetition_penalty,
                max_new_tokens,
                greedy,
            ),
        )
        elapsed = time.perf_counter() - start
        return self._result(wavs, sr, elapsed)

    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str = "en",
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

        Uses the Base model (not CustomVoice) which supports voice cloning.

        Args:
            text: Text to synthesize.
            language: Language code or full name.
            ref_audio: Path to the reference audio WAV file.
            ref_text: Transcription of the reference audio.
            chunk_size: Number of codec tokens per yielded audio chunk.
            xvec_only: Use only x-vector (no ICL acoustic context).
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty (>1.0).
            max_new_tokens: Maximum tokens to generate.
            greedy: If ``True``, disable sampling.

        Yields:
            Tuple of ``(audio_chunk, sample_rate, timing_dict)`` for each
            decoded chunk.
        """
        self._check_loaded()
        clone_model = self._load_clone_model()
        gen = clone_model.generate_voice_clone_streaming(
            text=text,
            language=self._lang(language),
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            chunk_size=chunk_size,
            xvec_only=xvec_only,
            **_sampling_kwargs(
                temperature,
                top_k,
                repetition_penalty,
                max_new_tokens,
                greedy,
            ),
        )
        yield from gen

    # ------------------------------------------------------------------
    # Voice Design (non-streaming / streaming)
    # ------------------------------------------------------------------

    def generate_voice_design(
        self,
        text: str,
        instruct: str = "",
        language: str = "en",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> dict:
        """Generate speech with a designed voice from instructions.

        Args:
            text: Text to synthesize.
            instruct: Voice/style design instruction.
            language: Language code or full name.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty.
            max_new_tokens: Maximum tokens.
            greedy: Disable sampling.

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        wavs, sr = self.model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=self._lang(language),
            **_sampling_kwargs(
                temperature,
                top_k,
                repetition_penalty,
                max_new_tokens,
                greedy,
            ),
        )
        elapsed = time.perf_counter() - start
        return self._result(wavs, sr, elapsed)

    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str = "",
        language: str = "en",
        *,
        chunk_size: int = 12,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = False,
    ) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """Stream voice-designed speech chunk by chunk.

        Args:
            text: Text to synthesize.
            instruct: Voice/style design instruction string.
            language: Language code or full name.
            chunk_size: Number of codec tokens per yielded audio chunk.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            repetition_penalty: Repetition penalty (>1.0).
            max_new_tokens: Maximum tokens to generate.
            greedy: If ``True``, disable sampling.

        Yields:
            Tuple of ``(audio_chunk, sample_rate, timing_dict)`` for each
            decoded chunk.
        """
        self._check_loaded()
        gen = self.model.generate_voice_design_streaming(
            text=text,
            instruct=instruct,
            language=self._lang(language),
            chunk_size=chunk_size,
            **_sampling_kwargs(
                temperature,
                top_k,
                repetition_penalty,
                max_new_tokens,
                greedy,
            ),
        )
        yield from gen

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_speakers(self) -> list[str]:
        """List available speaker IDs from model config.

        Returns:
            Sorted list of speaker name strings.
        """
        self._check_loaded()
        try:
            spk_id = self.model.model.config.talker_config.spk_id
            return sorted(spk_id.keys()) if isinstance(spk_id, dict) else []
        except AttributeError:
            return []

    def unload_model(self) -> None:
        """Free model from GPU memory."""
        del self.model
        self.model = None
        if self._clone_model is not None:
            del self._clone_model
            self._clone_model = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")
