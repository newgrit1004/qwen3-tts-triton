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
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 2048

# Batched-serving defaults (generate_batch). batch_size is a per-call argument
# (the runner constructors keep the v0.1.0/v0.2.0 signature).
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_NEW_TOKENS = 2
_CODEC_HZ = 12.0  # Qwen3-TTS codec frame rate

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


def _row_audio(wavs: Any, local: int) -> np.ndarray:
    """Extract the ``local``-th waveform from a batched output as 1-D float32."""
    w = wavs[local] if isinstance(wavs, list | tuple) else wavs
    if isinstance(w, torch.Tensor):
        return w.squeeze().detach().cpu().float().numpy().reshape(-1)
    return np.asarray(w, dtype=np.float32).reshape(-1)


def _assemble_batch_result(
    texts: list[str],
    audio_by_index: dict[int, tuple[np.ndarray, int]],
    wall: float,
    num_buckets: int,
    batch_size: int,
) -> dict:
    """Build the batched result dict (original order) from per-row audio.

    Schema matches ``FasterRunner.generate_batch`` so both batched
    families plug into the same eval/bench code paths.
    """
    results: list[dict] = []
    total_audio_s = 0.0
    for i, text in enumerate(texts):
        audio, sr = audio_by_index[i]
        dur = len(audio) / sr if sr > 0 else 0.0
        total_audio_s += dur
        results.append(
            {
                "audio": audio,
                "sample_rate": sr,
                "codec_steps": int(round(dur * _CODEC_HZ)),
                "text": text,
            }
        )
    return {
        "results": results,
        "num_samples": len(results),
        "total_audio_s": total_audio_s,
        "wall_s": wall,
        "rtf": (total_audio_s / wall) if wall > 0 else 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "num_buckets": num_buckets,
        "batch_size": batch_size,
    }


def _hf_generate_batch(
    runner: "BaseRunner",
    texts: list[str],
    language: str,
    speaker: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    repetition_penalty: float,
    temperature: float,
    top_k: int,
    greedy: bool,
    bucket: bool,
) -> dict:
    """Synthesise many clips through the HF native list API, one call per bucket.

    ``base``/``triton`` need no CUDA-graph fork: ``generate_custom_voice``
    accepts a list and HuggingFace handles left-padding + per-sequence EOS.
    Bucketing (by character length — a grouping-only proxy) keeps short clips
    from padding up to a long clip in the same ``model.generate`` call.
    """
    from qwen3_tts_triton.models.batched import bucket_by_length, chunk_in_order

    runner._check_loaded()
    bs = batch_size
    if not texts:
        return _assemble_batch_result([], {}, 0.0, 0, bs)
    lang_name = _LANG_MAP.get(language, language)
    lengths = [len(t) for t in texts]
    buckets = (
        bucket_by_length(lengths, bs) if bucket else chunk_in_order(len(texts), bs)
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    audio_by_index: dict[int, tuple[np.ndarray, int]] = {}
    for group in buckets:
        wavs, sr = runner._tts.generate_custom_voice(
            text=[texts[i] for i in group],
            language=lang_name,
            speaker=speaker,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
        )
        for local, original in enumerate(group):
            audio_by_index[original] = (_row_audio(wavs, local), int(sr))
    wall = time.perf_counter() - start
    return _assemble_batch_result(texts, audio_by_index, wall, len(buckets), bs)


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
        enable_turboquant: bool = False,
        tq_bits: int = 4,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.dtype = _resolve_dtype(dtype)
        self.enable_turboquant = enable_turboquant
        self.tq_bits = tq_bits
        self._tts: Any = None
        self._clone_tts: Any = None
        self._tq_cache: Any = None

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

        if self.enable_turboquant:
            self._init_turboquant_cache()

        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info("Model loaded. VRAM: %.2f GB", vram_gb)

    def _init_turboquant_cache(self) -> None:
        """Create TurboQuantKVCache and patch talker's generate().

        Wraps the talker's ``generate()`` so each call resets the cache
        and injects it as ``past_key_values``.  This makes HuggingFace's
        generation loop store/retrieve KV through our quantized cache.
        """
        from qwen3_tts_triton.kernels.turboquant import TurboQuantKVCache

        config = self._tts.model.talker.config
        self._tq_cache = TurboQuantKVCache(
            bits=self.tq_bits,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype,
        )

        # Monkey-patch talker.generate() to inject TQ cache
        talker = self._tts.model.talker
        original_generate = talker.generate

        tq_cache = self._tq_cache

        def _tq_generate(*args: Any, **kwargs: Any) -> Any:
            tq_cache.reset()
            kwargs.setdefault("past_key_values", tq_cache)
            return original_generate(*args, **kwargs)

        talker.generate = _tq_generate
        logger.info(
            "TurboQuant %d-bit KV cache injected into talker.generate()",
            self.tq_bits,
        )

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

    @torch.inference_mode()
    def generate_batch(
        self,
        texts: list[str],
        language: str = "en",
        speaker: str = "vivian",
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        greedy: bool = False,
        bucket: bool = True,
    ) -> dict:
        """Synthesise many clips at once via the HF native batched path.

        ``base``/``triton`` batch with no CUDA-graph fork — ``generate_custom_voice``
        natively accepts a list of texts (HuggingFace left-pads them and applies
        per-sequence EOS).  ``TritonRunner`` inherits this unchanged, so its
        Triton kernels run on every batched call.  Sampling is per-call here
        (unlike the faster/hybrid CUDA-graph path, which bakes it into the graph).

        Args:
            texts: Input strings (any mix of lengths).
            language: Language code/name (one value for the whole call).
            speaker: Custom-voice speaker id (one value for the whole call).
            batch_size: Max clips per ``generate_custom_voice`` call (bucket size).
            max_new_tokens: Hard per-clip decode cap (HF stops earlier at EOS).
            repetition_penalty: HF repetition penalty (> 1.0 discourages repeats).
            temperature: Sampling temperature.
            top_k: Top-k cutoff.
            greedy: Deterministic decoding (``do_sample=False``) when True.
            bucket: Length-bucket the inputs (recommended for varied lengths).

        Returns:
            Dict with the same schema as ``FasterRunner.generate_batch``:
            ``results`` (per-input ``{audio, sample_rate, codec_steps, text}`` in
            original order) plus aggregate ``wall_s``/``rtf``/``total_audio_s``/
            ``peak_vram_gb``/``num_buckets``/``batch_size``.
        """
        return _hf_generate_batch(
            self,
            texts,
            language,
            speaker,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
            bucket=bucket,
        )

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
