"""Internal batched CUDA-graph decode engine for the faster/hybrid runners.

This module is the throughput machinery behind ``FasterRunner.generate_batch``
(and, by inheritance, ``TritonFasterRunner`` = hybrid).  It is *not* a public
runner: there is no ``batched-*`` runner name.  ``FasterRunner`` lazily owns one
:class:`BatchedEngine` per (batch size, sampling) configuration and delegates
``generate_batch`` to it.

The stock ``faster``/``hybrid`` decode path is batch=1 (lowest single-clip
latency).  The engine widens the CUDA-graph decode to ``B`` clips at once — the
memory-bandwidth cost of reading the 1.7B weights every step is amortised across
the batch, so aggregate throughput scales far past the single-clip RTF.

Three productionisation pieces over the earlier spike:

* **Length bucketing** — variable-length inputs are sorted by token length and
  chunked into ``<= batch_size`` buckets so short clips don't idle while a long
  clip in the same batch keeps decoding.  (True per-token continuous batching
  does not help here: ``B`` is baked into the CUDA graph's static shapes, so
  evicting a finished row frees no graph compute — bucketing is the right lever.)
* **Batch sampling** — per-sequence stochastic sampling (temperature/top-k/top-p)
  and a *per-sequence* repetition penalty (a single flat history is wrong once
  rows diverge).
* **Per-sequence EOS** — each row stops at its own EOS; the batch ends when all
  rows finish or ``max_new_tokens`` is hit.

The hybrid path needs no special handling here: ``TritonFasterRunner`` patches
the Triton kernels into ``self.model`` before any ``generate_batch`` call, so the
graphs this engine captures already contain the fused kernels.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from qwen3_tts_triton.models.base_runner import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)

if TYPE_CHECKING:
    from qwen3_tts_triton.models.faster_runner import FasterRunner

logger = logging.getLogger(__name__)

_CODEC_HZ = 12.0  # Qwen3-TTS codec frame rate
_SUPPRESS_TAIL = 1024  # stock faster_qwen3_tts suppresses the last 1024 vocab ids


def bucket_by_length(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Group item indices into length-sorted buckets of at most ``batch_size``.

    Sorting by length keeps each bucket's clips similar in duration, which
    minimises both prefill left-padding and decode-step waste (short clips
    waiting on long ones).

    Args:
        lengths: Per-item input length (e.g. token count).
        batch_size: Maximum bucket size (the CUDA graph's static ``B``).

    Returns:
        A list of buckets; each bucket is a list of original indices, sorted so
        that shorter clips are grouped together.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def chunk_in_order(n: int, batch_size: int) -> list[list[int]]:
    """Chunk ``range(n)`` into in-order groups of at most ``batch_size``.

    Used when bucketing is disabled (preserves submission order).

    Args:
        n: Number of items.
        batch_size: Maximum chunk size.

    Returns:
        A list of index chunks in original order.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]


def batched_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply a per-sequence repetition penalty to batched logits in-place.

    Mirrors HuggingFace semantics independently for each row: a token that
    appears in row ``b``'s history has its logit divided (if > 0) or multiplied
    (if <= 0) by ``repetition_penalty``, penalising it exactly once regardless of
    how often it occurs.

    Args:
        logits: Float tensor ``[B, vocab]`` (modified in place).
        token_history: Long tensor ``[B, S]`` of each row's past token ids.
        repetition_penalty: HF-style penalty (> 1.0 discourages repetition).

    Returns:
        The same ``logits`` tensor, penalised.
    """
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits
    score = torch.gather(logits, 1, token_history)
    score = torch.where(
        score > 0, score / repetition_penalty, score * repetition_penalty
    )
    logits.scatter_(1, token_history, score)
    return logits


def _suppress_mask(
    vocab_size: int, eos_id: int, device: str | torch.device
) -> torch.Tensor:
    """Build the stock special-token suppression mask (last 1024 ids, keep EOS)."""
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    start = max(0, vocab_size - _SUPPRESS_TAIL)
    mask[start:] = True
    if start <= eos_id < vocab_size:
        mask[eos_id] = False
    return mask


class BatchedEngine:
    """Throughput-oriented batched decode engine bound to a loaded FasterRunner.

    Not a public runner — :meth:`FasterRunner.generate_batch` lazily owns one per
    ``(batch_size, sampling, max_seq_len)`` configuration and delegates to it.
    The engine reuses the runner's already-loaded model (and whatever Triton
    patches it applied), so the hybrid path comes for free.

    Sampling parameters are baked into the predictor's CUDA graph at capture
    time, so they are fixed for an engine instance.  ``repetition_penalty`` and
    the token budgets stay per-call (they act on the talker token, outside the
    graph).

    Args:
        runner: The loaded :class:`FasterRunner` (or subclass) to decode with.
        batch_size: Static micro-batch size ``B`` for the CUDA graphs.
        temperature: Sampling temperature (baked into predictor graph).
        top_k: Top-k cutoff (baked).
        top_p: Nucleus cutoff (baked).
        greedy: Deterministic decoding (``do_sample=False``) when True.
        max_seq_len: Static-cache length bound for the graphs.
    """

    def __init__(
        self,
        runner: FasterRunner,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        greedy: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        self.runner = runner
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.greedy = greedy
        self.do_sample = not greedy
        self.max_seq_len = max_seq_len
        # Captured graphs keyed by batch size B (persist across generate_batch calls).
        self._graph_cache: dict[int, tuple[Any, Any]] = {}

    # ------------------------------------------------------------------
    # Graph + input construction
    # ------------------------------------------------------------------

    def _get_graphs(self, batch: int, talker: Any, config: Any) -> tuple[Any, Any]:
        """Return (talker_graph, predictor_graph) for ``batch``, capturing once."""
        cached = self._graph_cache.get(batch)
        if cached is not None:
            return cached

        from qwen3_tts_triton.models.batched_graphs import (
            FASTER_AVAILABLE,
            BatchedPredictorGraph,
            BatchedTalkerGraph,
        )

        if not FASTER_AVAILABLE:  # pragma: no cover - guarded by _available
            raise ImportError("faster-qwen3-tts is required for batched generation")

        talker_hidden = config.hidden_size
        pred_config = talker.code_predictor.model.config
        tg = BatchedTalkerGraph(
            talker.model,
            config,
            batch,
            device=self.runner.device,
            dtype=self.runner.dtype,
            max_seq_len=self.max_seq_len,
        )
        pg = BatchedPredictorGraph(
            talker.code_predictor,
            pred_config,
            talker_hidden,
            batch,
            device=self.runner.device,
            dtype=self.runner.dtype,
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        logger.info("Capturing batched CUDA graphs for B=%d ...", batch)
        pg.capture(num_warmup=3)
        tg.capture(prefill_len=64, num_warmup=3)
        self._graph_cache[batch] = (tg, pg)
        return tg, pg

    def _build_inputs(self, input_ids: list, language: str, speaker: str) -> tuple:
        """Build batched talker inputs from pre-tokenised ids for one bucket."""
        fq = self.runner.model
        m = fq.model.model
        batch = len(input_ids)
        tie, tam, tth, tpe = fq._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=[None] * batch,
            voice_clone_prompt=None,
            languages=[language] * batch,
            speakers=[speaker] * batch,
            non_streaming_mode=False,
            instruct_ids=[None] * batch,
        )
        talker = m.talker
        talker.rope_deltas = None
        return tie, tam, tth, tpe, talker, m.config.talker_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        texts: list[str],
        language: str = "en",
        speaker: str = "vivian",
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        bucket: bool = True,
    ) -> dict:
        """Synthesise many clips at once on the batched CUDA-graph path.

        Args:
            texts: Input strings (any mix of lengths).
            language: Language code/name (single value for the whole call).
            speaker: Custom-voice speaker id (single value for the whole call).
            max_new_tokens: Hard per-clip decode-step cap.
            min_new_tokens: Steps before EOS is permitted.
            repetition_penalty: Per-sequence penalty (> 1.0 discourages repeats).
            bucket: Length-bucket the inputs (recommended for varied lengths).

        Returns:
            Dict with ``results`` (per-input ``{audio, sample_rate, codec_steps,
            text}`` in original order) plus aggregate ``wall_s``/``rtf``/
            ``total_audio_s``/``peak_vram_gb``/``num_buckets``/``batch_size``.
        """
        self.runner._check_loaded()
        if not texts:
            return self._empty_result()

        fq = self.runner.model
        fq.max_seq_len = self.max_seq_len
        lang = self.runner._lang(language)
        fq.model._validate_languages([lang])
        fq.model._validate_speakers([speaker])

        assistant = [fq.model._build_assistant_text(t) for t in texts]
        all_ids = fq.model._tokenize_texts(assistant)
        lengths = [int(x.shape[-1]) for x in all_ids]
        buckets = (
            bucket_by_length(lengths, self.batch_size)
            if bucket
            else chunk_in_order(len(texts), self.batch_size)
        )

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        codec_by_index: dict[int, torch.Tensor | None] = {}
        for idx_group in buckets:
            sub_ids = [all_ids[i] for i in idx_group]
            codecs = self._generate_bucket(
                sub_ids,
                lang,
                speaker,
                max_new=max_new_tokens,
                min_new=min_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            for local, original in enumerate(idx_group):
                codec_by_index[original] = codecs[local]
        wall = time.perf_counter() - start

        codec_per_index = [codec_by_index[i] for i in range(len(texts))]
        return self._decode_results(texts, codec_per_index, wall, len(buckets))

    @torch.inference_mode()
    def _generate_bucket(
        self,
        input_ids: list,
        language: str,
        speaker: str,
        *,
        max_new: int,
        min_new: int,
        repetition_penalty: float,
    ) -> list[torch.Tensor | None]:
        """Decode one bucket; returns per-row codec ids ``[Sᵢ, 16]`` (EOS-trimmed)."""
        from faster_qwen3_tts.sampling import sample_logits

        tie, tam, tth, tpe, talker, config = self._build_inputs(
            input_ids, language, speaker
        )
        batch = tie.shape[0]
        tg, pg = self._get_graphs(batch, talker, config)
        eos_id = config.codec_eos_token_id
        ncg = config.num_code_groups
        device = tie.device
        embed = talker.get_input_embeddings()
        head = talker.codec_head
        pred_embeds = talker.code_predictor.get_input_embeddings()
        smask = _suppress_mask(config.vocab_size, eos_id, device)

        out = talker.forward(
            inputs_embeds=tie,
            attention_mask=tam,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=tth,
            tts_pad_embed=tpe,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        past_hidden = out.past_hidden
        gen_step = out.generation_step
        token = sample_logits(
            out.logits[:, -1, :],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            suppress_mask=smask,
            suppress_tokens=[eos_id] if min_new > 0 else None,
        )
        prefill_len = tg.prefill_kv(out.past_key_values)
        tg.set_generation_state(tam, getattr(talker, "rope_deltas", None))

        lengths = torch.full((batch,), max_new, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        history: list[torch.Tensor] = []
        all_codec: list[torch.Tensor] = []
        for step in range(max_new):
            if step >= min_new:
                newly = (token == eos_id) & (~finished)
                if newly.any():
                    lengths = torch.where(
                        newly, torch.full_like(lengths, len(all_codec)), lengths
                    )
                finished = finished | (token == eos_id)
                if bool(finished.all()):
                    break
            last = embed(token.unsqueeze(1))
            cb = pg.run(torch.cat((past_hidden, last), dim=1))
            all_codec.append(torch.cat([token.unsqueeze(1), cb], dim=1))
            history.append(token)
            hiddens = [last]
            for i in range(ncg - 1):
                hiddens.append(pred_embeds[i](cb[:, i].unsqueeze(1)))
            ie = torch.cat(hiddens, dim=1).sum(1, keepdim=True)
            ie = ie + (
                tth[:, gen_step].unsqueeze(1) if gen_step < tth.shape[1] else tpe
            )
            pos = prefill_len + step
            if pos >= tg.max_seq_len - 1:
                break
            hid = tg.run(ie, position=pos)
            logits = head(hid[:, -1, :])
            if repetition_penalty != 1.0 and history:
                logits = batched_repetition_penalty(
                    logits, torch.stack(history, dim=1), repetition_penalty
                )
            token = sample_logits(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
                suppress_mask=smask,
                suppress_tokens=[eos_id] if len(all_codec) < min_new else None,
            )
            past_hidden = hid[:, -1:, :].clone()
            gen_step += 1

        lengths = torch.where(
            finished, lengths, torch.full_like(lengths, len(all_codec))
        )
        if not all_codec:
            return [None] * batch
        codec_full = torch.stack(all_codec, dim=1)
        lens = lengths.tolist()
        return [codec_full[i, : lens[i]] for i in range(batch)]

    # ------------------------------------------------------------------
    # Decoding + result assembly
    # ------------------------------------------------------------------

    def _decode_results(
        self,
        texts: list[str],
        codec_per_index: list[torch.Tensor | None],
        wall: float,
        num_buckets: int,
    ) -> dict:
        """Decode codec ids to audio and assemble the result dict (original order)."""
        speech_tokenizer = self.runner.model.model.model.speech_tokenizer
        results: list[dict] = []
        total_steps = 0
        for text, codec in zip(texts, codec_per_index, strict=True):
            if codec is None or codec.shape[0] == 0:
                results.append(
                    {
                        "audio": np.zeros(1, dtype=np.float32),
                        "sample_rate": self.runner.model.sample_rate,
                        "codec_steps": 0,
                        "text": text,
                    }
                )
                continue
            audio_list, sr = speech_tokenizer.decode(
                {"audio_codes": codec.unsqueeze(0)}
            )
            audio = audio_list[0]
            audio = (
                audio.flatten().cpu().numpy()
                if hasattr(audio, "cpu")
                else np.asarray(audio).flatten()
            )
            total_steps += int(codec.shape[0])
            results.append(
                {
                    "audio": audio,
                    "sample_rate": sr,
                    "codec_steps": int(codec.shape[0]),
                    "text": text,
                }
            )

        total_audio_s = total_steps / _CODEC_HZ
        return {
            "results": results,
            "num_samples": len(results),
            "total_audio_s": total_audio_s,
            "wall_s": wall,
            "rtf": (total_audio_s / wall) if wall > 0 else 0.0,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "num_buckets": num_buckets,
            "batch_size": self.batch_size,
        }

    def _empty_result(self) -> dict:
        return {
            "results": [],
            "num_samples": 0,
            "total_audio_s": 0.0,
            "wall_s": 0.0,
            "rtf": 0.0,
            "peak_vram_gb": 0.0,
            "num_buckets": 0,
            "batch_size": self.batch_size,
        }
