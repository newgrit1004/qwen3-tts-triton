"""Batched CUDA-graph wrappers for the faster-qwen3-tts decode path.

``faster-qwen3-tts`` ships ``TalkerGraph`` / ``PredictorGraph`` with the batch
dimension hard-coded to 1 (single-clip latency path).  These subclasses widen
every static buffer to ``(B, ...)`` and swap in ``StaticCache(max_batch_size=B)``
so the *same* CUDA-graph machinery synthesises ``B`` clips per replay.

The wrappers are intentionally thin: the decode step / 15-codebook loop bodies
are inherited unchanged from the base classes (they are written shape-generic),
so the Triton-kernel patches applied by the hybrid runner are captured here too.

Requires the optional ``faster`` extra (``uv sync --extra faster``).
"""

from __future__ import annotations

import torch
from transformers import StaticCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

try:  # optional dependency — see FasterRunner for the user-facing guard
    from faster_qwen3_tts.predictor_graph import PredictorGraph as _BasePredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph as _BaseTalkerGraph

    FASTER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _BaseTalkerGraph = object  # type: ignore[assignment,misc]
    _BasePredictorGraph = object  # type: ignore[assignment,misc]
    FASTER_AVAILABLE = False


class BatchedTalkerGraph(_BaseTalkerGraph):  # type: ignore[valid-type,misc]
    """``TalkerGraph`` with batch dimension ``B`` baked into the static buffers.

    Args:
        talker_model: The talker transformer backbone (``talker.model``).
        talker_config: The talker sub-config (``config.talker_config``).
        batch: Static batch size ``B`` baked into the graph.
        device: CUDA device string.
        dtype: Compute dtype (must match the loaded model).
        max_seq_len: Static-cache / attention-mask length bound.
    """

    def __init__(
        self,
        talker_model: torch.nn.Module,
        talker_config: object,
        batch: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__(
            talker_model,
            talker_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        self.batch = batch
        hidden = self.hidden_size
        self.static_cache = StaticCache(
            config=talker_config,
            max_batch_size=batch,
            max_cache_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.input_buf = torch.zeros(batch, 1, hidden, dtype=dtype, device=device)
        self.output_buf = torch.zeros(batch, 1, hidden, dtype=dtype, device=device)
        self.rope_deltas = torch.zeros(batch, 1, dtype=torch.float32, device=device)
        self.position_ids = torch.zeros(3, batch, 1, dtype=torch.float32, device=device)

    def _init_cache_layers(self) -> None:
        config = self.model.config
        num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dummy_k = torch.zeros(
            self.batch, num_kv, 1, head_dim, dtype=self.dtype, device=self.device
        )
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _build_attention_masks(
        self, attention_mask: torch.Tensor | None = None
    ) -> None:
        dummy = torch.zeros(
            self.batch, 1, self.hidden_size, dtype=self.dtype, device=self.device
        )
        max_len = self.max_seq_len
        mask_fn = (
            create_causal_mask
            if self.model.config.sliding_window is None
            else create_sliding_window_causal_mask
        )
        table = [
            mask_fn(
                config=self.model.config,
                input_embeds=dummy,
                attention_mask=attention_mask,
                cache_position=torch.tensor([i], device=self.device),
                past_key_values=self.static_cache,
            )
            for i in range(max_len)
        ]
        self.attn_mask_table = table
        if self.attn_mask is None:
            self.attn_mask = table[0].clone()  # type: ignore[unresolved-attribute]
        else:
            self.attn_mask.copy_(table[0])


class BatchedPredictorGraph(_BasePredictorGraph):  # type: ignore[valid-type,misc]
    """``PredictorGraph`` with batch dimension ``B`` baked into the 15-step loop.

    Sampling parameters (``do_sample``/``temperature``/``top_k``/``top_p``) are
    captured *inside* the graph, so they are fixed for the lifetime of the
    captured graph.  CUDA-graph RNG advances per replay, so ``do_sample=True``
    still yields independent draws across decode steps.

    Args:
        code_predictor: The talker's ``code_predictor`` module.
        pred_config: The predictor's inner-model config.
        talker_hidden: Talker hidden size (predictor input projection width).
        batch: Static batch size ``B``.
        device: CUDA device string.
        dtype: Compute dtype.
        do_sample: Stochastic vs greedy codebook sampling.
        top_k: Top-k cutoff (sampling only).
        top_p: Nucleus cutoff (sampling only).
        temperature: Softmax temperature (sampling only).
    """

    def __init__(
        self,
        code_predictor: torch.nn.Module,
        pred_config: object,
        talker_hidden: int,
        batch: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> None:
        super().__init__(
            code_predictor,
            pred_config,
            talker_hidden,
            device=device,
            dtype=dtype,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        self.batch = batch
        self.static_cache = StaticCache(
            config=pred_config,
            max_batch_size=batch,
            max_cache_len=self.max_seq,
            device=device,
            dtype=dtype,
        )
        self.input_buf = torch.zeros(
            batch, 2, talker_hidden, dtype=dtype, device=device
        )
        self.output_tokens = torch.zeros(
            batch, self.num_codebooks, dtype=torch.long, device=device
        )

    def _init_cache_layers(self) -> None:
        config = self.pred_model.config
        num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dummy_k = torch.zeros(
            self.batch, num_kv, 1, head_dim, dtype=self.dtype, device=self.device
        )
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _build_attention_masks(self) -> None:
        dummy_prefill = torch.zeros(
            self.batch, 2, self.hidden_size, dtype=self.dtype, device=self.device
        )
        dummy_decode = torch.zeros(
            self.batch, 1, self.hidden_size, dtype=self.dtype, device=self.device
        )
        self.prefill_attn = self._make_attn_mask(dummy_prefill, self.prefill_cache_pos)
        self.decode_attn = [
            self._make_attn_mask(dummy_decode, pos)
            for pos in self.decode_cache_positions
        ]

    def _full_loop(self) -> torch.Tensor:
        from faster_qwen3_tts.sampling import sample_logits

        h = self.small_to_mtp(self.input_buf)
        out = self.pred_model(
            inputs_embeds=h,
            attention_mask=self.prefill_attn,
            past_key_values=self.static_cache,
            cache_position=self.prefill_cache_pos,
            use_cache=True,
        )
        h = out.last_hidden_state
        logits = self.lm_heads[0](h[:, -1:, :])
        tok = sample_logits(
            logits[:, 0, :],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        self.output_tokens[:, 0] = tok
        for cb in range(1, self.num_codebooks):
            emb = self.codec_embeds[cb - 1](tok.unsqueeze(1))
            emb = self.small_to_mtp(emb)
            out = self.pred_model(
                inputs_embeds=emb,
                attention_mask=self.decode_attn[cb - 1],
                past_key_values=self.static_cache,
                cache_position=self.decode_cache_positions[cb - 1],
                use_cache=True,
            )
            h = out.last_hidden_state
            logits = self.lm_heads[cb](h[:, -1:, :])
            tok = sample_logits(
                logits[:, 0, :],
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            self.output_tokens[:, cb] = tok
        return self.output_tokens
