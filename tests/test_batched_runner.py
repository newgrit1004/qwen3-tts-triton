"""Tests for batched serving (length bucketing + per-row sampling).

Batched serving is a *capability* (``generate_batch``) on every runner, not a
separate ``batched-*`` runner identity, so the public interface stays the
v0.1.0/v0.2.0 seven-mode axis.  Pure-function tests (bucketing, per-sequence
repetition penalty, mask) run anywhere; the end-to-end generation test loads the
1.7B model and is gated behind both CUDA and ``RUN_BATCHED_RUNNER_TESTS=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from qwen3_tts_triton.models.batched import (
    batched_repetition_penalty,
    bucket_by_length,
    chunk_in_order,
)

# ---------------------------------------------------------------------------
# Interface: batched serving is a capability, not a runner name
# ---------------------------------------------------------------------------


def test_runner_names_match_v020_interface() -> None:
    """ALL_RUNNER_NAMES stays the v0.1.0/v0.2.0 7-mode axis (no batched-* names)."""
    from qwen3_tts_triton.models import ALL_RUNNER_NAMES

    assert ALL_RUNNER_NAMES == [
        "base",
        "base+tq",
        "triton",
        "triton+tq",
        "faster",
        "hybrid",
        "hybrid+tq",
    ]
    assert not any(name.startswith("batched-") for name in ALL_RUNNER_NAMES)


def test_every_runner_exposes_generate_batch() -> None:
    """Every canonical runner has generate_batch; the two engine families differ."""
    from qwen3_tts_triton.models import (
        BaseRunner,
        FasterRunner,
        TritonFasterRunner,
        TritonRunner,
        get_runner_class,
    )

    for name in ("base", "triton", "faster", "hybrid"):
        assert callable(getattr(get_runner_class(name), "generate_batch", None))
    # base/triton share the HF list-path implementation (defined on BaseRunner);
    # faster/hybrid share the CUDA-graph implementation (defined on FasterRunner).
    assert BaseRunner.generate_batch is TritonRunner.generate_batch
    assert FasterRunner.generate_batch is TritonFasterRunner.generate_batch
    assert BaseRunner.generate_batch is not FasterRunner.generate_batch


def test_no_batched_runner_classes_are_exported() -> None:
    """The internal batched engine must not leak into the public package API."""
    import qwen3_tts_triton as pkg

    assert not any("Batched" in name for name in pkg.__all__)


# ---------------------------------------------------------------------------
# Length bucketing
# ---------------------------------------------------------------------------


def test_bucket_by_length_groups_similar_lengths() -> None:
    lengths = [5, 1, 3, 2, 4]
    buckets = bucket_by_length(lengths, batch_size=2)
    # Sorted-by-length order is [1,3,2,4,5] -> indices [1,3,2,4,0].
    assert buckets == [[1, 3], [2, 4], [0]]


def test_bucket_by_length_covers_every_index_once() -> None:
    lengths = [9, 2, 7, 4, 1, 8, 3, 6]
    buckets = bucket_by_length(lengths, batch_size=3)
    flat = [i for b in buckets for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    assert all(len(b) <= 3 for b in buckets)
    # Each bucket's lengths are contiguous in the global sorted order.
    sorted_lengths = sorted(lengths)
    seen = [lengths[i] for b in buckets for i in b]
    assert seen == sorted_lengths


def test_chunk_in_order_preserves_submission_order() -> None:
    assert chunk_in_order(5, 2) == [[0, 1], [2, 3], [4]]
    assert chunk_in_order(0, 4) == []


def test_bucketing_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        bucket_by_length([1, 2], 0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        chunk_in_order(4, -1)


# ---------------------------------------------------------------------------
# Per-sequence repetition penalty
# ---------------------------------------------------------------------------


def test_repetition_penalty_is_sign_aware() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]])
    history = torch.tensor([[1, 3], [0, 0]])  # row1 repeats token 0
    out = batched_repetition_penalty(logits, history, 2.0)
    # Row 0: positive logits at 1,3 are divided.
    assert out[0].tolist() == [1.0, 1.0, 3.0, 2.0, 5.0]
    # Row 1: negative logit at 0 is multiplied (penalised once despite duplicate).
    assert out[1].tolist() == [-2.0, -2.0, -3.0, -4.0, -5.0]


def test_repetition_penalty_is_per_sequence() -> None:
    """Row b's history must not penalise row b'≠b (the flat-history bug)."""
    logits = torch.ones(2, 4)
    history = torch.tensor([[1], [2]])
    out = batched_repetition_penalty(logits.clone(), history, 2.0)
    assert out[0].tolist() == [1.0, 0.5, 1.0, 1.0]  # only token 1 penalised
    assert out[1].tolist() == [1.0, 1.0, 0.5, 1.0]  # only token 2 penalised


def test_repetition_penalty_noop_paths() -> None:
    logits = torch.randn(3, 6)
    ref = logits.clone()
    assert torch.equal(
        batched_repetition_penalty(logits, torch.zeros(3, 0, dtype=torch.long), 2.0),
        ref,
    )
    assert torch.equal(
        batched_repetition_penalty(logits, torch.tensor([[0], [1], [2]]), 1.0), ref
    )


# ---------------------------------------------------------------------------
# HF batched path helpers (generate_batch on base / triton)
# ---------------------------------------------------------------------------


def test_hf_row_audio_extracts_1d_float() -> None:
    from qwen3_tts_triton.models.base_runner import _row_audio

    # List of tensors (the generate_custom_voice batched return shape).
    wavs = [torch.zeros(1, 4), torch.ones(1, 3)]
    row0 = _row_audio(wavs, 0)
    row1 = _row_audio(wavs, 1)
    assert row0.ndim == 1 and row0.shape == (4,)
    assert row1.ndim == 1 and row1.tolist() == [1.0, 1.0, 1.0]


def test_hf_assemble_batch_result_schema() -> None:
    from qwen3_tts_triton.models.base_runner import _assemble_batch_result

    # Empty path: no rows, zeroed aggregates, batch_size echoed.
    empty = _assemble_batch_result([], {}, 0.0, 0, 32)
    assert empty["results"] == []
    assert empty["num_samples"] == 0
    assert empty["rtf"] == 0.0
    assert empty["batch_size"] == 32

    # Populated path: per-row schema + codec_steps from audio duration (12 Hz).
    audio = {0: (np.zeros(24000, dtype=np.float32), 24000)}
    out = _assemble_batch_result(["hi"], audio, 0.5, 1, 8)
    assert out["num_samples"] == 1
    assert out["results"][0]["text"] == "hi"
    assert out["results"][0]["codec_steps"] == 12  # 1.0s * 12 Hz
    assert out["total_audio_s"] == 1.0
    assert out["num_buckets"] == 1


def test_suppress_mask_keeps_eos_blocks_tail() -> None:
    from qwen3_tts_triton.models.batched import _suppress_mask

    vocab, eos = 2048, 1100  # eos inside the suppressed tail
    mask = _suppress_mask(vocab, eos, "cpu")
    assert mask.shape == (vocab,)
    assert mask[eos].item() is False  # EOS stays sample-able
    assert mask[2047].item() is True  # tail suppressed
    assert mask[0].item() is False  # head untouched
    assert int(mask.sum()) == 1024 - 1  # 1024 tail ids minus EOS


# ---------------------------------------------------------------------------
# End-to-end generation (GPU + opt-in)
# ---------------------------------------------------------------------------

_E2E_TEXTS = [
    "Hi.",
    "Good morning everyone, I hope you slept well.",
    "Thanks!",
    "Could you please send me the quarterly report before tomorrow afternoon?",
    "Okay.",
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    os.environ.get("RUN_BATCHED_RUNNER_TESTS") != "1",
    reason="set RUN_BATCHED_RUNNER_TESTS=1 to run the batched E2E generation test",
)
def test_faster_generate_batch_varlen_order_preserved() -> None:
    from qwen3_tts_triton.models import FasterRunner, create_runner

    runner = create_runner("faster")
    assert isinstance(runner, FasterRunner)
    runner.load_model()
    try:
        out = runner.generate_batch(
            _E2E_TEXTS,
            language="en",
            speaker="vivian",
            batch_size=4,
            greedy=True,
            max_new_tokens=128,
        )
        results = out["results"]
        assert out["num_samples"] == len(_E2E_TEXTS)
        # Original submission order is preserved despite internal bucketing.
        assert [r["text"] for r in results] == _E2E_TEXTS
        for r in results:
            assert r["audio"].ndim == 1
            assert r["audio"].size > 1
            assert r["codec_steps"] >= 1
        # Varied-length inputs must yield varied codec lengths.
        steps = [r["codec_steps"] for r in results]
        assert max(steps) > min(steps)
        assert out["rtf"] > 0
    finally:
        runner.unload_model()
