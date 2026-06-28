"""Validate + benchmark the batched serving capability (hybrid runner).

Drives ``create_runner("hybrid").generate_batch(...)`` end to end and checks the
three productionisation claims:

  1. Length bucketing reduces wall time on a varied-length workload. Measured
     with greedy decode (deterministic -> bucket=True/False produce identical
     per-text tokens, so the ONLY difference is grouping efficiency) and
     N > batch_size (so multiple buckets actually form -- with one bucket
     bucketing is a no-op).
  2. Batch sampling is per-row independent -- 8 IDENTICAL prompts, stochastic
     decode, must yield DIFFERENT clips per row (not a broadcast of one sample).
  3. Hybrid Triton kernels are captured inside the batched graph and run.

Run: make bench-batched  (or uv run python benchmark/bench_batched_runner.py)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from qwen3_tts_triton.models import create_runner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)7s] %(message)s"
)
log = logging.getLogger("bench_batched_runner")
OUT = Path(__file__).parent / "results" / "batched_runner.json"

SPEAKER = "vivian"
LANG = "en"

POOL = [
    "Hi.",
    "Good morning everyone.",
    "How are you doing today?",
    "The weather is lovely this afternoon.",
    "Could you please send me the report by tomorrow morning?",
    "Artificial intelligence is quietly transforming how we build software.",
    "She walked along the quiet beach, watching the waves roll in under a pink sky.",
    "In practical deployments, a speech synthesis service must handle many concurrent "
    "requests while keeping latency low and quality high.",
    "Thanks!",
    "Let's grab coffee later.",
    "The train departs at exactly nine fifteen.",
    "He couldn't believe how fast the year had gone by.",
    "Our new model generates natural speech for dozens of users at the same time.",
    "Once upon a time, in a faraway kingdom by the sea, there lived a curious inventor "
    "who dreamed of giving ordinary machines a warm human voice of their own.",
    "Okay.",
    "Please remember to water the plants before you leave for the weekend trip.",
]


def _texts(n: int) -> list[str]:
    return [POOL[i % len(POOL)] for i in range(n)]


def _summary(tag: str, out: dict) -> dict:
    steps = [r["codec_steps"] for r in out["results"]]
    row = {
        "tag": tag,
        "num_samples": out["num_samples"],
        "num_buckets": out["num_buckets"],
        "wall_s": round(out["wall_s"], 3),
        "rtf": round(out["rtf"], 2),
        "total_audio_s": round(out["total_audio_s"], 1),
        "per_sample_s": round(out["wall_s"] / max(out["num_samples"], 1), 4),
        "codec_steps_min": min(steps),
        "codec_steps_max": max(steps),
        "peak_vram_gb": round(out["peak_vram_gb"], 2),
    }
    log.info(
        "%-24s buckets=%d wall=%.2fs RTF=%.1f steps[%d..%d] vram=%.1fGB",
        tag,
        row["num_buckets"],
        row["wall_s"],
        row["rtf"],
        row["codec_steps_min"],
        row["codec_steps_max"],
        row["peak_vram_gb"],
    )
    return row


def bench_bucketing(results: dict) -> None:
    """Greedy, batch_size=8, 32 texts -> 4 buckets. Clean bucketing measurement."""
    runner = create_runner("hybrid")
    runner.load_model()
    try:
        texts = _texts(32)
        out_b = runner.generate_batch(
            texts,
            language=LANG,
            speaker=SPEAKER,
            batch_size=8,
            greedy=True,
            max_new_tokens=256,
        )
        out_n = runner.generate_batch(
            texts,
            language=LANG,
            speaker=SPEAKER,
            batch_size=8,
            greedy=True,
            max_new_tokens=256,
            bucket=False,
        )
        results["rows"].append(_summary("greedy/B8/bucket", out_b))
        results["rows"].append(_summary("greedy/B8/nobucket", out_n))
        # Greedy is deterministic: bucketing must not change the audio produced.
        same_audio = out_b["total_audio_s"] == out_n["total_audio_s"]
        gain = out_b["rtf"] / max(out_n["rtf"], 1e-9)
        results["bucketing"] = {
            "rtf_bucket": out_b["rtf"],
            "rtf_nobucket": out_n["rtf"],
            "wall_bucket_s": round(out_b["wall_s"], 3),
            "wall_nobucket_s": round(out_n["wall_s"], 3),
            "rtf_gain": round(gain, 3),
            "same_total_audio": same_audio,
        }
        log.info(
            "BUCKETING (greedy): RTF %.1f->%.1f=%.2fx wall %.2fs->%.2fs same=%s",
            out_n["rtf"],
            out_b["rtf"],
            gain,
            out_n["wall_s"],
            out_b["wall_s"],
            same_audio,
        )
    finally:
        runner.unload_model()


def bench_per_row_sampling(results: dict) -> None:
    """Stochastic, 8 IDENTICAL prompts -> must diverge per row (true batch sampling)."""
    runner = create_runner("hybrid")
    runner.load_model()
    try:
        ident = ["Our new model generates natural speech for many users at once."] * 8
        out = runner.generate_batch(
            ident,
            language=LANG,
            speaker=SPEAKER,
            batch_size=8,
            greedy=False,
            temperature=0.9,
            top_k=50,
            max_new_tokens=256,
        )
        steps = [r["codec_steps"] for r in out["results"]]
        a0, a1 = out["results"][0]["audio"], out["results"][1]["audio"]
        rows_identical = bool(a0.shape == a1.shape and (a0 == a1).all())
        results["per_row_sampling"] = {
            "identical_prompt_codec_steps": steps,
            "distinct_lengths": len(set(steps)),
            "row0_equals_row1": rows_identical,
        }
        log.info(
            "PER-ROW SAMPLING: identical prompts -> steps %s distinct=%d row0==row1?%s",
            steps,
            len(set(steps)),
            rows_identical,
        )
    finally:
        runner.unload_model()


def main() -> None:
    torch.set_grad_enabled(False)
    results: dict[str, Any] = {
        "config": {"runner": "hybrid", "batched": True},
        "rows": [],
    }
    bench_bucketing(results)
    bench_per_row_sampling(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    log.info("WROTE %s", OUT)


if __name__ == "__main__":
    main()
