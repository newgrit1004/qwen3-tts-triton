"""4-way batched comparison: every canonical runner under batched serving.

Batched serving is a capability — ``runner.generate_batch(texts, batch_size=B)``
— shared by all four canonical runners (base, triton, faster, hybrid); there is
no separate ``batched-*`` runner name.  This script runs all four on the SAME
varied-length workload and prints one table so the differences are directly
comparable.

Read the table on two axes:

* **Engine family (the big axis)** — ``base``/``triton`` use the HF eager
  attention path (batched via the list API); ``faster``/``hybrid`` use the
  faster-qwen3-tts static-cache + CUDA-graph path (batch baked into the graph).
  Cross-family numbers are a serving-throughput leaderboard (different decode
  engines), not an isolation of batching alone.
* **Triton kernels (the small axis)** — ``base`` → ``triton`` and ``faster`` →
  ``hybrid`` are the SAME engine with Triton kernels added, so those deltas are
  the kernel contribution under batch (expected small: the elementwise share
  shrinks as batch grows).

Greedy decode is used (deterministic per engine) and timing is min-of-N to
resist contention.  But greedy across *different* engines diverges into
different sequence lengths, so RTF (= audio / wall) is length-confounded — it
rewards whichever engine happened to decode longer.  The FAIR speed metric is
``ms_per_step = wall / max(codec_steps)`` (a batch runs ~max-steps lock-step
iterations); that normalises out the length divergence.

Run: make bench-batched-matrix  (or uv run python benchmark/bench_batched_matrix.py)
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
log = logging.getLogger("bench_batched_matrix")
OUT = Path(__file__).parent / "results" / "batched_matrix.json"

SPEAKER = "vivian"
LANG = "en"
BATCH_SIZE = 16
MAX_NEW = 256
MEASURE = 2  # min-of-N measured runs (after one warmup)
RUNNERS = ["base", "triton", "faster", "hybrid"]
_ENGINE = {
    "base": "HF-eager",
    "triton": "HF-eager",
    "faster": "CUDA-graph",
    "hybrid": "CUDA-graph",
}

# 16 varied-length clips (short → long) to exercise bucketing + padding.
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


def _measure_runner(name: str, texts: list[str]) -> dict[str, Any]:
    """Load one runner, warm it, then min-of-N measure a batched generation.

    ``generate_batch`` is uniform across runners: ``batch_size`` and the sampling
    config are per-call arguments.  The CUDA-graph runners (faster/hybrid) bake
    them into the graph captured on the first call; the HF runners (base/triton)
    apply them per call.
    """
    gen_kwargs: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "language": LANG,
        "speaker": SPEAKER,
        "max_new_tokens": MAX_NEW,
        "greedy": True,
    }

    runner = create_runner(name)
    runner.load_model()
    try:
        # Warmup (captures CUDA graphs for the graph runners, lazy init for HF).
        runner.generate_batch(texts, **gen_kwargs)
        best: dict[str, Any] | None = None
        for _ in range(MEASURE):
            out = runner.generate_batch(texts, **gen_kwargs)
            if best is None or out["wall_s"] < best["wall_s"]:
                best = out
        assert best is not None
        steps = [r["codec_steps"] for r in best["results"]]
        # A batch decodes all rows in lock-step until the longest row finishes,
        # so it runs ~max(codec_steps) iterations.  wall / max_steps is the true
        # per-step serving cost — the FAIR cross-engine speed metric, since
        # greedy across different engines diverges into different lengths (RTF,
        # being audio/wall, rewards whichever engine happened to decode longer).
        ms_per_step = best["wall_s"] / max(max(steps), 1) * 1000.0
        row = {
            "runner": name,
            "engine": _ENGINE[name],
            "num_samples": best["num_samples"],
            "num_buckets": best["num_buckets"],
            "wall_s": round(best["wall_s"], 3),
            "per_sample_s": round(best["wall_s"] / max(best["num_samples"], 1), 4),
            "ms_per_step": round(ms_per_step, 1),
            "rtf": round(best["rtf"], 2),
            "total_audio_s": round(best["total_audio_s"], 1),
            "codec_steps_min": min(steps),
            "codec_steps_max": max(steps),
            "peak_vram_gb": round(best["peak_vram_gb"], 2),
        }
        log.info(
            "%-16s [%-10s] wall=%.2fs ms/step=%.1f RTF=%.1f vram=%.1fGB steps[%d..%d]",
            name,
            row["engine"],
            row["wall_s"],
            row["ms_per_step"],
            row["rtf"],
            row["peak_vram_gb"],
            row["codec_steps_min"],
            row["codec_steps_max"],
        )
        return row
    finally:
        runner.unload_model()
        torch.cuda.empty_cache()


def main() -> None:
    torch.set_grad_enabled(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    texts = POOL
    results: dict[str, Any] = {
        "config": {
            "batch_size": BATCH_SIZE,
            "num_texts": len(texts),
            "max_new_tokens": MAX_NEW,
            "decode": "greedy",
            "measure_min_of": MEASURE,
            "speaker": SPEAKER,
            "language": LANG,
            "note": (
                "HF-eager (base/triton) vs CUDA-graph (faster/hybrid) are "
                "different decode engines; cross-family = serving leaderboard, "
                "same-family delta = Triton kernels."
            ),
        },
        "rows": [],
    }
    for name in RUNNERS:
        try:
            results["rows"].append(_measure_runner(name, texts))
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM for %s at batch_size=%d", name, BATCH_SIZE)
            results["rows"].append({"runner": name, "oom": True})
            torch.cuda.empty_cache()
        OUT.write_text(json.dumps(results, indent=2))  # incremental save

    log.info("WROTE %s", OUT)
    print(
        "\n=== 4-way batched comparison "
        "(ms/step = fair speed, lower=better; RTF length-confounded) ===",
        flush=True,
    )
    print(
        f"{'runner':<16} {'engine':<11} {'ms/step':>8} {'wall_s':>7} "
        f"{'RTF':>6} {'vram_gb':>8}",
        flush=True,
    )
    for row in results["rows"]:
        if row.get("oom"):
            print(f"{row['runner']:<16} {'OOM':<11}", flush=True)
            continue
        print(
            f"{row['runner']:<16} {row['engine']:<11} {row['ms_per_step']:>8} "
            f"{row['wall_s']:>7} {row['rtf']:>6} {row['peak_vram_gb']:>8}",
            flush=True,
        )


if __name__ == "__main__":
    main()
