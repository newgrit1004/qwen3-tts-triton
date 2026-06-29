"""Verify the README headline: base single-clip (B=1) vs hybrid batched (B=16).

The README quotes two numbers that COMPOUND:

* ``~5x`` single-clip latency  — hybrid vs base, both at batch=1 (``make bench-e2e``).
* ``~14x`` per-sample throughput — base single-clip (batch=1, sequential PyTorch
  eager) vs hybrid batched (batch=16), measured here.

This script runs those two end points on the SAME varied-length pool and prints
``base_b1_per_sample / hybrid_b16_per_sample`` so the ~14x claim is directly
reproducible.  ``per_sample = wall / num_samples``: for base batch=1 that is the
mean single-clip latency; for hybrid batch=16 it is the amortised per-request
cost under batched serving (the throughput win, not a latency win).

Run: make bench-speedup  (or uv run python benchmark/bench_speedup.py)
Takes ~3-4 min on an RTX 5090 (the base B=1 baseline is deliberately the slow path).
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
log = logging.getLogger("bench_speedup")
OUT = Path(__file__).parent / "results" / "speedup.json"

SPEAKER = "vivian"
LANG = "en"
MAX_NEW = 256

# Same varied-length pool as bench_batched_matrix so the per-sample numbers are
# directly comparable to the committed batched matrix and the README figures.
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


def _measure(
    name: str,
    batch_size: int,
    warmup_texts: list[str],
    measure_texts: list[str],
    measure: int,
) -> dict[str, Any]:
    """Load ``name``, warm it, then min-of-``measure`` measure a batched run.

    ``warmup_texts`` is run once to absorb lazy init / CUDA-graph capture; the
    timed ``measure_texts`` run(s) are what we report.  ``per_sample`` is
    ``wall / num_samples`` of the best (fastest) measured run.
    """
    gen_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "language": LANG,
        "speaker": SPEAKER,
        "max_new_tokens": MAX_NEW,
        "greedy": True,
    }
    runner = create_runner(name)
    runner.load_model()
    try:
        runner.generate_batch(warmup_texts, **gen_kwargs)  # warmup (discarded)
        best: dict[str, Any] | None = None
        for _ in range(measure):
            out = runner.generate_batch(measure_texts, **gen_kwargs)
            if best is None or out["wall_s"] < best["wall_s"]:
                best = out
        assert best is not None
        n = max(best["num_samples"], 1)
        return {
            "runner": name,
            "batch_size": batch_size,
            "wall_s": round(best["wall_s"], 3),
            "num_samples": best["num_samples"],
            "per_sample_s": round(best["wall_s"] / n, 4),
            "peak_vram_gb": round(best["peak_vram_gb"], 2),
        }
    finally:
        runner.unload_model()
        torch.cuda.empty_cache()


def main() -> None:
    torch.set_grad_enabled(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Baseline: base, batch=1 = sequential single-clip PyTorch eager.  Warm with
    # one short clip (cheap lazy init), then time the full varied pool once;
    # per_sample is then the mean single-clip latency over the pool.
    base = _measure("base", 1, POOL[:1], POOL, measure=1)
    log.info(
        "base   b=1  per_sample=%.3fs vram=%.1fGB",
        base["per_sample_s"],
        base["peak_vram_gb"],
    )

    # Best: hybrid batched at B=16 (the README operating point).  Warm at B=16 to
    # capture the right CUDA graph, then min-of-2.
    hybrid = _measure("hybrid", 16, POOL, POOL, measure=2)
    speedup = base["per_sample_s"] / max(hybrid["per_sample_s"], 1e-9)
    hybrid["speedup_vs_base_b1"] = round(speedup, 1)
    log.info(
        "hybrid b=16 per_sample=%.3fs vram=%.1fGB  speedup=%.1fx",
        hybrid["per_sample_s"],
        hybrid["peak_vram_gb"],
        hybrid["speedup_vs_base_b1"],
    )

    OUT.write_text(json.dumps({"base_b1": base, "hybrid_b16": hybrid}, indent=2))
    print(
        "\n=== per-sample throughput: base(B=1 eager) -> hybrid(batched B=16) ===",
        flush=True,
    )
    print(f"base   B=1 : {base['per_sample_s']:.3f} s/sample", flush=True)
    print(f"hybrid B=16: {hybrid['per_sample_s']:.3f} s/sample", flush=True)
    print(
        f"\nSPEEDUP: {hybrid['speedup_vs_base_b1']:.1f}x  (README claims ~14x)",
        flush=True,
    )
    log.info("WROTE %s", OUT)


if __name__ == "__main__":
    main()
