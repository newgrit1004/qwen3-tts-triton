"""Phase 5: VRAM utilization experiments with TurboQuant.

Tests how freed VRAM from KV cache quantization can be used for:
  A) Larger batch sizes → higher throughput
  B) Longer context lengths → longer audio generation

Usage:
    python -m benchmark.bench_throughput_scaling [--output PATH]
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

SAMPLE_TEXT = "Hello, welcome to the Qwen3 text-to-speech system."
SAMPLE_LANG = "en"


def _reset_gpu() -> None:
    """Empty CUDA cache and reset peak VRAM stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb() -> float:
    """Return peak GPU VRAM in GB."""
    return torch.cuda.max_memory_allocated() / (1024**3)


def _current_vram_gb() -> float:
    """Return current GPU VRAM in GB."""
    return torch.cuda.memory_allocated() / (1024**3)


def _bench_token_length(
    runner: Any,
    mode: str,
    max_tokens: int,
    repeat: int,
    baseline_vram: float,
) -> dict[str, Any]:
    """Benchmark a single max_new_tokens setting, return result dict."""
    logger.info("[%s] Testing max_new_tokens=%d...", mode, max_tokens)
    timings: list[float] = []
    oom = False
    peak_vram = 0.0

    for run_idx in range(repeat):
        try:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            runner.generate(
                text=SAMPLE_TEXT, language=SAMPLE_LANG, max_new_tokens=max_tokens
            )
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            peak_vram = max(peak_vram, _peak_vram_gb())
            logger.debug(
                "  Run %d/%d: %.2fs, peak %.2f GB",
                run_idx + 1,
                repeat,
                elapsed,
                peak_vram,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "[%s] OOM at max_new_tokens=%d, run %d",
                mode,
                max_tokens,
                run_idx + 1,
            )
            oom = True
            _reset_gpu()
            break

    entry: dict[str, Any] = {
        "mode": mode,
        "max_new_tokens": max_tokens,
        "oom": oom,
        "peak_vram_gb": round(peak_vram, 3),
        "baseline_vram_gb": round(baseline_vram, 3),
    }
    if timings:
        arr = np.array(timings)
        entry["time_mean_s"] = round(float(np.mean(arr)), 3)
        entry["time_std_s"] = round(float(np.std(arr)), 3)
        entry["successful_runs"] = len(timings)
    return entry


def bench_context_scaling(
    modes: list[str] | None = None,
    max_tokens_list: list[int] | None = None,
    warmup: int = 1,
    repeat: int = 3,
) -> list[dict[str, Any]]:
    """Test context length scaling: how long can we generate before OOM?

    Args:
        modes: Runner modes to test.
        max_tokens_list: List of max_new_tokens values.
        warmup: Number of warmup runs.
        repeat: Number of measured runs.

    Returns:
        List of result dicts.
    """
    from qwen3_tts_triton.models import create_runner

    modes = modes or ["triton", "triton+tq"]
    max_tokens_list = max_tokens_list or [500, 1000, 2000, 4000]
    results: list[dict[str, Any]] = []

    for mode in modes:
        logger.info("=== Context scaling: %s ===", mode)
        runner = create_runner(mode)
        try:
            _reset_gpu()
            runner.load_model()
            torch.cuda.synchronize()

            baseline_vram = _current_vram_gb()
            logger.info("[%s] Baseline VRAM: %.2f GB", mode, baseline_vram)

            for _ in range(warmup):
                runner.generate(
                    text=SAMPLE_TEXT, language=SAMPLE_LANG, max_new_tokens=200
                )

            for max_tokens in max_tokens_list:
                entry = _bench_token_length(
                    runner, mode, max_tokens, repeat, baseline_vram
                )
                results.append(entry)
                if entry["oom"]:
                    break

        finally:
            runner.unload_model()
            _reset_gpu()

    return results


def _format_context_table(results: list[dict[str, Any]]) -> str:
    """Format context scaling results as ASCII table."""
    header = (
        f"{'Mode':<15} {'MaxTokens':>10} {'OOM':>4} "
        f"{'Mean(s)':>8} {'Std(s)':>7} {'PeakVRAM':>9}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        mean = f"{r.get('time_mean_s', 0):.2f}" if not r["oom"] else "N/A"
        std = f"{r.get('time_std_s', 0):.2f}" if not r["oom"] else "N/A"
        lines.append(
            f"{r['mode']:<15} {r['max_new_tokens']:>10} "
            f"{'YES' if r['oom'] else 'no':>4} "
            f"{mean:>8} {std:>7} {r['peak_vram_gb']:>8.2f}G"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_throughput_scaling(
    output: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run all throughput scaling experiments.

    Args:
        output: Output JSON path.

    Returns:
        Dict with 'context_scaling' results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}

    # Context length scaling
    logger.info("=" * 60)
    logger.info("Context Length Scaling Experiment")
    logger.info("=" * 60)
    context_results = bench_context_scaling()
    all_results["context_scaling"] = context_results

    table = _format_context_table(context_results)
    for line in table.split("\n"):
        logger.info(line)

    # Save results
    out_path = output or str(RESULTS_DIR / "throughput_scaling.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TurboQuant VRAM utilization experiments"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    args = _parse_args()
    run_throughput_scaling(output=args.output)
