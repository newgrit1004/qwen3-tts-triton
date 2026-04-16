"""Fixed-length E2E benchmarks for fair per-token speed comparison.

Controls for generation length variation (the confounding variable in
bench_e2e.py/bench_e2e_long.py) by fixing max_new_tokens across all runners.
This isolates the actual per-token latency difference introduced by TurboQuant.

Usage:
    python -m benchmark.bench_e2e_fixed --warmup 2 --repeat 5
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

SAMPLE_TEXTS = [
    {"text": "안녕하세요, 오늘 날씨가 정말 좋네요.", "language": "ko"},
    {
        "text": "Hello, welcome to the Qwen3 text-to-speech system.",
        "language": "en",
    },
]

TOKEN_COUNTS = [500, 1000, 2000]


def _reset_gpu() -> None:
    """Empty CUDA cache and reset peak VRAM statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _get_runners() -> dict[str, Any]:
    """Import TQ-relevant runners."""
    runners: dict[str, Any] = {}

    try:
        from qwen3_tts_triton.models.triton_runner import TritonRunner

        runners["Triton"] = TritonRunner
        runners["Triton+TQ"] = lambda **kw: TritonRunner(enable_turboquant=True, **kw)
    except ImportError:
        logger.warning("TritonRunner not available")

    try:
        from qwen3_tts_triton.models.faster_runner import FasterRunner

        runners["Faster"] = FasterRunner
    except ImportError:
        logger.warning("FasterRunner not available")

    try:
        from qwen3_tts_triton.models.triton_faster_runner import TritonFasterRunner

        runners["Hybrid"] = TritonFasterRunner
        runners["Hybrid+TQ"] = lambda **kw: TritonFasterRunner(
            enable_turboquant=True, **kw
        )
    except ImportError:
        logger.warning("TritonFasterRunner not available")

    return runners


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistical summary."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _measure_runs(
    runner: Any,
    sample: dict[str, str],
    runner_name: str,
    max_new_tokens: int,
    repeat: int,
) -> dict[str, Any]:
    """Run measured iterations with fixed max_new_tokens."""
    timings_ms: list[float] = []
    audio_durations: list[float] = []
    rtf_values: list[float] = []

    baseline_vram = torch.cuda.memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()

    for run_idx in range(repeat):
        torch.cuda.empty_cache()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = runner.generate(
            text=sample["text"],
            language=sample["language"],
            max_new_tokens=max_new_tokens,
        )
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        timings_ms.append(elapsed_ms)

        audio = output.get("audio")
        sr = output.get("sample_rate", 24000)
        audio_len = len(audio) if audio is not None else 0
        audio_dur = audio_len / sr if sr > 0 else 0.0
        audio_durations.append(audio_dur)

        rtf = audio_dur / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
        rtf_values.append(rtf)

        logger.debug(
            "[%s] tok=%d run %d/%d: %.1f ms, audio %.1fs (RTF %.2f)",
            runner_name,
            max_new_tokens,
            run_idx + 1,
            repeat,
            elapsed_ms,
            audio_dur,
            rtf,
        )

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "timings_ms": timings_ms,
        "audio_durations": audio_durations,
        "rtf_values": rtf_values,
        "peak_vram_gb": round(peak_vram, 3),
        "baseline_vram_gb": round(baseline_vram, 3),
    }


def bench_runner_fixed(
    runner_cls: type,
    runner_name: str,
    texts: list[dict[str, str]],
    token_counts: list[int],
    warmup: int = 2,
    repeat: int = 5,
) -> list[dict[str, Any]]:
    """Benchmark a single runner across fixed token counts."""
    results = []
    runner = runner_cls()

    try:
        _reset_gpu()
        t_load = time.perf_counter()
        runner.load_model()
        torch.cuda.synchronize()
        load_time = round(time.perf_counter() - t_load, 3)
        logger.info("[%s] Model loaded in %.3fs", runner_name, load_time)

        for sample in texts:
            for max_tokens in token_counts:
                logger.info(
                    "[%s] %s, max_new_tokens=%d",
                    runner_name,
                    sample["language"],
                    max_tokens,
                )

                # Warmup
                for i in range(warmup):
                    logger.debug("[%s] Warmup %d/%d", runner_name, i + 1, warmup)
                    runner.generate(
                        text=sample["text"],
                        language=sample["language"],
                        max_new_tokens=max_tokens,
                    )
                torch.cuda.synchronize()

                # Measured runs
                data = _measure_runs(runner, sample, runner_name, max_tokens, repeat)

                entry: dict[str, Any] = {
                    "runner": runner_name,
                    "language": sample["language"],
                    "max_new_tokens": max_tokens,
                    "warmup": warmup,
                    "repeat": repeat,
                    "time_ms": _compute_stats(data["timings_ms"]),
                    "audio_duration_s": _compute_stats(data["audio_durations"]),
                    "rtf": _compute_stats(data["rtf_values"]),
                    "peak_vram_gb": data["peak_vram_gb"],
                    "baseline_vram_gb": data["baseline_vram_gb"],
                    "model_load_time_s": load_time,
                }
                results.append(entry)
    finally:
        runner.unload_model()
        _reset_gpu()

    return results


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format results as readable ASCII table."""
    header = (
        f"{'Runner':<15} {'Lang':<5} {'Tokens':>6} "
        f"{'Mean(ms)':>9} {'Std':>7} {'P50':>8} {'P95':>8} "
        f"{'RTF':>6} {'Audio(s)':>8} {'VRAM':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        t = r["time_ms"]
        lines.append(
            f"{r['runner']:<15} {r['language']:<5} {r['max_new_tokens']:>6} "
            f"{t['mean']:>9.1f} {t['std']:>7.1f} "
            f"{t['p50']:>8.1f} {t['p95']:>8.1f} "
            f"{r['rtf']['mean']:>6.2f} "
            f"{r['audio_duration_s']['mean']:>8.1f} "
            f"{r['peak_vram_gb']:>6.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_comparison(results: list[dict[str, Any]]) -> str:
    """Format TQ speedup comparison table grouped by token count."""
    # Group by (language, max_new_tokens)
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for r in results:
        key = (r["language"], r["max_new_tokens"])
        if key not in by_key:
            by_key[key] = {}
        by_key[key][r["runner"]] = r

    lines = [
        "",
        "=== TQ Speedup (P50 basis, fixed max_new_tokens) ===",
        f"{'Lang':<5} {'Tokens':>6} {'Baseline':<12} {'P50(ms)':>8} "
        f"{'TQ variant':<12} {'P50(ms)':>8} {'Speedup':>8}",
        "-" * 75,
    ]

    pairs = [("Triton", "Triton+TQ"), ("Hybrid", "Hybrid+TQ")]
    for (lang, tokens), runners in sorted(by_key.items()):
        for base_name, tq_name in pairs:
            if base_name in runners and tq_name in runners:
                base_p50 = runners[base_name]["time_ms"]["p50"]
                tq_p50 = runners[tq_name]["time_ms"]["p50"]
                speedup = base_p50 / tq_p50 if tq_p50 > 0 else 0
                lines.append(
                    f"{lang:<5} {tokens:>6} {base_name:<12} {base_p50:>8.1f} "
                    f"{tq_name:<12} {tq_p50:>8.1f} {speedup:>7.2f}x"
                )

    lines.append("-" * 75)
    return "\n".join(lines)


def run_fixed_benchmarks(
    texts: list[dict[str, str]] | None = None,
    token_counts: list[int] | None = None,
    warmup: int = 2,
    repeat: int = 5,
    output: str | None = None,
) -> list[dict[str, Any]]:
    """Run fixed-length E2E benchmarks.

    Args:
        texts: Text samples to use.
        token_counts: List of max_new_tokens values.
        warmup: Number of warmup runs.
        repeat: Number of measured runs.
        output: Output JSON path.

    Returns:
        List of result dicts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = texts or SAMPLE_TEXTS
    token_counts = token_counts or TOKEN_COUNTS

    runner_classes = _get_runners()
    if not runner_classes:
        logger.error("No runners available.")
        return []

    all_results: list[dict[str, Any]] = []

    for name, cls in runner_classes.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("Benchmarking %s runner...", name)
        try:
            results = bench_runner_fixed(cls, name, texts, token_counts, warmup, repeat)
            all_results.extend(results)
        except Exception:
            logger.exception("Failed to benchmark %s", name)

    # Log tables
    if all_results:
        table = _format_table(all_results)
        for line in table.split("\n"):
            logger.info(line)

        comparison = _format_comparison(all_results)
        for line in comparison.split("\n"):
            logger.info(line)

    # Save JSON
    out_path = output or str(RESULTS_DIR / "e2e_fixed_benchmarks.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fixed-length E2E benchmark for per-token speed comparison"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of measured runs per config (default: 5)",
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
    run_fixed_benchmarks(
        warmup=args.warmup,
        repeat=args.repeat,
        output=args.output,
    )
