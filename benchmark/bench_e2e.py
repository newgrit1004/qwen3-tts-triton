"""End-to-end benchmarks for Qwen3-TTS inference modes.

Compares all 4 runners (Base, Triton, Faster, Hybrid) on RTF, total time,
and peak VRAM with proper CUDA event timing, warmup, and statistical
reporting (mean, std, p50, p95, p99).

Usage:
    python -m benchmark.bench_e2e --warmup 3 --repeat 20 --output results.json
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


def _reset_gpu() -> None:
    """Empty CUDA cache and reset peak VRAM statistics."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb() -> float:
    """Return peak GPU VRAM allocated since last reset, in gigabytes.

    Returns:
        Peak VRAM usage in GB.
    """
    return torch.cuda.max_memory_allocated() / (1024**3)


def _check_cuda_graph_status(runner: Any, runner_name: str) -> None:
    """Warn if Faster/Hybrid runner fell back to eager mode (no CUDA Graph)."""
    if runner_name not in ("Faster", "Hybrid"):
        return
    model = getattr(runner, "model", None)
    if model is None:
        return
    # Check common attributes for CUDA Graph capture status
    for attr in ("_cuda_graphs", "cuda_graphs", "_graph", "_cuda_graph_captured"):
        val = getattr(model, attr, None)
        if val is not None:
            if isinstance(val, bool) and not val:
                logger.warning(
                    "[%s] CUDA Graph NOT captured - running in eager mode. "
                    "Results will NOT reflect CUDA Graph speedup. "
                    "Check GPU VRAM availability.",
                    runner_name,
                )
            return
    # If no attribute found, check VRAM heuristic: after warmup, Faster/Hybrid
    # should use ~4.2+ GB if CUDA Graph is captured
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    if peak < 4.15:
        logger.warning(
            "[%s] Peak VRAM %.2f GB is lower than expected (~4.2+ GB). "
            "CUDA Graph may not have been captured.",
            runner_name,
            peak,
        )


def _get_runners() -> dict[str, Any]:
    """Import available runners with graceful fallback."""
    runners: dict[str, Any] = {}

    try:
        from qwen3_tts_triton.models.base_runner import BaseRunner

        runners["Base"] = BaseRunner
    except ImportError:
        logger.warning("BaseRunner not available")

    try:
        from qwen3_tts_triton.models.triton_runner import TritonRunner

        runners["Triton"] = TritonRunner
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
    except ImportError:
        logger.warning("TritonFasterRunner not available")

    return runners


def _calculate_rtf(
    audio_samples: int, sample_rate: int, generation_time: float
) -> float:
    """Calculate Real-Time Factor: audio_duration / generation_time."""
    if generation_time <= 0:
        return 0.0
    audio_duration = audio_samples / sample_rate
    return audio_duration / generation_time


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistical summary for a list of timing values."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _measure_runs(
    runner: Any,
    sample: dict[str, str],
    runner_name: str,
    repeat: int,
) -> tuple[list[float], list[float], float, float]:
    """Run measured iterations with CUDA event timing.

    Returns:
        Tuple of (timings_ms, rtf_values, peak_delta, baseline_vram).
    """
    timings_ms: list[float] = []
    rtf_values: list[float] = []

    baseline_vram = torch.cuda.memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()

    for run_idx in range(repeat):
        torch.cuda.empty_cache()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = runner.generate(text=sample["text"], language=sample["language"])
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        timings_ms.append(elapsed_ms)

        audio = output.get("audio")
        sr = output.get("sample_rate", 24000)
        audio_len = len(audio) if audio is not None else 0
        rtf = _calculate_rtf(audio_len, sr, elapsed_ms / 1000.0)
        rtf_values.append(rtf)

        logger.debug(
            "[%s] Run %d/%d: %.1f ms (RTF %.2f)",
            runner_name,
            run_idx + 1,
            repeat,
            elapsed_ms,
            rtf,
        )

    peak_delta = torch.cuda.max_memory_allocated() / (1024**3) - baseline_vram
    return timings_ms, rtf_values, peak_delta, baseline_vram


def bench_runner(
    runner_cls: type,
    runner_name: str,
    texts: list[dict[str, str]],
    warmup: int = 3,
    repeat: int = 20,
) -> list[dict[str, Any]]:
    """Benchmark a single runner with warmup and statistical measurement.

    Args:
        runner_cls: Runner class to benchmark.
        runner_name: Display name for the runner.
        texts: List of text/language dicts to generate.
        warmup: Number of warmup runs (discarded).
        repeat: Number of measured runs per text.

    Returns:
        List of result dicts with timing statistics.
    """
    results = []
    runner = runner_cls()

    try:
        _reset_gpu()
        t_load_start = time.perf_counter()
        runner.load_model()
        torch.cuda.synchronize()
        model_load_time_s = round(time.perf_counter() - t_load_start, 3)
        logger.info("[%s] Model loaded in %.3fs", runner_name, model_load_time_s)

        for sample in texts:
            # --- Warmup phase ---
            for i in range(warmup):
                logger.debug("[%s] Warmup %d/%d", runner_name, i + 1, warmup)
                runner.generate(text=sample["text"], language=sample["language"])
            torch.cuda.synchronize()
            _check_cuda_graph_status(runner, runner_name)

            timings_ms, rtf_values, peak_delta, baseline_vram = _measure_runs(
                runner, sample, runner_name, repeat
            )

            time_stats = _compute_stats(timings_ms)
            rtf_stats = _compute_stats(rtf_values)

            # Compile metrics (if available)
            compile_time = getattr(runner, "compile_time_s", None)
            first_run = getattr(runner, "first_run_s", None)

            entry: dict[str, Any] = {
                "runner": runner_name,
                "text": sample["text"][:40],
                "language": sample["language"],
                "warmup": warmup,
                "repeat": repeat,
                "time_ms": time_stats,
                "rtf": rtf_stats,
                "peak_vram_gb": round(baseline_vram + peak_delta, 3),
                "inference_delta_gb": round(peak_delta, 3),
                "baseline_vram_gb": round(baseline_vram, 3),
                "model_load_time_s": model_load_time_s,
            }
            if compile_time is not None:
                entry["compile_time_s"] = round(compile_time, 3)
            if first_run is not None:
                entry["first_run_s"] = round(first_run, 3)

            results.append(entry)
    finally:
        runner.unload_model()
        _reset_gpu()

    return results


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format results as a readable ASCII table with statistics."""
    header = (
        f"{'Runner':<15} {'Lang':<5} {'Mean(ms)':>9} {'Std':>7} "
        f"{'P50':>8} {'P95':>8} {'P99':>8} {'RTF':>6} {'VRAM':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        t = r["time_ms"]
        lines.append(
            f"{r['runner']:<15} {r['language']:<5} "
            f"{t['mean']:>9.1f} {t['std']:>7.1f} "
            f"{t['p50']:>8.1f} {t['p95']:>8.1f} {t['p99']:>8.1f} "
            f"{r['rtf']['mean']:>6.2f} {r['peak_vram_gb']:>6.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_e2e_benchmarks(
    texts: list[dict[str, str]] | None = None,
    warmup: int = 3,
    repeat: int = 20,
    output: str | None = None,
) -> list[dict[str, Any]]:
    """Run end-to-end benchmarks for all available runners.

    Discovers all importable runner classes, benchmarks each one with warmup
    and statistical measurement, logs an ASCII summary table, and saves
    results to JSON.

    Args:
        texts: List of text/language dicts to synthesize. Defaults to
            SAMPLE_TEXTS if None.
        warmup: Number of warmup runs discarded before measurement.
        repeat: Number of measured runs per text sample.
        output: Path for the output JSON file. Defaults to
            benchmark/results/e2e_benchmarks.json.

    Returns:
        List of result dicts, one entry per (runner, text) combination.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = texts or SAMPLE_TEXTS

    runner_classes = _get_runners()
    if not runner_classes:
        logger.error("No runners available. Install model runners first.")
        return []

    all_results: list[dict[str, Any]] = []

    for name, cls in runner_classes.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("Benchmarking %s runner...", name)
        try:
            results = bench_runner(cls, name, texts, warmup, repeat)
            all_results.extend(results)
        except Exception:
            logger.exception("Failed to benchmark %s", name)

    # Log table
    if all_results:
        table = _format_table(all_results)
        for line in table.split("\n"):
            logger.info(line)

    # Save JSON
    out_path = output or str(RESULTS_DIR / "e2e_benchmarks.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the E2E benchmark script.

    Returns:
        Parsed argument namespace with warmup, repeat, and output fields.
    """
    parser = argparse.ArgumentParser(
        description="E2E TTS benchmark with CUDA event timing"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of measured runs per text (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/e2e_benchmarks.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    args = _parse_args()
    run_e2e_benchmarks(
        warmup=args.warmup,
        repeat=args.repeat,
        output=args.output,
    )
