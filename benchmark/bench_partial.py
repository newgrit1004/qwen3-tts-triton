"""Partial patching benchmark: Base vs Hybrid vs Hybrid+Patch(N).

Compares configurations on latency, RTF, VRAM, and speedup.
Supports multiple patch ranges in a single sweep and optional audio saving.

Usage:
    # Single range
    python -m benchmark.bench_partial --patch-range 0,20

    # Multi-range sweep with audio saving
    python -m benchmark.bench_partial \
        --patch-range 0,16 --patch-range 0,20 \
        --patch-range 0,22 --patch-range 0,24 \
        --save-audio --warmup 3 --repeat 5
"""

import argparse
import gc
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
AUDIO_DIR = Path(__file__).parent / "output" / "partial"

SAMPLE_TEXTS = [
    {"text": "안녕하세요, 오늘 날씨가 정말 좋네요.", "language": "ko"},
    {
        "text": "Hello, welcome to the Qwen3 text-to-speech system.",
        "language": "en",
    },
]


# ────────────────────────────────────────────────────────────
# Deterministic environment
# ────────────────────────────────────────────────────────────


def _setup_deterministic_env(seed: int = 42) -> None:
    """Set up a reproducible benchmark environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ────────────────────────────────────────────────────────────
# GPU helpers
# ────────────────────────────────────────────────────────────


def _reset_gpu() -> None:
    """Empty CUDA cache and reset peak VRAM statistics."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _gpu_name() -> str:
    """Return GPU device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


# ────────────────────────────────────────────────────────────
# Measurement helpers
# ────────────────────────────────────────────────────────────


def _calculate_rtf(
    audio_samples: int, sample_rate: int, generation_time: float
) -> float:
    """Calculate Real-Time Factor: audio_duration / generation_time."""
    if generation_time <= 0:
        return 0.0
    audio_duration = audio_samples / sample_rate
    return audio_duration / generation_time


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistical summary for a list of values."""
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


# ────────────────────────────────────────────────────────────
# Runner benchmark (with optional audio saving)
# ────────────────────────────────────────────────────────────


def _measure_sample(
    runner: Any,
    sample: dict[str, str],
    repeat: int,
    save_audio: bool,
    runner_name: str,
    sample_idx: int,
) -> tuple[list[float], list[float], str | None]:
    """Run measured iterations for a single sample, return timings and RTF."""
    timings_ms: list[float] = []
    rtf_values: list[float] = []
    saved_audio_path: str | None = None

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
        rtf_values.append(_calculate_rtf(audio_len, sr, elapsed_ms / 1000.0))

        if save_audio and run_idx == 0 and audio is not None:
            saved_audio_path = _save_audio(
                audio, sr, runner_name, sample["language"], sample_idx
            )

    return timings_ms, rtf_values, saved_audio_path


def _bench_runner(
    runner_factory: Any,
    runner_name: str,
    texts: list[dict[str, str]],
    warmup: int,
    repeat: int,
    save_audio: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark a single runner configuration.

    Args:
        runner_factory: Callable that returns a runner instance.
        runner_name: Display name for the runner.
        texts: List of text/language dicts.
        warmup: Number of warmup runs.
        repeat: Number of measured runs per text.
        save_audio: If True, save the first measured run's audio to disk.

    Returns:
        List of result dicts with timing statistics.
    """
    results: list[dict[str, Any]] = []
    runner = runner_factory() if callable(runner_factory) else runner_factory

    try:
        _reset_gpu()
        t_load_start = time.perf_counter()
        runner.load_model()
        torch.cuda.synchronize()
        model_load_time_s = round(time.perf_counter() - t_load_start, 3)
        logger.info("[%s] Model loaded in %.3fs", runner_name, model_load_time_s)

        for sample_idx, sample in enumerate(texts):
            for i in range(warmup):
                logger.debug("[%s] Warmup %d/%d", runner_name, i + 1, warmup)
                runner.generate(text=sample["text"], language=sample["language"])
            torch.cuda.synchronize()

            baseline_vram = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()

            timings_ms, rtf_values, saved_audio_path = _measure_sample(
                runner, sample, repeat, save_audio, runner_name, sample_idx
            )

            peak_delta = torch.cuda.max_memory_allocated() / (1024**3) - baseline_vram
            entry: dict[str, Any] = {
                "runner": runner_name,
                "text": sample["text"],
                "language": sample["language"],
                "sample_idx": sample_idx,
                "warmup": warmup,
                "repeat": repeat,
                "time_ms": _compute_stats(timings_ms),
                "rtf": _compute_stats(rtf_values),
                "peak_vram_gb": round(baseline_vram + peak_delta, 3),
                "inference_delta_gb": round(peak_delta, 3),
                "baseline_vram_gb": round(baseline_vram, 3),
                "model_load_time_s": model_load_time_s,
            }
            if saved_audio_path:
                entry["audio_path"] = saved_audio_path
            results.append(entry)
    finally:
        runner.unload_model()
        _reset_gpu()

    return results


def _save_audio(
    audio: np.ndarray,
    sample_rate: int,
    runner_name: str,
    language: str,
    sample_idx: int,
) -> str:
    """Save audio to WAV file under AUDIO_DIR.

    Returns:
        Relative path from project root.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitize runner name for filename
    safe_name = runner_name.lower().replace("+", "_").replace(" ", "_")
    filename = f"{safe_name}_{language}_{sample_idx}.wav"
    filepath = AUDIO_DIR / filename
    sf.write(str(filepath), audio, sample_rate)
    logger.info("[%s] Audio saved: %s", runner_name, filepath)
    return str(filepath.relative_to(Path(__file__).parent.parent))


# ────────────────────────────────────────────────────────────
# Output formatting
# ────────────────────────────────────────────────────────────


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format results as a readable ASCII table."""
    header = (
        f"{'Runner':<20} {'Lang':<5} {'Mean(ms)':>9} {'Std':>7} "
        f"{'P50':>8} {'P95':>8} {'RTF':>6} {'VRAM':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        t = r["time_ms"]
        lines.append(
            f"{r['runner']:<20} {r['language']:<5} "
            f"{t['mean']:>9.1f} {t['std']:>7.1f} "
            f"{t['p50']:>8.1f} {t['p95']:>8.1f} "
            f"{r['rtf']['mean']:>6.2f} {r['peak_vram_gb']:>6.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_speedup(results: list[dict[str, Any]]) -> str:
    """Format speedup comparison vs Base."""
    base_means: dict[str, float] = {}
    for r in results:
        if r["runner"] == "Base":
            base_means[r["language"]] = r["time_ms"]["mean"]

    lines = ["\nSpeedup vs Base:"]
    for r in results:
        if r["runner"] == "Base":
            continue
        lang = r["language"]
        base_mean = base_means.get(lang)
        if base_mean and base_mean > 0:
            speedup = base_mean / r["time_ms"]["mean"]
            lines.append(f"  {r['runner']:<20} {lang}: {speedup:>6.2f}x")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# Main benchmark
# ────────────────────────────────────────────────────────────


def _build_runners(
    patch_ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Build runner factories for Base, Hybrid, and each Hybrid+Patch config."""
    from qwen3_tts_triton.models.base_runner import BaseRunner
    from qwen3_tts_triton.models.triton_faster_runner import TritonFasterRunner

    runners: dict[str, Any] = {
        "Base": BaseRunner,
        "Hybrid": TritonFasterRunner,
    }
    for pr in patch_ranges:
        name = f"Hybrid+P({pr[0]},{pr[1]})"
        # Use default arg to capture pr in closure
        runners[name] = lambda _pr=pr: TritonFasterRunner(patch_range=_pr)
    return runners


def run_partial_benchmark(
    patch_ranges: list[tuple[int, int]],
    texts: list[dict[str, str]] | None = None,
    warmup: int = 3,
    repeat: int = 10,
    save_audio: bool = False,
    output: str | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run Base vs Hybrid vs Hybrid+Patch(N) benchmark sweep.

    Args:
        patch_ranges: List of (start, end) ranges to test.
        texts: Test sentences. Defaults to SAMPLE_TEXTS.
        warmup: Number of warmup runs.
        repeat: Number of measured runs per text.
        save_audio: Save first run's audio to disk.
        output: JSON output path.
        seed: Random seed.

    Returns:
        List of result dicts.
    """
    _setup_deterministic_env(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = texts or SAMPLE_TEXTS

    runner_factories = _build_runners(patch_ranges)
    all_results: list[dict[str, Any]] = []

    for name, factory in runner_factories.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("=" * 60)
        logger.info("Benchmarking %s ...", name)
        logger.info("=" * 60)
        try:
            results = _bench_runner(
                factory, name, texts, warmup, repeat, save_audio=save_audio
            )
            all_results.extend(results)
        except Exception:
            logger.exception("Failed to benchmark %s", name)

    # Log table
    if all_results:
        table = _format_table(all_results)
        for line in table.split("\n"):
            logger.info(line)
        speedup = _format_speedup(all_results)
        for line in speedup.split("\n"):
            logger.info(line)

    # Build output with metadata
    output_data: dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "gpu": _gpu_name(),
            "patch_ranges": [list(pr) for pr in patch_ranges],
            "warmup": warmup,
            "repeat": repeat,
            "seed": seed,
            "save_audio": save_audio,
        },
        "results": all_results,
    }

    # Save JSON
    range_tag = "_".join(f"{s}-{e}" for s, e in patch_ranges)
    out_path = output or str(RESULTS_DIR / f"partial_sweep_{range_tag}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(output_data, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def _parse_patch_range(value: str) -> tuple[int, int]:
    """Parse 'start,end' string into a (start, end) tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'start,end' format, got '{value}'")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Non-integer values in '{value}'") from e
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            f"Must satisfy 0 <= start < end, got ({start}, {end})"
        )
    return (start, end)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Partial patching benchmark: Base vs Hybrid vs Hybrid+Patch"
    )
    parser.add_argument(
        "--patch-range",
        type=_parse_patch_range,
        action="append",
        required=True,
        dest="patch_ranges",
        help="Layer range (repeatable), e.g. --patch-range 0,20 --patch-range 0,24.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of measured runs per text (default: 10).",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        default=False,
        help="Save audio samples for each config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    args = _parse_args()
    run_partial_benchmark(
        patch_ranges=args.patch_ranges,
        warmup=args.warmup,
        repeat=args.repeat,
        save_audio=args.save_audio,
        output=args.output,
        seed=args.seed,
    )
