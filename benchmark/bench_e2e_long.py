"""End-to-end benchmarks for long-form TTS generation.

Tests TurboQuant effectiveness on longer sequences where KV cache
compression benefits are most pronounced. Uses paragraph-length texts
that generate 1000+ tokens, unlike bench_e2e.py's short sentences.

Usage:
    python -m benchmark.bench_e2e_long --warmup 2 --repeat 10 --output results.json
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

# Long-form texts: paragraphs that produce 1000+ talker tokens (~10-30s audio)
LONG_TEXTS = [
    {
        "text": (
            "오늘 서울의 날씨는 맑고 따뜻했습니다. "
            "아침부터 햇살이 창문으로 들어와 기분 좋은 하루를 시작할 수 있었죠. "
            "점심때는 동료들과 함께 근처 식당에서 비빔밥을 먹었는데, "
            "역시 한국 음식은 세계 최고라는 생각이 들었습니다. "
            "오후에는 회의가 두 개 있었지만, 생각보다 빨리 끝나서 "
            "여유 있게 코딩에 집중할 수 있었습니다."
        ),
        "language": "ko",
        "label": "ko_paragraph",
    },
    {
        "text": (
            "Artificial intelligence has transformed the way we interact "
            "with technology in our daily lives. From voice assistants that "
            "understand natural language to recommendation systems that "
            "predict our preferences, AI is everywhere. Text-to-speech "
            "technology, in particular, has made remarkable progress in "
            "recent years. Modern neural TTS systems can produce speech "
            "that is nearly indistinguishable from human recordings, "
            "with natural prosody, emotion, and rhythm."
        ),
        "language": "en",
        "label": "en_paragraph",
    },
    {
        "text": (
            "대한민국은 사계절이 뚜렷한 나라입니다. "
            "봄에는 벚꽃이 피고 산과 들이 초록빛으로 물듭니다. "
            "여름에는 장마와 무더위가 찾아오지만, "
            "시원한 계곡과 바다에서 여름을 즐길 수 있습니다. "
            "가을에는 단풍이 들어 전국의 산들이 붉게 타오르고, "
            "겨울에는 하얀 눈이 내려 아름다운 설경을 만들어 냅니다. "
            "이렇게 다양한 자연 풍경은 한국만의 특별한 매력입니다."
        ),
        "language": "ko",
        "label": "ko_long",
    },
    {
        "text": (
            "The development of open-source large language models has "
            "accelerated dramatically over the past two years. Projects "
            "like LLaMA, Mistral, and Qwen have demonstrated that "
            "competitive performance can be achieved without proprietary "
            "data or massive corporate budgets. This democratization of "
            "AI technology enables researchers and developers worldwide "
            "to build innovative applications, from automated code "
            "generation to multilingual speech synthesis systems that "
            "serve communities in every corner of the globe."
        ),
        "language": "en",
        "label": "en_long",
    },
]


def _reset_gpu() -> None:
    """Empty CUDA cache and reset peak VRAM statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb() -> float:
    """Return peak GPU VRAM allocated since last reset, in gigabytes."""
    return torch.cuda.max_memory_allocated() / (1024**3)


def _get_runners() -> dict[str, Any]:
    """Import available runners — focus on TQ-relevant pairs."""
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


def _calculate_rtf(
    audio_samples: int, sample_rate: int, generation_time: float
) -> float:
    """Calculate Real-Time Factor: audio_duration / generation_time."""
    if generation_time <= 0:
        return 0.0
    audio_duration = audio_samples / sample_rate
    return audio_duration / generation_time


def _measure_runs(
    runner: Any,
    sample: dict[str, str],
    runner_name: str,
    repeat: int,
) -> dict[str, Any]:
    """Run measured iterations and collect per-run statistics."""
    timings_ms: list[float] = []
    rtf_values: list[float] = []
    audio_durations: list[float] = []

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
        audio_dur = audio_len / sr if sr > 0 else 0.0
        audio_durations.append(audio_dur)

        rtf = _calculate_rtf(audio_len, sr, elapsed_ms / 1000.0)
        rtf_values.append(rtf)

        logger.debug(
            "[%s] Run %d/%d: %.1f ms, audio %.1fs (RTF %.2f)",
            runner_name,
            run_idx + 1,
            repeat,
            elapsed_ms,
            audio_dur,
            rtf,
        )

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "timings_ms": timings_ms,
        "rtf_values": rtf_values,
        "audio_durations": audio_durations,
        "peak_vram_gb": round(peak_vram, 3),
        "baseline_vram_gb": round(baseline_vram, 3),
    }


def bench_runner(
    runner_cls: type,
    runner_name: str,
    texts: list[dict[str, str]],
    warmup: int = 2,
    repeat: int = 10,
) -> list[dict[str, Any]]:
    """Benchmark a single runner on long texts."""
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
            label = sample.get("label", sample["text"][:30])
            logger.info("[%s] Testing: %s", runner_name, label)

            # Warmup
            for i in range(warmup):
                logger.debug("[%s] Warmup %d/%d", runner_name, i + 1, warmup)
                runner.generate(text=sample["text"], language=sample["language"])
            torch.cuda.synchronize()

            # Measured runs
            data = _measure_runs(runner, sample, runner_name, repeat)

            entry: dict[str, Any] = {
                "runner": runner_name,
                "label": label,
                "language": sample["language"],
                "text_chars": len(sample["text"]),
                "warmup": warmup,
                "repeat": repeat,
                "time_ms": _compute_stats(data["timings_ms"]),
                "rtf": _compute_stats(data["rtf_values"]),
                "audio_duration_s": _compute_stats(data["audio_durations"]),
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
    """Format results as a readable ASCII table."""
    header = (
        f"{'Runner':<15} {'Label':<15} "
        f"{'Mean(ms)':>9} {'Std':>7} {'P50':>8} {'P95':>8} "
        f"{'RTF':>6} {'Audio(s)':>8} {'VRAM':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        t = r["time_ms"]
        lines.append(
            f"{r['runner']:<15} {r['label']:<15} "
            f"{t['mean']:>9.1f} {t['std']:>7.1f} "
            f"{t['p50']:>8.1f} {t['p95']:>8.1f} "
            f"{r['rtf']['mean']:>6.2f} "
            f"{r['audio_duration_s']['mean']:>8.1f} "
            f"{r['peak_vram_gb']:>6.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_comparison(results: list[dict[str, Any]]) -> str:
    """Format TQ speedup comparison table."""
    # Group by label
    by_label: dict[str, dict[str, dict]] = {}
    for r in results:
        label = r["label"]
        runner = r["runner"]
        if label not in by_label:
            by_label[label] = {}
        by_label[label][runner] = r

    lines = [
        "",
        "=== TQ Speedup Comparison (P50 basis) ===",
        f"{'Label':<15} {'Baseline':<12} {'P50(ms)':>8} "
        f"{'TQ variant':<12} {'P50(ms)':>8} {'Speedup':>8}",
        "-" * 75,
    ]

    pairs = [("Triton", "Triton+TQ"), ("Hybrid", "Hybrid+TQ")]
    for label, runners in sorted(by_label.items()):
        for base_name, tq_name in pairs:
            if base_name in runners and tq_name in runners:
                base_p50 = runners[base_name]["time_ms"]["p50"]
                tq_p50 = runners[tq_name]["time_ms"]["p50"]
                speedup = base_p50 / tq_p50 if tq_p50 > 0 else 0
                lines.append(
                    f"{label:<15} {base_name:<12} {base_p50:>8.1f} "
                    f"{tq_name:<12} {tq_p50:>8.1f} {speedup:>7.2f}x"
                )

    lines.append("-" * 75)
    return "\n".join(lines)


def run_long_benchmarks(
    texts: list[dict[str, str]] | None = None,
    warmup: int = 2,
    repeat: int = 10,
    output: str | None = None,
) -> list[dict[str, Any]]:
    """Run long-form TTS benchmarks for TQ-relevant runners.

    Args:
        texts: List of text/language/label dicts. Defaults to LONG_TEXTS.
        warmup: Number of warmup runs (discarded).
        repeat: Number of measured runs per text.
        output: Output JSON path.

    Returns:
        List of result dicts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = texts or LONG_TEXTS

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
            results = bench_runner(cls, name, texts, warmup, repeat)
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
    out_path = output or str(RESULTS_DIR / "e2e_long_benchmarks.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Long-form TTS benchmark for TurboQuant evaluation"
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
        default=10,
        help="Number of measured runs per text (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/e2e_long_benchmarks.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    args = _parse_args()
    run_long_benchmarks(
        warmup=args.warmup,
        repeat=args.repeat,
        output=args.output,
    )
