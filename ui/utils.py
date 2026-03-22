"""UI helper functions for VRAM measurement, RTF/TTFA calculation.

Provides utilities shared across the Streamlit dashboard for measuring
GPU memory usage, computing audio quality metrics (RTF, TTFA), formatting
display values, and loading benchmark result files.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_vram_usage_gb() -> float:
    """Get current GPU VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def get_peak_vram_gb() -> float:
    """Get peak GPU VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_vram_stats() -> None:
    """Reset VRAM statistics and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def calculate_rtf(
    audio_samples: int,
    sample_rate: int,
    generation_time_s: float,
) -> float:
    """Calculate Real-Time Factor.

    RTF = audio_duration / generation_time.
    RTF > 1 means faster than real-time.

    Args:
        audio_samples: Number of audio samples generated.
        sample_rate: Audio sample rate in Hz.
        generation_time_s: Wall-clock generation time in seconds.

    Returns:
        Real-Time Factor (higher is better).
    """
    if generation_time_s <= 0 or sample_rate <= 0:
        return 0.0
    audio_duration = audio_samples / sample_rate
    return audio_duration / generation_time_s


def calculate_ttfa_s(first_chunk_time_s: float, start_time_s: float) -> float:
    """Calculate Time To First Audio in seconds.

    Args:
        first_chunk_time_s: Timestamp when first audio chunk was ready.
        start_time_s: Timestamp when generation started.

    Returns:
        TTFA in seconds.
    """
    return max(0.0, first_chunk_time_s - start_time_s)


def format_delta_percent(current: float, baseline: float) -> tuple[str, float]:
    """Format improvement percentage vs baseline for st.metric delta.

    Args:
        current: Current value (e.g. latency in seconds).
        baseline: Baseline value to compare against.

    Returns:
        Tuple of (formatted string, raw delta percentage).
        Negative delta means improvement (lower is better for latency).
    """
    if baseline <= 0:
        return ("N/A", 0.0)
    delta_pct = ((current - baseline) / baseline) * 100
    sign = "+" if delta_pct >= 0 else ""
    return (f"{sign}{delta_pct:.1f}%", delta_pct)


def format_speedup(baseline_time: float, optimized_time: float) -> str:
    """Format speedup ratio as string.

    Args:
        baseline_time: Baseline execution time.
        optimized_time: Optimized execution time.

    Returns:
        Formatted speedup string like "2.5x".
    """
    if optimized_time <= 0:
        return "N/A"
    ratio = baseline_time / optimized_time
    return f"{ratio:.2f}x"


def load_benchmark_results(path: str) -> list[dict[str, Any]]:
    """Load benchmark results from JSON file.

    Args:
        path: Path to the JSON results file.

    Returns:
        List of result dictionaries, empty list on error.
    """
    import json
    from pathlib import Path as P

    results_path = P(path)
    if not results_path.exists():
        logger.warning("Results file not found: %s", path)
        return []

    try:
        return json.loads(results_path.read_text())
    except json.JSONDecodeError:
        logger.exception("Failed to parse results: %s", path)
        return []
