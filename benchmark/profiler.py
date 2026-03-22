"""Profiling helper for Qwen3-TTS model forward pass.

Wraps torch.profiler to capture layer-by-layer timing,
export Chrome trace JSON, and log top-N slowest operations.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def profile_forward_pass(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    trace_name: str = "qwen3_tts_trace",
    top_n: int = 10,
    **forward_kwargs: Any,
) -> str:
    """Profile a single forward pass and export Chrome trace.

    Args:
        model: The model to profile.
        input_ids: Input tensor for the forward pass.
        trace_name: Name for the trace output file.
        top_n: Number of slowest operations to log.
        **forward_kwargs: Additional keyword arguments for model.forward().

    Returns:
        Path to the exported Chrome trace JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = RESULTS_DIR / f"{trace_name}.json"

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                model(input_ids, **forward_kwargs)

    # Export Chrome trace
    prof.export_chrome_trace(str(trace_path))
    logger.info("Chrome trace exported to %s", trace_path)

    # Log top-N slowest CUDA operations
    _log_top_operations(prof, top_n)

    return str(trace_path)


def _log_top_operations(
    prof: torch.profiler.profile,
    top_n: int,
) -> None:
    """Log the top-N slowest operations from profiler results."""
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=top_n,
    )
    logger.info("Top-%d slowest CUDA operations:\n%s", top_n, table)


def profile_kernel(
    fn: Any,
    *args: Any,
    name: str = "kernel",
    warmup: int = 10,
    repeat: int = 100,
    **kwargs: Any,
) -> dict[str, float]:
    """Profile a single kernel function call.

    Args:
        fn: Callable to profile.
        *args: Positional arguments for fn.
        name: Label for the profiled region.
        warmup: Number of warmup iterations.
        repeat: Number of timed iterations.
        **kwargs: Keyword arguments for fn.

    Returns:
        Dict with mean/min/max CUDA time in microseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities) as prof:
        for _ in range(repeat):
            with record_function(name):
                fn(*args, **kwargs)
        torch.cuda.synchronize()

    events = [e for e in prof.key_averages() if e.key == name and e.cuda_time_total > 0]

    if not events:
        logger.warning("No CUDA events found for '%s'", name)
        return {"mean_us": 0, "min_us": 0, "max_us": 0}

    evt = events[0]
    mean_us = evt.cuda_time_total / max(evt.count, 1)

    return {
        "mean_us": round(mean_us, 2),
        "total_us": round(evt.cuda_time_total, 2),
        "count": evt.count,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    logger.info("Profiler ready. Import and use profile_forward_pass().")
