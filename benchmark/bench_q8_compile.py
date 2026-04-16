"""Test torch.compile + W8A16 quantization on Base runner.

torch.compile fuses dequant+matmul into a single kernel,
which is the canonical way to get speedup from torchao.

Usage:
    uv run python benchmark/bench_q8_compile.py
"""

import gc
import logging
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

TEXT = {
    "text": "Hello, welcome to the Qwen3 text-to-speech system.",
    "language": "en",
}
WARMUP = 3
REPEAT = 5


def _bench(runner, name: str) -> dict:
    """Benchmark one runner, return stats."""
    for i in range(WARMUP):
        logger.info("[%s] warmup %d/%d", name, i + 1, WARMUP)
        runner.generate(text=TEXT["text"], language=TEXT["language"])
    torch.cuda.synchronize()

    timings = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(REPEAT):
        torch.cuda.empty_cache()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        runner.generate(
            text=TEXT["text"],
            language=TEXT["language"],
        )
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        timings.append(ms)
        logger.info("[%s] run %d: %.0fms", name, i + 1, ms)

    arr = np.array(timings)
    vram = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "runner": name,
        "mean": float(np.mean(arr)),
        "p50": float(np.median(arr)),
        "vram": round(vram, 2),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )

    from qwen3_tts_triton.models.base_runner import BaseRunner

    # --- 1. Base+Q8 (no compile) ---
    logger.info("=" * 50)
    logger.info("Base+Q8 (no compile)")
    logger.info("=" * 50)
    runner = BaseRunner(enable_w8a16=True)
    runner.load_model()
    torch.cuda.synchronize()
    r1 = _bench(runner, "Base+Q8")
    runner.unload_model()
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. Base+Q8 + torch.compile ---
    logger.info("=" * 50)
    logger.info("Base+Q8 + torch.compile")
    logger.info("=" * 50)
    runner2 = BaseRunner(enable_w8a16=True)
    runner2.load_model()

    # Compile just the talker (the bottleneck)
    talker = runner2._tts.model.talker
    logger.info("Compiling talker with torch.compile...")
    t0 = time.perf_counter()
    runner2._tts.model.talker = torch.compile(
        talker,
        mode="max-autotune",
    )
    logger.info(
        "torch.compile setup in %.1fs",
        time.perf_counter() - t0,
    )
    torch.cuda.synchronize()
    r2 = _bench(runner2, "Base+Q8+Compile")
    runner2.unload_model()
    gc.collect()
    torch.cuda.empty_cache()

    # --- Results ---
    sep = "-" * 50
    hdr = f"{'Runner':<20} {'Mean(ms)':>9} {'P50':>8} {'VRAM':>6}"
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in [r1, r2]:
        print(f"{r['runner']:<20} {r['mean']:>9.0f} {r['p50']:>8.0f} {r['vram']:>6.2f}")
    print(sep)
    speedup = r1["mean"] / r2["mean"]
    print(f"\ntorch.compile effect: {speedup:.2f}x")


if __name__ == "__main__":
    main()
