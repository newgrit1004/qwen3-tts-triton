"""Phase 4: KV cache memory measurement benchmark.

Measures TurboQuant KV cache compression ratio vs uncompressed FP16/BF16
across various sequence lengths, bit-widths, and model configurations.

Can run on CPU (synthetic data) or GPU (actual VRAM measurement).

Usage:
    python -m benchmark.bench_kv_memory [--device cpu|cuda] [--output PATH]
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from qwen3_tts_triton.kernels.turboquant import TurboQuantKVCache

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Qwen3-TTS 1.7B Talker config
QWEN3_TTS_CONFIG = {
    "num_layers": 28,
    "num_kv_heads": 8,
    "head_dim": 128,
}

SEQ_LENGTHS = [128, 256, 512, 1024, 2048]
BIT_WIDTHS = [3, 4]


def measure_cache_memory(
    bits: int,
    seq_len: int,
    device: str = "cpu",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Measure TurboQuant KV cache memory for a given configuration.

    Args:
        bits: Quantization bit-width (3 or 4).
        seq_len: Sequence length to simulate.
        device: Device for tensors ("cpu" or "cuda").
        batch_size: Batch size.

    Returns:
        Dict with memory stats and timing info.
    """
    cfg = QWEN3_TTS_CONFIG
    cache = TurboQuantKVCache(
        bits=bits,
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        device=device,
        dtype=torch.bfloat16,
    )

    # Generate synthetic KV states
    torch.manual_seed(42)
    k = torch.randn(
        batch_size,
        cfg["num_kv_heads"],
        seq_len,
        cfg["head_dim"],
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    # Measure quantization time (all layers)
    start = time.perf_counter()
    for layer_idx in range(cfg["num_layers"]):
        cache.update(k, v, layer_idx)
    quant_time = time.perf_counter() - start

    stats: dict[str, Any] = {**cache.get_memory_stats()}
    stats["bits"] = bits
    stats["seq_len"] = seq_len
    stats["batch_size"] = batch_size
    stats["device"] = device
    stats["quantize_time_s"] = round(quant_time, 4)
    stats["quantize_per_layer_ms"] = round(quant_time / cfg["num_layers"] * 1000, 3)

    return stats


def measure_gpu_vram(
    bits: int,
    seq_len: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Measure actual GPU VRAM usage for TurboQuant KV cache.

    Uses torch.cuda.memory_allocated() for precise measurement.

    Args:
        bits: Quantization bit-width.
        seq_len: Sequence length.
        batch_size: Batch size.

    Returns:
        Dict with VRAM measurements.
    """
    cfg = QWEN3_TTS_CONFIG

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    cache = TurboQuantKVCache(
        bits=bits,
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        device="cuda",
        dtype=torch.bfloat16,
    )

    rotation_mem = torch.cuda.memory_allocated() - baseline

    torch.manual_seed(42)
    k = torch.randn(
        batch_size,
        cfg["num_kv_heads"],
        seq_len,
        cfg["head_dim"],
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    for layer_idx in range(cfg["num_layers"]):
        cache.update(k, v, layer_idx)

    # Free input tensors to measure only cache footprint
    del k, v
    torch.cuda.empty_cache()

    cache_mem = torch.cuda.memory_allocated() - baseline
    peak_mem = torch.cuda.max_memory_allocated() - baseline

    # Theoretical uncompressed: B * 2(K+V) * L * H * S * D * 2bytes
    uncompressed = (
        2
        * batch_size
        * cfg["num_layers"]
        * cfg["num_kv_heads"]
        * seq_len
        * cfg["head_dim"]
        * 2  # bf16 = 2 bytes
    )

    return {
        "bits": bits,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "rotation_mb": round(rotation_mem / 1024**2, 2),
        "cache_mb": round(cache_mem / 1024**2, 2),
        "peak_mb": round(peak_mem / 1024**2, 2),
        "uncompressed_mb": round(uncompressed / 1024**2, 2),
        "vram_ratio": round(uncompressed / cache_mem if cache_mem > 0 else 0, 2),
    }


def _format_table(results: list[dict[str, Any]], gpu: bool = False) -> str:
    """Format memory benchmark results as ASCII table."""
    if gpu:
        header = (
            f"{'Bits':>4} {'SeqLen':>7} {'Cache(MB)':>10} "
            f"{'Uncomp(MB)':>11} {'Ratio':>6} {'Rot(MB)':>8} "
            f"{'Peak(MB)':>9}"
        )
    else:
        header = (
            f"{'Bits':>4} {'SeqLen':>7} {'Comp(MB)':>9} "
            f"{'Uncomp(MB)':>11} {'Ratio':>6} {'Q time(ms)':>11}"
        )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        if gpu:
            lines.append(
                f"{r['bits']:>4} {r['seq_len']:>7} "
                f"{r['cache_mb']:>10.2f} {r['uncompressed_mb']:>11.2f} "
                f"{r['vram_ratio']:>6.2f} {r['rotation_mb']:>8.2f} "
                f"{r['peak_mb']:>9.2f}"
            )
        else:
            q_ms = r.get("quantize_time_s", 0) * 1000
            lines.append(
                f"{r['bits']:>4} {r['seq_len']:>7} "
                f"{r['compressed_mb']:>9.2f} {r['uncompressed_mb']:>11.2f} "
                f"{r['compression_ratio']:>6.2f} {q_ms:>11.1f}"
            )
    lines.append(sep)
    return "\n".join(lines)


def run_kv_memory_benchmark(
    device: str = "cpu",
    output: str | None = None,
) -> list[dict[str, Any]]:
    """Run KV cache memory benchmarks across seq lengths and bit-widths.

    Args:
        device: "cpu" for theoretical, "cuda" for actual VRAM measurement.
        output: Output JSON path.

    Returns:
        List of result dicts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    use_gpu = device == "cuda" and torch.cuda.is_available()

    for bits in BIT_WIDTHS:
        for seq_len in SEQ_LENGTHS:
            logger.info(
                "Measuring %d-bit, seq_len=%d on %s...",
                bits,
                seq_len,
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                result = measure_gpu_vram(bits, seq_len)
            else:
                result = measure_cache_memory(bits, seq_len, device="cpu")
            results.append(result)
            summary = {k: v for k, v in result.items() if k != "per_layer_mb"}
            logger.info("  → %s", summary)

    # Log table
    table = _format_table(results, gpu=use_gpu)
    for line in table.split("\n"):
        logger.info(line)

    # Save JSON
    out_path = output or str(RESULTS_DIR / "kv_memory_benchmark.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)

    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="TurboQuant KV cache memory benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for measurement (default: cpu)",
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
    run_kv_memory_benchmark(device=args.device, output=args.output)
