"""Micro-benchmarks for Triton kernels vs PyTorch baselines.

Compares latency (us) and peak GPU memory for each kernel pair.
Results saved to benchmark/results/kernel_benchmarks.json.
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import triton.testing

from qwen3_tts_triton.kernels.fused_norm_residual import triton_fused_add_rms_norm
from qwen3_tts_triton.kernels.rms_norm import triton_rms_norm
from qwen3_tts_triton.kernels.swiglu import triton_swiglu_forward

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Default benchmark dimensions (Qwen3-TTS Talker)
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
HEAD_DIM = 128
N_HEADS = 16
SEQ_LEN = 512
BATCH_SIZE = 1


def _reset_memory() -> None:
    """Empty CUDA cache and reset peak memory statistics."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb() -> float:
    """Return peak GPU memory allocated since last reset, in megabytes.

    Returns:
        Peak memory usage in MB.
    """
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _measure_compile_time(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Measure Triton JIT compilation time on first invocation.

    Args:
        fn: Triton kernel wrapper function to call.
        *args: Positional arguments for fn.
        **kwargs: Keyword arguments for fn.

    Returns:
        Wall-clock seconds for the first call (includes JIT compilation).
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return round(time.perf_counter() - start, 3)


def _pytorch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute RMSNorm using plain PyTorch (reference baseline).

    Args:
        x: Input tensor of shape (..., hidden_size).
        weight: Scale parameter of shape (hidden_size,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized and scaled tensor with the same shape as x.
    """
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def _pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU activation using plain PyTorch (reference baseline).

    Args:
        gate: Gate projection tensor of shape (..., intermediate_size).
        up: Up projection tensor of the same shape as gate.

    Returns:
        Element-wise product of silu(gate) and up.
    """
    return torch.nn.functional.silu(gate) * up


def _pytorch_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused residual add + RMSNorm using plain PyTorch (reference baseline).

    Args:
        x: Input tensor of shape (..., hidden_size).
        residual: Residual tensor of the same shape as x.
        weight: Scale parameter of shape (hidden_size,).
        eps: Epsilon for numerical stability.

    Returns:
        Tuple of (normalized output, updated residual) both with the same
        shape as x.
    """
    s = x + residual
    variance = s.to(torch.float32).pow(2).mean(-1, keepdim=True)
    s_norm = s * torch.rsqrt(variance + eps)
    return s_norm * weight, s


def bench_rms_norm() -> dict[str, Any]:
    """Benchmark RMSNorm: PyTorch vs Triton."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    w = torch.ones(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    # PyTorch baseline
    _reset_memory()
    pt_us = triton.testing.do_bench(lambda: _pytorch_rms_norm(x, w)) * 1000
    pt_mem = _peak_memory_mb()

    # Triton kernel
    compile_time = _measure_compile_time(triton_rms_norm, x, w)
    _reset_memory()
    tr_us = triton.testing.do_bench(lambda: triton_rms_norm(x, w)) * 1000
    tr_mem = _peak_memory_mb()

    return {
        "kernel": "RMSNorm",
        "hidden_size": HIDDEN_SIZE,
        "pytorch_us": round(pt_us, 2),
        "triton_us": round(tr_us, 2),
        "speedup": round(pt_us / tr_us, 2) if tr_us > 0 else 0,
        "pytorch_mem_mb": round(pt_mem, 2),
        "triton_mem_mb": round(tr_mem, 2),
        "compile_time_s": compile_time,
    }


def bench_swiglu() -> dict[str, Any]:
    """Benchmark SwiGLU: PyTorch vs Triton."""
    gate = torch.randn(
        BATCH_SIZE, SEQ_LEN, INTERMEDIATE_SIZE, device=DEVICE, dtype=DTYPE
    )
    up = torch.randn_like(gate)

    _reset_memory()
    pt_us = triton.testing.do_bench(lambda: _pytorch_swiglu(gate, up)) * 1000
    pt_mem = _peak_memory_mb()

    compile_time = _measure_compile_time(triton_swiglu_forward, gate, up)
    _reset_memory()
    tr_us = triton.testing.do_bench(lambda: triton_swiglu_forward(gate, up)) * 1000
    tr_mem = _peak_memory_mb()

    return {
        "kernel": "SwiGLU",
        "intermediate_size": INTERMEDIATE_SIZE,
        "pytorch_us": round(pt_us, 2),
        "triton_us": round(tr_us, 2),
        "speedup": round(pt_us / tr_us, 2) if tr_us > 0 else 0,
        "pytorch_mem_mb": round(pt_mem, 2),
        "triton_mem_mb": round(tr_mem, 2),
        "compile_time_s": compile_time,
    }


def bench_mrope() -> dict[str, Any] | None:
    """Benchmark M-RoPE: PyTorch vs Triton."""
    try:
        from qwen3_tts_triton.kernels.rope import triton_mrope_forward
    except (ImportError, ModuleNotFoundError):
        logger.warning("kernels.rope not available, skipping M-RoPE bench")
        return None

    mrope_section = [24, 20, 20]  # Qwen3-TTS sections
    half_hd = HEAD_DIM // 2

    # Input tensors: q (bsz, n_q_head, seq_len, head_dim)
    q = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(
        BATCH_SIZE, N_HEADS // 2, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    # cos/sin: (3, bsz, seq_len, head_dim) — 3 positional dims (t, h, w)
    cos = torch.randn(3, BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(3, BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    def _pytorch_mrope(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference M-RoPE: section-wise interleaved rotation in PyTorch."""
        q_out = q.clone().transpose(1, 2)  # (bsz, seq, heads, hd)
        k_out = k.clone().transpose(1, 2)
        cos_t = cos.contiguous()
        sin_t = sin.contiguous()

        boundaries = [0, mrope_section[0], mrope_section[0] + mrope_section[1], half_hd]
        for dim_idx in range(3):
            s = boundaries[dim_idx]
            e = boundaries[dim_idx + 1]
            c = cos_t[dim_idx]  # (bsz, seq, hd)
            sn = sin_t[dim_idx]

            for t in (q_out, k_out):
                even = t[..., 2 * s : 2 * e : 2]
                odd = t[..., 2 * s + 1 : 2 * e + 1 : 2]
                c_sl = c[:, :, 2 * s : 2 * e : 2].unsqueeze(2)
                s_sl = sn[:, :, 2 * s : 2 * e : 2].unsqueeze(2)
                new_even = even * c_sl - odd * s_sl
                new_odd = odd * c_sl + even * s_sl
                t[..., 2 * s : 2 * e : 2] = new_even
                t[..., 2 * s + 1 : 2 * e + 1 : 2] = new_odd

        return q_out.transpose(1, 2), k_out.transpose(1, 2)

    # PyTorch baseline
    _reset_memory()
    pt_us = triton.testing.do_bench(lambda: _pytorch_mrope(q, k, cos, sin)) * 1000
    pt_mem = _peak_memory_mb()

    # Triton kernel — compile time
    compile_time = _measure_compile_time(
        triton_mrope_forward, q.clone(), k.clone(), cos, sin, mrope_section
    )

    _reset_memory()
    tr_us = (
        triton.testing.do_bench(
            lambda: triton_mrope_forward(q.clone(), k.clone(), cos, sin, mrope_section)
        )
        * 1000
    )
    tr_mem = _peak_memory_mb()

    return {
        "kernel": "M-RoPE",
        "head_dim": HEAD_DIM,
        "n_heads": N_HEADS,
        "pytorch_us": round(pt_us, 2),
        "triton_us": round(tr_us, 2),
        "speedup": round(pt_us / tr_us, 2) if tr_us > 0 else 0,
        "pytorch_mem_mb": round(pt_mem, 2),
        "triton_mem_mb": round(tr_mem, 2),
        "compile_time_s": compile_time,
    }


def bench_fused_norm_residual() -> dict[str, Any]:
    """Benchmark fused Norm+Residual: PyTorch vs Triton."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    residual = torch.randn_like(x)
    w = torch.ones(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    _reset_memory()
    pt_us = (
        triton.testing.do_bench(lambda: _pytorch_fused_add_rms_norm(x, residual, w))
        * 1000
    )
    pt_mem = _peak_memory_mb()

    compile_time = _measure_compile_time(triton_fused_add_rms_norm, x, residual, w)
    _reset_memory()
    tr_us = (
        triton.testing.do_bench(lambda: triton_fused_add_rms_norm(x, residual, w))
        * 1000
    )
    tr_mem = _peak_memory_mb()

    return {
        "kernel": "FusedNorm+Residual",
        "hidden_size": HIDDEN_SIZE,
        "pytorch_us": round(pt_us, 2),
        "triton_us": round(tr_us, 2),
        "speedup": round(pt_us / tr_us, 2) if tr_us > 0 else 0,
        "pytorch_mem_mb": round(pt_mem, 2),
        "triton_mem_mb": round(tr_mem, 2),
        "compile_time_s": compile_time,
    }


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format kernel benchmark results as an ASCII table string.

    Args:
        results: List of result dicts from bench_* functions.

    Returns:
        Multi-line ASCII table string ready for logging.
    """
    header = (
        f"{'Kernel':<22} {'PyTorch(us)':>12} {'Triton(us)':>12} "
        f"{'Speedup':>8} {'PT Mem(MB)':>11} {'TR Mem(MB)':>11}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        if r.get("status") == "not_implemented":
            lines.append(f"{r['kernel']:<22} {'N/A':>12} {'N/A':>12}")
            continue
        lines.append(
            f"{r['kernel']:<22} {r['pytorch_us']:>12.2f} "
            f"{r['triton_us']:>12.2f} {r['speedup']:>7.2f}x "
            f"{r['pytorch_mem_mb']:>10.2f} {r['triton_mem_mb']:>10.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_all_benchmarks() -> list[dict[str, Any]]:
    """Run all kernel benchmarks and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        bench_rms_norm,
        bench_swiglu,
        bench_mrope,
        bench_fused_norm_residual,
    ]

    results = []
    for bench_fn in benchmarks:
        logger.info("Running %s...", bench_fn.__name__)
        result = bench_fn()
        if result is not None:
            results.append(result)

    # Log table
    table = _format_table(results)
    for line in table.split("\n"):
        logger.info(line)

    # Save JSON
    out_path = RESULTS_DIR / "kernel_benchmarks.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    run_all_benchmarks()
