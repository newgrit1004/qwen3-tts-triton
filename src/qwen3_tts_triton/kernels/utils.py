"""Kernel utility functions for Triton kernels."""

import triton


def calculate_settings(n: int) -> tuple[int, int]:
    """Calculate BLOCK_SIZE and num_warps for a given dimension size.

    Based on Liger Kernel / Unsloth heuristics.

    Args:
        n: The dimension size (e.g. hidden_size).

    Returns:
        Tuple of (BLOCK_SIZE, num_warps).

    Raises:
        RuntimeError: If n exceeds the maximum fused size.
    """
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} "
            f"exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps
