"""Tests for kernel utility functions."""

import pytest

from qwen3_tts_triton.kernels.utils import calculate_settings


@pytest.mark.parametrize(
    "n, expected_block, expected_warps",
    [
        (1, 1, 4),
        (64, 64, 4),
        (128, 128, 4),
        (2000, 2048, 8),  # non-power-of-2 rounds up
        (2048, 2048, 8),  # boundary: >=2048 → 8 warps
        (4096, 4096, 8),
        (8192, 8192, 16),  # boundary: >=8192 → 16 warps
        (16384, 16384, 16),
        (32768, 32768, 32),  # boundary: >=32768 → 32 warps
        (65536, 65536, 32),  # max allowed
    ],
)
def test_calculate_settings(n: int, expected_block: int, expected_warps: int) -> None:
    """calculate_settings returns correct BLOCK_SIZE and num_warps."""
    block_size, num_warps = calculate_settings(n)
    assert (
        block_size == expected_block
    ), f"n={n}: expected BLOCK_SIZE={expected_block}, got {block_size}"
    assert (
        num_warps == expected_warps
    ), f"n={n}: expected num_warps={expected_warps}, got {num_warps}"


def test_calculate_settings_exceeds_max_raises() -> None:
    """calculate_settings raises RuntimeError when n exceeds MAX_FUSED_SIZE."""
    with pytest.raises(RuntimeError, match="exceeds the recommended"):
        calculate_settings(65537)


def test_calculate_settings_block_is_power_of_two() -> None:
    """BLOCK_SIZE is always a power of two for any valid n."""
    for n in [1, 3, 7, 100, 1000, 2048, 5000, 12345, 65536]:
        block_size, _ = calculate_settings(n)
        assert (
            block_size & (block_size - 1) == 0 or block_size == 0
        ), f"n={n}: BLOCK_SIZE={block_size} is not a power of 2"
        assert block_size >= n, f"n={n}: BLOCK_SIZE={block_size} is smaller than n"
