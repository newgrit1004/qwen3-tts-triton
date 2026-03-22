"""Shared pytest fixtures for kernel tests."""

import pytest


@pytest.fixture
def hidden_size() -> int:
    """Qwen3-TTS Talker hidden size."""
    return 2048


@pytest.fixture
def eps() -> float:
    """RMSNorm epsilon."""
    return 1e-6


@pytest.fixture
def intermediate_size() -> int:
    """Qwen3-TTS Talker intermediate size."""
    return 6144
