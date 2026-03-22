"""Shared pytest fixtures for qwen3-tts-triton tests."""

import pytest
import torch


@pytest.fixture
def device() -> str:
    """Return CUDA device if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"
