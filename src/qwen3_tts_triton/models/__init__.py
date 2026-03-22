"""Model runners and patching for Qwen3-TTS."""

from qwen3_tts_triton.models.base_runner import BaseRunner
from qwen3_tts_triton.models.faster_runner import FasterRunner
from qwen3_tts_triton.models.patching import apply_triton_kernels
from qwen3_tts_triton.models.triton_faster_runner import TritonFasterRunner
from qwen3_tts_triton.models.triton_runner import TritonRunner

__all__ = [
    "BaseRunner",
    "FasterRunner",
    "TritonFasterRunner",
    "TritonRunner",
    "apply_triton_kernels",
    "get_runner_class",
]

_RUNNER_MAP: dict[str, type] = {
    "base": BaseRunner,
    "faster": FasterRunner,
    "triton": TritonRunner,
    "hybrid": TritonFasterRunner,
}


def get_runner_class(name: str) -> type:
    """Look up a runner class by short name.

    Args:
        name: Runner name (base, faster, triton, hybrid).

    Returns:
        The runner class.

    Raises:
        KeyError: If the runner name is unknown.
    """
    key = name.lower()
    if key not in _RUNNER_MAP:
        available = ", ".join(sorted(_RUNNER_MAP))
        msg = f"Unknown runner '{name}'. Available: {available}"
        raise KeyError(msg)
    return _RUNNER_MAP[key]
