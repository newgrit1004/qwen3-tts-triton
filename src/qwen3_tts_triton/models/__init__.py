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
    "create_runner",
    "ALL_RUNNER_NAMES",
]

_RUNNER_MAP: dict[str, type] = {
    "base": BaseRunner,
    "faster": FasterRunner,
    "triton": TritonRunner,
    "hybrid": TritonFasterRunner,
}

# TurboQuant variants reuse same classes with enable_turboquant=True.
_TQ_VARIANTS: set[str] = {"base+tq", "triton+tq", "hybrid+tq"}

ALL_RUNNER_NAMES: list[str] = [
    "base",
    "base+tq",
    "triton",
    "triton+tq",
    "faster",
    "hybrid",
    "hybrid+tq",
]


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


def create_runner(name: str, **kwargs) -> BaseRunner:  # type: ignore[return-value]
    """Create a runner instance by name, including TurboQuant variants.

    Supports ``'base+tq'``, ``'triton+tq'``, and ``'hybrid+tq'``
    in addition to the base runner names.  TQ variants set
    ``enable_turboquant=True`` on the underlying runner class.

    Args:
        name: Runner name (base, base+tq, triton, triton+tq,
            faster, hybrid, hybrid+tq).
        **kwargs: Additional keyword arguments passed to the runner
            constructor.

    Returns:
        A runner instance.
    """
    key = name.lower()
    if key in _TQ_VARIANTS:
        base_name = key.removesuffix("+tq")
        cls = get_runner_class(base_name)
        return cls(enable_turboquant=True, **kwargs)  # type: ignore[call-arg]
    cls = get_runner_class(key)
    return cls(**kwargs)  # type: ignore[call-arg]
