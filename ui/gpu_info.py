"""GPU information helper for the Streamlit dashboard."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information using pynvml with torch.cuda fallback.

    Returns:
        Dict with keys: name, driver_version, total_vram_gb, used_vram_gb,
        free_vram_gb, utilization_pct, temperature_c.
        Returns empty/default values if no GPU available.
    """
    info = _try_pynvml()
    if info is not None:
        return info

    info = _try_torch_cuda()
    if info is not None:
        return info

    return {
        "name": "No GPU detected",
        "driver_version": "N/A",
        "total_vram_gb": 0.0,
        "used_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "utilization_pct": None,
        "temperature_c": None,
    }


def _try_pynvml() -> dict[str, Any] | None:
    """Attempt to read GPU info via pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        name = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except Exception:
            gpu_util = None

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None

        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
        except Exception:
            driver = "N/A"

        return {
            "name": name if isinstance(name, str) else name.decode(),
            "driver_version": (driver if isinstance(driver, str) else driver.decode()),
            "total_vram_gb": mem.total / (1024**3),
            "used_vram_gb": mem.used / (1024**3),
            "free_vram_gb": mem.free / (1024**3),
            "utilization_pct": gpu_util,
            "temperature_c": temp,
        }
    except Exception:
        return None


def _try_torch_cuda() -> dict[str, Any] | None:
    """Attempt to read GPU info via torch.cuda."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0)
        return {
            "name": props.name,
            "driver_version": "N/A",
            "total_vram_gb": props.total_mem / (1024**3),
            "used_vram_gb": allocated / (1024**3),
            "free_vram_gb": (props.total_mem - allocated) / (1024**3),
            "utilization_pct": None,
            "temperature_c": None,
        }
    except Exception:
        return None
