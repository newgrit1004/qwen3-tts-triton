"""Benchmarks tab for the Streamlit dashboard.

Two sections:
  A) Pre-computed E2E benchmarks
  B) Kernel micro-benchmarks
"""

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.charts import render_e2e_chart, render_kernel_chart
from ui.i18n import t

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"


def render_benchmarks_tab() -> None:
    """Render the Benchmarks tab with two sections."""
    _render_e2e_benchmarks()
    st.markdown("---")
    _render_kernel_benchmarks()


# ── Section A: Pre-computed E2E Benchmarks ──────────────────────────


def _render_e2e_benchmarks() -> None:
    """Render pre-computed E2E benchmarks from e2e_benchmarks.json."""
    st.subheader(t("benchmarks.e2e_title"))

    source_path = RESULTS_DIR / "e2e_benchmarks.json"
    data = _load_json_list(source_path)
    if not data:
        st.info(t("benchmarks.no_e2e"))
        return

    st.caption(f"`{source_path.relative_to(source_path.parent.parent.parent)}`")

    # Build table rows
    rows: list[dict[str, Any]] = []
    for entry in data:
        runner = _normalize_runner(entry["runner"])
        tm = entry["time_ms"]
        rt = entry["rtf"]
        load_time = entry.get("model_load_time_s")
        load_str = f"{load_time:.1f}" if load_time is not None else "-"
        rows.append(
            {
                t("table.runner"): runner,
                t("table.language"): entry["language"],
                t("table.model_load_time"): load_str,
                t("table.mean_ms"): f"{tm['mean']:.0f}",
                t("table.std_ms"): f"{tm['std']:.0f}",
                t("table.p50_ms"): f"{tm['p50']:.0f}",
                t("table.p95_ms"): f"{tm['p95']:.0f}",
                t("table.rtf_mean"): f"{rt['mean']:.2f}",
                t("table.peak_vram"): f"{entry['peak_vram_gb']:.3f}",
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)
    render_e2e_chart(data)


# ── Section B: Kernel Micro-benchmarks ──────────────────────────────


def _render_kernel_benchmarks() -> None:
    """Render kernel benchmark results if available."""
    st.subheader(t("benchmarks.kernel_title"))

    results = _load_json_list(RESULTS_DIR / "kernel_benchmarks.json")
    if not results:
        st.info(t("benchmarks.no_kernels"))
        return

    render_kernel_chart(results)


# ── Helpers ─────────────────────────────────────────────────────────


def _normalize_runner(name: str) -> str:
    """Normalize runner names (e.g. 'Turbo' -> 'Hybrid')."""
    if name == "Turbo":
        return "Hybrid"
    return name


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list from a file, returning [] on error."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []
