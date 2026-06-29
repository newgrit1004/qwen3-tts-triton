"""Benchmarks tab for the Streamlit dashboard.

Three sections:
  A) Pre-computed E2E benchmarks (single-clip, batch size 1)
  B) Batched serving (4-way generate_batch matrix, v0.3.0)
  C) Kernel micro-benchmarks
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
    """Render the Benchmarks tab with three sections."""
    _render_e2e_benchmarks()
    st.markdown("---")
    _render_batched_serving()
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


# ── Section B: Batched Serving (v0.3.0) ─────────────────────────────


def _render_batched_serving() -> None:
    """Render the 4-way batched serving matrix from batched_matrix.json."""
    st.subheader(t("benchmarks.batched_title"))

    payload = _load_json_dict(RESULTS_DIR / "batched_matrix.json")
    rows = [r for r in payload.get("rows", []) if not r.get("oom")]
    if not rows:
        st.info(t("benchmarks.no_batched"))
        return

    cfg = payload.get("config", {})
    st.caption(
        t(
            "benchmarks.batched_caption",
            batch_size=cfg.get("batch_size", "?"),
            num_texts=cfg.get("num_texts", "?"),
        )
    )

    table: list[dict[str, Any]] = []
    chart_data: list[dict[str, Any]] = []
    for entry in rows:
        runner = _normalize_runner(entry["runner"])
        table.append(
            {
                t("table.runner"): runner,
                t("table.engine"): entry.get("engine", "-"),
                t("table.ms_per_step"): f"{entry['ms_per_step']:.1f}",
                t("table.rtf_mean"): f"{entry['rtf']:.1f}",
                t("table.per_sample_s"): f"{entry['per_sample_s']:.3f}",
                t("table.num_buckets"): entry.get("num_buckets", "-"),
                t("table.peak_vram"): f"{entry['peak_vram_gb']:.2f}",
            }
        )
        chart_data.append(
            {"runner": runner, "ms_per_step": float(entry["ms_per_step"])}
        )

    st.dataframe(table, width="stretch", hide_index=True)
    # ms/step = fair cross-engine speed (lower is better; RTF is length-confounded).
    st.caption(t("benchmarks.batched_chart_note"))
    st.bar_chart(chart_data, x="runner", y="ms_per_step", horizontal=True)


# ── Section C: Kernel Micro-benchmarks ──────────────────────────────


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
    """Normalize runner names to display form."""
    _MAP: dict[str, str] = {
        "Turbo": "Hybrid",
        "triton": "Triton",
        "base": "Base",
        "faster": "Faster",
        "hybrid": "Hybrid",
        "Base+TQ": "Base+TQ",
        "Triton+TQ": "Triton+TQ",
        "Hybrid+TQ": "Hybrid+TQ",
        "base+tq": "Base+TQ",
        "triton+tq": "Triton+TQ",
        "hybrid+tq": "Hybrid+TQ",
    }
    return _MAP.get(name, name)


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list from a file, returning [] on error."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON object from a file, returning {} on error."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}
