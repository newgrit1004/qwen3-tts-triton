"""Overview tab for the Streamlit dashboard.

Displays GPU info, 3-Tier verification summary badges,
and a quick benchmark summary at a glance.
"""

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.gpu_info import get_gpu_info
from ui.i18n import t

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"


def render_overview_tab() -> None:
    """Render the Overview tab with GPU info, verification, and benchmarks."""
    _render_gpu_details()
    st.markdown("---")
    _render_verification_summary()
    st.markdown("---")
    _render_quick_benchmarks()


def _render_gpu_details() -> None:
    """Render full GPU information card."""
    st.subheader(t("overview.gpu_info"))
    info = get_gpu_info()

    col1, col2, col3 = st.columns(3)
    col1.metric(t("overview.gpu"), info["name"])
    col2.metric(t("overview.driver"), info["driver_version"])

    total = info["total_vram_gb"]
    used = info["used_vram_gb"]
    free = info["free_vram_gb"]
    col3.metric(
        t("overview.vram"),
        f"{used:.1f} / {total:.1f} GB",
        delta=t("overview.vram_free", free=f"{free:.1f}"),
    )

    if total > 0:
        st.progress(
            used / total,
            text=t(
                "overview.vram_usage",
                used=f"{used:.1f}",
                total=f"{total:.1f}",
            ),
        )

    extras: list[str] = []
    if info["temperature_c"] is not None:
        extras.append(t("overview.temperature", temp=info["temperature_c"]))
    if info["utilization_pct"] is not None:
        extras.append(t("overview.utilization", pct=info["utilization_pct"]))
    if extras:
        st.caption(" | ".join(extras))


def _render_verification_summary() -> None:
    """Render compact 3-Tier verification badges."""
    st.subheader(t("overview.verification"))

    report = _load_verification_report()
    if not report:
        st.info(t("overview.no_verification"))
        return

    timestamp = report.get("timestamp", "")
    if timestamp:
        st.caption(t("overview.last_run", timestamp=timestamp))

    col1, col2, col3 = st.columns(3)

    tier1 = report.get("tier1")
    if tier1:
        passed = tier1.get("passed", 0)
        total = tier1.get("total", 0)
        badge = _badge(tier1.get("status"))
        col1.metric(t("overview.tier1"), f"{badge} {passed}/{total}")
    else:
        col1.metric(t("overview.tier1"), t("overview.not_run"))

    tier2 = report.get("tier2")
    if tier2:
        layers = tier2.get("layers", {})
        min_cos = min(v["cosine_sim"] for v in layers.values()) if layers else 0.0
        badge = _badge(tier2.get("status"))
        col2.metric(t("overview.tier2"), f"{badge} min={min_cos:.4f}")
    else:
        col2.metric(t("overview.tier2"), t("overview.not_run"))

    tier3 = report.get("tier3")
    if tier3:
        badge = _badge(tier3.get("overall_verdict", tier3.get("status")))
        col3.metric(t("overview.tier3"), badge)
    else:
        col3.metric(t("overview.tier3"), t("overview.not_run"))


def _render_quick_benchmarks() -> None:
    """Render quick benchmark summary from e2e_benchmarks.json."""
    st.subheader(t("overview.bench_summary"))

    data = _load_e2e_benchmarks()
    if not data:
        st.info(t("overview.no_e2e"))
        return

    # Build summary table
    rows: list[dict[str, Any]] = []
    for entry in data:
        runner = _normalize_runner(entry["runner"])
        load_time = entry.get("model_load_time_s")
        load_str = f"{load_time:.1f}" if load_time is not None else "-"
        rows.append(
            {
                t("table.runner"): runner,
                t("table.language"): entry["language"],
                t("table.model_load_time"): load_str,
                t("table.mean_latency"): f"{entry['time_ms']['mean']:.0f}",
                t("table.rtf_mean"): f"{entry['rtf']['mean']:.2f}",
                t("table.peak_vram"): f"{entry['peak_vram_gb']:.3f}",
            }
        )

    # Find fastest runner by mean latency
    base_mean = None
    fastest_name = None
    fastest_mean = float("inf")
    for entry in data:
        name = _normalize_runner(entry["runner"])
        mean_ms = entry["time_ms"]["mean"]
        if name == "Base" and base_mean is None:
            base_mean = mean_ms
        if mean_ms < fastest_mean:
            fastest_mean = mean_ms
            fastest_name = name

    if fastest_name and base_mean and fastest_mean > 0:
        speedup = base_mean / fastest_mean
        st.success(
            t(
                "overview.fastest",
                name=fastest_name,
                speedup=f"{speedup:.1f}",
            )
        )

    st.dataframe(rows, width="stretch", hide_index=True)


def _badge(status: str | None) -> str:
    """Convert status to a text badge."""
    if status == "PASS":
        return "PASS"
    if status == "FAIL":
        return "FAIL"
    return "PENDING"


def _normalize_runner(name: str) -> str:
    """Normalize runner names (e.g. 'Turbo' -> 'Hybrid')."""
    if name == "Turbo":
        return "Hybrid"
    return name


def _load_verification_report() -> dict[str, Any] | None:
    """Load verification_report.json from the results directory."""
    report_path = RESULTS_DIR / "verification_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_e2e_benchmarks() -> list[dict[str, Any]]:
    """Load e2e_benchmarks.json from the results directory."""
    bench_path = RESULTS_DIR / "e2e_benchmarks.json"
    if not bench_path.exists():
        return []
    try:
        return json.loads(bench_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
