"""Plotly chart rendering functions for the Streamlit dashboard.

All chart functions handle plotly import failure gracefully with
st.warning fallbacks.
"""

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def render_comparison_chart(results: dict[str, dict[str, Any]]) -> None:
    """Render grouped bar chart comparing runner metrics.

    Displays TTFA, total time, and peak VRAM side-by-side for all runners
    that completed without error. Requires at least two valid results.

    Args:
        results: Mapping of runner display name to result dict.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    valid = {k: v for k, v in results.items() if not v.get("error")}
    if len(valid) < 2:
        return

    names = list(valid.keys())
    metrics = {
        "TTFA (s)": [valid[n]["ttfa_s"] for n in names],
        "Total Time (s)": [valid[n]["total_s"] for n in names],
        "Peak VRAM (GB)": [valid[n]["peak_vram_gb"] for n in names],
    }

    fig = go.Figure()
    for metric_name, values in metrics.items():
        fig.add_trace(go.Bar(name=metric_name, x=names, y=values))

    fig.update_layout(
        barmode="group",
        title="Speed & Memory Comparison",
        yaxis_title="Value",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def render_layer_chart(layers: dict[str, Any], *, key: str | None = None) -> None:
    """Render cosine similarity bar chart for decoder layers.

    Args:
        layers: Mapping of layer index string to dict with 'cosine_sim'.
        key: Unique Streamlit element key to avoid duplicate ID errors.
    """
    try:
        import plotly.graph_objects as go

        layer_ids = [f"L{k}" for k in layers]
        cos_vals = [v["cosine_sim"] for v in layers.values()]
        colors = ["#2ecc71" if v > 0.95 else "#e74c3c" for v in cos_vals]

        fig = go.Figure(data=[go.Bar(x=layer_ids, y=cos_vals, marker_color=colors)])
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="red",
            annotation_text="threshold=0.95",
        )
        fig.update_layout(
            title="Layer Cosine Similarity",
            yaxis_title="Cosine Similarity",
            yaxis_range=[0.94, 1.001],
            height=350,
            margin={"t": 40, "b": 30},
        )
        st.plotly_chart(fig, width="stretch", key=key)
    except ImportError:
        for k, v in layers.items():
            st.text(f"  L{k}: cos={v['cosine_sim']:.6f}")


def render_kernel_chart(results: list[dict[str, Any]]) -> None:
    """Render kernel benchmark bar chart (PyTorch vs Triton latency).

    Args:
        results: List of dicts with 'kernel', 'pytorch_us', 'triton_us'.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    kernels = [r["kernel"] for r in results if "pytorch_us" in r]
    pt_times = [r["pytorch_us"] for r in results if "pytorch_us" in r]
    tr_times = [r["triton_us"] for r in results if "triton_us" in r]

    if not kernels:
        return

    fig = go.Figure(
        data=[
            go.Bar(name="PyTorch", x=kernels, y=pt_times),
            go.Bar(name="Triton", x=kernels, y=tr_times),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Kernel Latency (us)",
        yaxis_title="Latency (us)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def render_e2e_chart(data: list[dict[str, Any]]) -> None:
    """Render grouped bar chart for pre-computed E2E benchmark data.

    Groups bars by runner, showing mean latency per language.

    Args:
        data: List of benchmark entries with 'runner', 'language',
            'time_ms.mean' nested structure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    if not data:
        return

    # Group by language
    languages = sorted({d["language"] for d in data})
    runners = []
    seen: set[str] = set()
    for d in data:
        name = _normalize_runner_name(d["runner"])
        if name not in seen:
            runners.append(name)
            seen.add(name)

    fig = go.Figure()
    for lang in languages:
        means = []
        for runner in runners:
            entry = _find_entry(data, runner, lang)
            means.append(entry["time_ms"]["mean"] if entry else 0)
        fig.add_trace(go.Bar(name=lang, x=runners, y=means))

    fig.update_layout(
        barmode="group",
        title="E2E Mean Latency by Runner & Language",
        yaxis_title="Latency (ms)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def _normalize_runner_name(name: str) -> str:
    """Normalize runner names (e.g. 'Turbo' -> 'Hybrid')."""
    if name == "Turbo":
        return "Hybrid"
    return name


def _find_entry(
    data: list[dict[str, Any]], runner: str, language: str
) -> dict[str, Any] | None:
    """Find a benchmark entry by normalized runner name and language."""
    for d in data:
        if _normalize_runner_name(d["runner"]) == runner and d["language"] == language:
            return d
    return None


def render_tier2_heatmap(
    pair_name: str,
    layers: dict[str, Any],
    *,
    key: str | None = None,
) -> None:
    """Render a heatmap of layer metrics (cosine_sim, SNR, relative_L2).

    Args:
        pair_name: Display name for the pair.
        layers: Mapping of layer index to dict with cosine_sim, snr_db, relative_l2.
        key: Unique Streamlit element key.
    """
    try:
        import plotly.graph_objects as go

        layer_ids = [f"L{k}" for k in layers]
        metrics = ["cosine_sim", "snr_db", "relative_l2"]
        display_names = ["Cosine Sim", "SNR (dB)", "Relative L2"]

        z_values = []
        annotations = []
        for mi, metric in enumerate(metrics):
            row = []
            for li, (lk, lv) in enumerate(layers.items()):
                val = lv.get(metric, 0)
                row.append(val)
                # Format annotation text
                if metric == "cosine_sim":
                    text = f"{val:.6f}"
                elif metric == "snr_db":
                    text = f"{val:.1f}"
                else:
                    text = f"{val:.4f}"
                annotations.append(
                    dict(
                        x=li,
                        y=mi,
                        text=text,
                        showarrow=False,
                        font=dict(
                            size=11,
                            color="white" if metric != "relative_l2" else "black",
                        ),
                    )
                )
            z_values.append(row)

        # Custom color scales per metric row
        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=layer_ids,
                y=display_names,
                colorscale=[
                    [0, "#e74c3c"],
                    [0.5, "#f39c12"],
                    [1, "#2ecc71"],
                ],
                showscale=False,
            )
        )
        fig.update_layout(
            title=pair_name,
            height=200,
            margin=dict(t=35, b=20, l=100, r=20),
            xaxis=dict(side="top"),
        )
        for ann in annotations:
            fig.add_annotation(ann)

        st.plotly_chart(fig, width="stretch", key=key)
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")


def render_tier3_radar(
    comparisons: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    key: str | None = None,
) -> None:
    """Render a radar chart comparing all Tier 3 comparisons.

    Args:
        comparisons: List of comparison dicts with ref, opt, cer_delta, utmos_delta,
            speaker_sim_mean.
        thresholds: Dict with cer_delta_max, utmos_delta_max, speaker_sim_min.
        key: Unique Streamlit element key.
    """
    try:
        import plotly.graph_objects as go

        # Normalize all metrics to 0-1 scale where 1.0 = at threshold boundary
        # For "lower is better" metrics: score = 1 - (value / threshold)
        # For "higher is better" metrics: score = value / threshold
        categories = ["CER delta", "UTMOS delta", "Speaker Sim"]

        cer_max = thresholds.get("cer_delta_max", 0.05)
        utmos_max = thresholds.get("utmos_delta_max", 0.3)
        sim_min = thresholds.get("speaker_sim_min", 0.75)

        colors = [
            "rgba(46, 204, 113, 0.6)",  # green
            "rgba(52, 152, 219, 0.6)",  # blue
            "rgba(155, 89, 182, 0.6)",  # purple
            "rgba(241, 196, 15, 0.6)",  # yellow
        ]
        line_colors = [
            "rgba(46, 204, 113, 1)",
            "rgba(52, 152, 219, 1)",
            "rgba(155, 89, 182, 1)",
            "rgba(241, 196, 15, 1)",
        ]

        fig = go.Figure()

        # Threshold boundary (the "safe zone")
        fig.add_trace(
            go.Scatterpolar(
                r=[1.0, 1.0, 1.0, 1.0],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(200, 200, 200, 0.15)",
                line=dict(color="rgba(150, 150, 150, 0.5)", dash="dash"),
                name="Threshold",
            )
        )

        for i, comp in enumerate(comparisons):
            label = f"{comp.get('ref', '?')} vs {comp.get('opt', '?')}"
            cer = comp.get("cer_delta", 0)
            utmos = comp.get("utmos_delta", 0)
            sim = comp.get("speaker_sim_mean", 0)

            # Normalize: lower-is-better -> invert so "better" = higher on radar
            cer_score = max(0, 1 - cer / cer_max) if cer_max > 0 else 1
            utmos_score = max(0, 1 - utmos / utmos_max) if utmos_max > 0 else 1
            sim_score = sim / sim_min if sim_min > 0 else 1

            values = [cer_score, utmos_score, sim_score]
            ci = i % len(colors)

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor=colors[ci],
                    line=dict(color=line_colors[ci], width=2),
                    name=label,
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.3],
                    tickvals=[0.5, 1.0],
                    ticktext=["50%", "Threshold"],
                ),
            ),
            height=400,
            margin=dict(t=30, b=30),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
            ),
        )

        st.plotly_chart(fig, width="stretch", key=key)
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
