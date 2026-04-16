"""Partial Patching comparison tab.

Loads bench_partial sweep results (JSON + audio) and displays:
- Side-by-side audio playback for each config
- Latency / RTF / VRAM bar charts
- Speedup comparison
"""

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_sweep_results() -> list[Path]:
    """Find all partial_sweep_*.json files."""
    if not _RESULTS_DIR.exists():
        return []
    return sorted(_RESULTS_DIR.glob("partial_sweep_*.json"), reverse=True)


def _find_eval_results() -> list[Path]:
    """Find all eval_partial_*.json files."""
    if not _RESULTS_DIR.exists():
        return []
    return sorted(_RESULTS_DIR.glob("eval_partial_*.json"), reverse=True)


def _load_results(path: Path) -> dict[str, Any] | None:
    """Load a sweep results JSON."""
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _get_unique_runners(results: list[dict[str, Any]]) -> list[str]:
    """Get unique runner names in order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for r in results:
        name = r["runner"]
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _get_unique_languages(results: list[dict[str, Any]]) -> list[str]:
    """Get unique language codes."""
    return sorted({r["language"] for r in results})


def _render_audio_comparison(results: list[dict[str, Any]], lang_filter: str) -> None:
    """Render audio playback grid."""
    st.subheader("Audio Comparison")

    # Filter results that have audio and match language
    audio_results = [
        r
        for r in results
        if r.get("audio_path")
        and (lang_filter == "All" or r["language"] == lang_filter)
    ]

    if not audio_results:
        st.info("No audio samples found. Run benchmark with `--save-audio`.")
        return

    # Group by sample_idx
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for r in audio_results:
        idx = r.get("sample_idx", 0)
        by_sample.setdefault(idx, []).append(r)

    for sample_idx, sample_results in sorted(by_sample.items()):
        text = sample_results[0].get("text", "")
        lang = sample_results[0].get("language", "")
        st.markdown(f"**Sample {sample_idx}** ({lang}): _{text}_")

        cols = st.columns(len(sample_results))
        for col, r in zip(cols, sample_results):
            audio_path = _PROJECT_ROOT / r["audio_path"]
            with col:
                st.caption(r["runner"])
                if audio_path.exists():
                    st.audio(str(audio_path))
                else:
                    st.warning(f"Missing: {r['audio_path']}")
        st.divider()


def _render_latency_chart(results: list[dict[str, Any]], lang_filter: str) -> None:
    """Render latency comparison bar chart."""
    import plotly.graph_objects as go

    st.subheader("Latency (ms)")

    filtered = [
        r for r in results if lang_filter == "All" or r["language"] == lang_filter
    ]

    # Average across samples per runner
    runner_stats: dict[str, dict[str, list[float]]] = {}
    for r in filtered:
        name = r["runner"]
        runner_stats.setdefault(name, {"mean": [], "std": []})
        runner_stats[name]["mean"].append(r["time_ms"]["mean"])
        runner_stats[name]["std"].append(r["time_ms"]["std"])

    runners = list(runner_stats.keys())
    means = [
        sum(runner_stats[r]["mean"]) / len(runner_stats[r]["mean"]) for r in runners
    ]
    stds = [sum(runner_stats[r]["std"]) / len(runner_stats[r]["std"]) for r in runners]

    colors = _get_colors(runners)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=runners,
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            marker_color=colors,
            text=[f"{m:.0f}" for m in means],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Latency (ms)",
        height=400,
        margin={"t": 30, "b": 30},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_rtf_chart(results: list[dict[str, Any]], lang_filter: str) -> None:
    """Render RTF comparison bar chart."""
    import plotly.graph_objects as go

    st.subheader("Real-Time Factor (higher = faster)")

    filtered = [
        r for r in results if lang_filter == "All" or r["language"] == lang_filter
    ]

    runner_rtfs: dict[str, list[float]] = {}
    for r in filtered:
        runner_rtfs.setdefault(r["runner"], []).append(r["rtf"]["mean"])

    runners = list(runner_rtfs.keys())
    means = [sum(runner_rtfs[r]) / len(runner_rtfs[r]) for r in runners]
    colors = _get_colors(runners)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=runners,
            y=means,
            marker_color=colors,
            text=[f"{m:.1f}x" for m in means],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="RTF",
        height=400,
        margin={"t": 30, "b": 30},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_vram_chart(results: list[dict[str, Any]], lang_filter: str) -> None:
    """Render VRAM comparison bar chart."""
    import plotly.graph_objects as go

    st.subheader("Peak VRAM (GB)")

    filtered = [
        r for r in results if lang_filter == "All" or r["language"] == lang_filter
    ]

    runner_vram: dict[str, list[float]] = {}
    for r in filtered:
        runner_vram.setdefault(r["runner"], []).append(r["peak_vram_gb"])

    runners = list(runner_vram.keys())
    means = [sum(runner_vram[r]) / len(runner_vram[r]) for r in runners]
    colors = _get_colors(runners)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=runners,
            y=means,
            marker_color=colors,
            text=[f"{m:.2f}" for m in means],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="VRAM (GB)",
        height=400,
        margin={"t": 30, "b": 30},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_speedup_table(results: list[dict[str, Any]]) -> None:
    """Render speedup vs Base table."""
    st.subheader("Speedup vs Base")

    # Compute base mean per language
    base_means: dict[str, float] = {}
    for r in results:
        if r["runner"] == "Base":
            base_means[r["language"]] = r["time_ms"]["mean"]

    rows = []
    for r in results:
        lang = r["language"]
        base_mean = base_means.get(lang)
        if base_mean and base_mean > 0:
            speedup = base_mean / r["time_ms"]["mean"]
        else:
            speedup = 1.0
        rows.append(
            {
                "Runner": r["runner"],
                "Language": lang,
                "Mean (ms)": round(r["time_ms"]["mean"], 1),
                "Std (ms)": round(r["time_ms"]["std"], 1),
                "RTF": round(r["rtf"]["mean"], 1),
                "Speedup": f"{speedup:.2f}x",
                "VRAM (GB)": r["peak_vram_gb"],
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _get_colors(runners: list[str]) -> list[str]:
    """Get consistent colors for runner names."""
    palette = {
        "Base": "#636EFA",
        "Hybrid": "#EF553B",
    }
    # Partial patching configs get green shades
    greens = ["#00CC96", "#00B4D8", "#48CAE4", "#90E0EF", "#ADE8F4"]
    green_idx = 0
    colors = []
    for r in runners:
        if r in palette:
            colors.append(palette[r])
        else:
            colors.append(greens[green_idx % len(greens)])
            green_idx += 1
    return colors


def _render_quality_table(eval_data: dict[str, Any]) -> None:
    """Render quality metrics table from eval_partial results."""
    st.subheader("Quality Metrics (vs Base)")

    runners = eval_data.get("runners", {})
    comparisons = eval_data.get("comparisons", [])
    cmp_map = {c["opt"]: c for c in comparisons}

    rows = []
    for name, stats in runners.items():
        cmp = cmp_map.get(name, {})
        rows.append(
            {
                "Runner": name,
                "UTMOS": f"{stats['utmos_mean']:.3f}\u00b1{stats['utmos_std']:.3f}",
                "CER": f"{stats['cer_mean']:.3f}\u00b1{stats['cer_std']:.3f}",
                "Spk Sim": f"{cmp.get('speaker_sim_mean', '-')}"
                if isinstance(cmp.get("speaker_sim_mean"), float)
                else "-",
                "UTMOS \u0394": f"{cmp.get('utmos_delta', '-')}",
                "CER \u0394": f"{cmp.get('cer_delta', '-')}",
                "Status": cmp.get("status", "(ref)"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Failures
    for cmp in comparisons:
        if cmp.get("failures"):
            st.warning(f"**{cmp['opt']}**: " + ", ".join(cmp["failures"]))


def _render_quality_by_language(eval_data: dict[str, Any]) -> None:
    """Render per-language UTMOS and CER grouped bar charts."""
    import plotly.graph_objects as go

    runners = eval_data.get("runners", {})

    # Collect all languages
    all_langs: set[str] = set()
    for stats in runners.values():
        per_lang = stats.get("per_language", {})
        all_langs.update(per_lang.keys())
    langs = sorted(all_langs)

    if not langs:
        st.info("No per-language data available.")
        return

    runner_names = list(runners.keys())
    colors = _get_colors(runner_names)

    # UTMOS by language
    st.subheader("UTMOS by Language")
    fig_utmos = go.Figure()
    for i, name in enumerate(runner_names):
        per_lang = runners[name].get("per_language", {})
        vals = [per_lang.get(lg, {}).get("utmos_mean", 0) for lg in langs]
        errs = [per_lang.get(lg, {}).get("utmos_std", 0) for lg in langs]
        fig_utmos.add_trace(
            go.Bar(
                name=name,
                x=langs,
                y=vals,
                error_y={"type": "data", "array": errs, "visible": True},
                marker_color=colors[i],
            )
        )
    fig_utmos.update_layout(
        barmode="group",
        yaxis_title="UTMOS",
        height=400,
        margin={"t": 30, "b": 30},
    )
    st.plotly_chart(fig_utmos, use_container_width=True)

    # CER by language
    st.subheader("CER by Language (lower = better)")
    fig_cer = go.Figure()
    for i, name in enumerate(runner_names):
        per_lang = runners[name].get("per_language", {})
        vals = [per_lang.get(lg, {}).get("cer_mean", 0) for lg in langs]
        errs = [per_lang.get(lg, {}).get("cer_std", 0) for lg in langs]
        fig_cer.add_trace(
            go.Bar(
                name=name,
                x=langs,
                y=vals,
                error_y={"type": "data", "array": errs, "visible": True},
                marker_color=colors[i],
            )
        )
    fig_cer.update_layout(
        barmode="group",
        yaxis_title="CER",
        height=400,
        margin={"t": 30, "b": 30},
    )
    st.plotly_chart(fig_cer, use_container_width=True)


def render_partial_tab() -> None:
    """Render the Partial Patching comparison tab."""
    # === Section 1: Quality Evaluation ===
    st.header("Quality Evaluation (Tier 3)")
    eval_files = _find_eval_results()

    if eval_files:
        selected_eval = st.selectbox(
            "Eval results file",
            eval_files,
            format_func=lambda p: p.name,
            key="eval_select",
        )
        if selected_eval:
            eval_data = _load_results(selected_eval)
            if eval_data:
                with st.expander("Eval metadata", expanded=False):
                    meta = {
                        k: v
                        for k, v in eval_data.items()
                        if k not in ("runners", "comparisons")
                    }
                    st.json(meta)

                _render_quality_table(eval_data)
                _render_quality_by_language(eval_data)
    else:
        st.info(
            "No quality eval results. Run:\n\n"
            "```bash\n"
            "uv run python -m benchmark.eval_partial \\\n"
            "    --patch-range 0,20 --mode full\n"
            "```"
        )

    # === Section 2: Speed Benchmark ===
    st.header("Speed Benchmark")
    sweep_files = _find_sweep_results()

    if not sweep_files:
        st.info(
            "No speed benchmark results. Run:\n\n"
            "```bash\n"
            "uv run python -m benchmark.bench_partial \\\n"
            "    --patch-range 0,20 --save-audio\n"
            "```"
        )
        return

    selected_file = st.selectbox(
        "Speed results file",
        sweep_files,
        format_func=lambda p: p.name,
        key="speed_select",
    )
    if selected_file is None:
        return

    data = _load_results(selected_file)
    if data is None:
        st.error(f"Failed to load {selected_file}")
        return

    results = data.get("results", [])
    meta = data.get("meta", {})

    if not results:
        st.warning("Results file is empty.")
        return

    with st.expander("Speed benchmark metadata", expanded=False):
        st.json(meta)

    languages = _get_unique_languages(results)
    lang_filter = st.selectbox(
        "Filter by language",
        ["All"] + languages,
        key="speed_lang",
    )

    _render_speedup_table(
        results
        if lang_filter == "All"
        else [r for r in results if r["language"] == lang_filter]
    )

    col1, col2 = st.columns(2)
    with col1:
        _render_latency_chart(results, lang_filter)
    with col2:
        _render_rtf_chart(results, lang_filter)

    _render_vram_chart(results, lang_filter)

    # === Section 3: Audio Comparison ===
    st.header("Audio Comparison")
    _render_audio_comparison(results, lang_filter)
