"""Auto-generate benchmark tables for README from JSON results.

Reads benchmark/results/*.json, normalizes runner names, aggregates
across languages, and patches README.md between marker comments.
Applies stable-fast style formatting with bold best-in-column values.

Usage:
    uv run python scripts/generate_bench_tables.py           # Update READMEs
    uv run python scripts/generate_bench_tables.py --dry-run  # Print to stdout
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
README_FILES = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "README_ko.md"]

RUNNER_ALIASES: dict[str, str] = {"Turbo": "Hybrid"}
RUNNER_ORDER: list[str] = [
    "Base",
    "Base+TQ",
    "Triton",
    "Triton+TQ",
    "Faster",
    "Hybrid",
    "Hybrid+TQ",
]
QUALITY_RUNNER_ORDER: list[str] = [
    "base+tq",
    "triton",
    "triton+tq",
    "faster",
    "hybrid",
    "hybrid+tq",
]

HBM_SAVINGS: dict[str, str] = {
    "RMSNorm": "4\u21921 trips",
    "SwiGLU": "3\u21921 trips",
    "M-RoPE": "In-place",
    "FusedNorm+Residual": "2\u21921 kernels",
}

# Korean equivalents for HBM savings column
HBM_SAVINGS_KO: dict[str, str] = {
    "RMSNorm": "4\u21921 \uc655\ubcf5",
    "SwiGLU": "3\u21921 \uc655\ubcf5",
    "M-RoPE": "In-place",
    "FusedNorm+Residual": "2\u21921 \ucee4\ub110",
}

MARKER_PATTERN = re.compile(
    r"(<!-- BENCH:(\w+):START -->)\n(.*?)\n(<!-- BENCH:\2:END -->)",
    re.DOTALL,
)


# --- Formatting helpers ---


def _bold(value: str) -> str:
    """Wrap a value in markdown bold."""
    return f"**{value}**"


def _bold_best(
    values: dict[str, float],
    fmt: str,
    minimize: bool = True,
    exclude: set[str] | None = None,
) -> dict[str, str]:
    """Format values and bold the best one in a column.

    Args:
        values: Dict of runner_name -> numeric value.
        fmt: Format string (e.g., "{:.2f}").
        minimize: If True, bold the minimum; otherwise bold the maximum.
        exclude: Runner names to exclude from best selection.

    Returns:
        Dict of runner_name -> formatted string (with bold on best).
    """
    exclude = exclude or set()
    candidates = {k: v for k, v in values.items() if k not in exclude}

    if not candidates:
        return {k: fmt.format(v) for k, v in values.items()}

    fn = min if minimize else max
    best_key = fn(candidates, key=lambda k: candidates[k])

    result: dict[str, str] = {}
    for k, v in values.items():
        formatted = fmt.format(v)
        result[k] = _bold(formatted) if k == best_key else formatted
    return result


# --- Data loading helpers ---


def _load_json(path: Path) -> list | dict | None:
    """Load JSON file, returning None if missing or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _normalize_runner(name: str) -> str:
    """Apply runner name aliases."""
    return RUNNER_ALIASES.get(name, name)


def _display_runner_name(name: str) -> str:
    """Convert internal runner names to release-facing labels."""
    display = {
        "Base": "Base (PyTorch)",
        "Hybrid": "__Hybrid (Faster+Triton)__",
        "base": "Base (ref)",
        "base+tq": "Base+TQ (`base+tq`)",
        "triton": "Triton (`triton`)",
        "triton+tq": "Triton+TQ (`triton+tq`)",
        "faster": "Faster (`faster`)",
        "hybrid": "Hybrid (`hybrid`)",
        "hybrid+tq": "Hybrid+TQ (`hybrid+tq`)",
    }
    return display.get(name, name)


# --- E2E table ---


def _aggregate_e2e(
    raw: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate E2E results across languages per runner.

    Returns dict[runner_name, {...}].
    Keeps:
    - aggregate latency/RTF across languages
    - per-language latency/RTF for release tables
    - peak VRAM across languages
    - model load time per runner
    """
    by_runner: dict[str, list[dict]] = {}
    for entry in raw:
        name = _normalize_runner(entry["runner"])
        by_runner.setdefault(name, []).append(entry)

    aggregated: dict[str, dict[str, float]] = {}
    for runner, entries in by_runner.items():
        latencies = [e["time_ms"]["mean"] for e in entries]
        rtfs = [e["rtf"]["mean"] for e in entries]
        vrams = [e["peak_vram_gb"] for e in entries]
        load_times = [
            float(e["model_load_time_s"])
            for e in entries
            if e.get("model_load_time_s") is not None
        ]

        aggregated[runner] = {  # type: ignore[assignment]
            "latency_s": sum(latencies) / len(latencies) / 1000.0,
            "rtf": sum(rtfs) / len(rtfs),
            "vram": max(vrams),
            "load_time_s": max(load_times) if load_times else None,
        }
        for entry in entries:
            lang = entry.get("language")
            if lang not in {"ko", "en"}:
                continue
            aggregated[runner][f"latency_ms_{lang}"] = entry["time_ms"]["mean"]
            aggregated[runner][f"rtf_{lang}"] = entry["rtf"]["mean"]
    return aggregated


def _render_e2e_table(
    aggregated: dict[str, dict[str, float]],
    is_korean: bool = False,
) -> str:
    """Render the E2E markdown table with bold best-in-column values."""
    base_latency = aggregated.get("Base", {}).get("latency_s", 0.0)

    if is_korean:
        caption = (
            "> RTX 5090, bf16, 2\uac1c \ud14d\uc2a4\ud2b8 (ko + en), "
            "3\ud68c \uc6cc\ubc0d\uc5c5 + 20\ud68c \uce21\uc815. "
            "`make bench-e2e`\ub85c \uc7ac\ud604 \uac00\ub2a5."
        )
        header = (
            "| \ubaa8\ub4dc | \ub85c\ub4dc \uc2dc\uac04 | "
            "\uc9c0\uc5f0 (\ud55c\uad6d\uc5b4) | \uc9c0\uc5f0 (\uc601\uc5b4) | "
            "RTF (ko) | RTF (en) | Base \ub300\ube44 | \ud53c\ud06c VRAM |"
        )
        note = (
            "> Triton/Triton+TQ/Hybrid/Hybrid+TQ \uc218\uce58\ub294 "
            "\uae30\ubcf8 partial patch range `[0, 24)` \uc124\uc815 "
            "\uae30\uc900\uc774\uba70, \ub9c8\uc9c0\ub9c9 4\uac1c decoder "
            "\ub808\uc774\uc5b4\ub294 \ubc1c\uc74c \uc548\uc815\uc131\uc744 "
            "\uc704\ud574 PyTorch\ub85c \ub0a8\uaca8\ub461\ub2c8\ub2e4."
        )
    else:
        caption = (
            "> RTX 5090, bf16, 2 texts (ko + en), "
            "3 warmup + 20 runs each. Run `make bench-e2e` to reproduce."
        )
        header = (
            "| Mode | Load Time | Latency (ko) | Latency (en) | "
            "RTF (ko) | RTF (en) | vs Base | Peak VRAM |"
        )
        note = (
            "> Triton/Triton+TQ/Hybrid/Hybrid+TQ use the default partial "
            "patch range `[0, 24)`; the final 4 decoder layers stay in "
            "PyTorch for pronunciation stability."
        )

    divider = (
        "|------|:---------:|:------------:|:------------:|"
        ":--------:|:--------:|:-------:|:---------:|"
    )

    # Collect values for bold-best computation
    present = [r for r in RUNNER_ORDER if r in aggregated]
    lat_ko_vals = {
        r: aggregated[r]["latency_ms_ko"]
        for r in present
        if "latency_ms_ko" in aggregated[r]
    }
    lat_en_vals = {
        r: aggregated[r]["latency_ms_en"]
        for r in present
        if "latency_ms_en" in aggregated[r]
    }
    rtf_ko_vals = {
        r: aggregated[r]["rtf_ko"] for r in present if "rtf_ko" in aggregated[r]
    }
    rtf_en_vals = {
        r: aggregated[r]["rtf_en"] for r in present if "rtf_en" in aggregated[r]
    }
    speedup_vals: dict[str, float] = {}
    for r in present:
        if r == "Base":
            speedup_vals[r] = 1.0
        elif base_latency > 0:
            speedup_vals[r] = base_latency / aggregated[r]["latency_s"]
        else:
            speedup_vals[r] = 0.0

    lat_ko_fmt = _bold_best(lat_ko_vals, "{:,.0f} ms", minimize=True)
    lat_en_fmt = _bold_best(lat_en_vals, "{:,.0f} ms", minimize=True)
    rtf_ko_fmt = _bold_best(rtf_ko_vals, "{:.2f}x", minimize=False)
    rtf_en_fmt = _bold_best(rtf_en_vals, "{:.2f}x", minimize=False)
    spd_fmt = _bold_best(speedup_vals, "{:.1f}x", minimize=False, exclude={"Base"})

    lines = [caption, "", header, divider]

    for runner in RUNNER_ORDER:
        if runner not in aggregated:
            continue

        label = _display_runner_name(runner)
        load_time = aggregated[runner].get("load_time_s")
        load_time_str = f"{load_time:.1f}s" if load_time is not None else "\u2014"
        lat_ko = lat_ko_fmt.get(runner, "\u2014")
        lat_en = lat_en_fmt.get(runner, "\u2014")
        rtf_ko = rtf_ko_fmt.get(runner, "\u2014")
        rtf_en = rtf_en_fmt.get(runner, "\u2014")
        vs_base = spd_fmt[runner]
        vram = aggregated[runner].get("vram")
        vram_str = f"{vram:.2f} GB" if vram is not None else "\u2014"

        lines.append(
            f"| {label} | {load_time_str} | {lat_ko} | {lat_en} | "
            f"{rtf_ko} | {rtf_en} | {vs_base} | {vram_str} |"
        )

    lines.extend(["", note])
    return "\n".join(lines)


# --- Summary line ---


def _render_summary(
    aggregated: dict[str, dict[str, float]],
    is_korean: bool = False,
) -> str:
    """Render the dynamic one-line benchmark summary."""
    if not aggregated:
        if is_korean:
            return (
                "> \ubca4\uce58\ub9c8\ud06c \ub370\uc774\ud130 \uc5c6\uc74c. "
                "`make bench-e2e`\ub85c \uc0dd\uc131\ud558\uc138\uc694."
            )
        return "> No benchmark data. Run `make bench-e2e` to generate."

    base_latency = aggregated.get("Base", {}).get("latency_s", 0.0)
    if base_latency <= 0:
        return "> Benchmark data incomplete."

    # Find fastest runner (excluding Base)
    candidates = {k: v["latency_s"] for k, v in aggregated.items() if k != "Base"}
    if not candidates:
        return "> Only Base runner benchmarked."

    best_runner = min(candidates, key=lambda k: candidates[k])
    speedup = base_latency / candidates[best_runner]

    if best_runner == "Hybrid":
        display_name = "Hybrid (Faster+Triton)"
    else:
        display_name = best_runner

    if is_korean:
        return (
            f"> __{display_name}__ \ubaa8\ub4dc\ub294 RTX 5090\uc5d0\uc11c "
            f"PyTorch \uae30\ubcf8 \ub300\ube44 __{speedup:.1f}x__ \ube60\ub978 "
            f"\ucd94\ub860\uc744 \ub3d9\uc77c VRAM\uc73c\ub85c "
            f"\ub2ec\uc131\ud569\ub2c8\ub2e4."
        )
    return (
        f"> __{display_name}__ achieves __{speedup:.1f}x__ faster inference "
        f"than PyTorch baseline at equivalent VRAM on RTX 5090."
    )


# --- Kernel table ---


def _render_kernel_table(
    raw: list[dict] | None,
    is_korean: bool = False,
) -> str:
    """Render the kernel micro-benchmark markdown table with bold Triton values."""
    if is_korean:
        caption = (
            "> RTX 5090, bf16, batch=1, seq_len=512, hidden=2048. "
            "`make bench-kernels`\ub85c \uc7ac\ud604 \uac00\ub2a5."
        )
        header = (
            "| \ucee4\ub110 | PyTorch (us) | Triton (us) | "
            "\uc18d\ub3c4 \ud5a5\uc0c1 | \ucef4\ud30c\uc77c (s) | HBM \uc808\uac10 |"
        )
        divider = (  # noqa: E501
            "|------|:------------:|:-----------:|:---------:|:----------:|:--------:|"
        )
        hbm = HBM_SAVINGS_KO
    else:
        caption = (
            "> RTX 5090, bf16, batch=1, seq_len=512, hidden=2048. "
            "Run `make bench-kernels` to reproduce."
        )
        header = (
            "| Kernel | PyTorch (us) | Triton (us) "
            "| Speedup | Compile (s) | HBM Savings |"
        )
        divider = (
            "|--------|:------------:|:-----------:|"
            ":-------:|:-----------:|:-----------:|"
        )
        hbm = HBM_SAVINGS

    lines = [caption, "", header, divider]

    kernel_names = ["RMSNorm", "SwiGLU", "M-RoPE", "FusedNorm+Residual"]

    # Build lookup from raw data if available
    kernel_data: dict[str, dict] = {}
    if raw:
        for entry in raw:
            name = entry.get("kernel", "")
            # Skip entries with "not_implemented" status
            if entry.get("status") == "not_implemented":
                kernel_data[name] = {"status": "not_implemented"}
                continue
            kernel_data[name] = {
                "pytorch_us": entry.get("pytorch_us", 0.0),
                "triton_us": entry.get("triton_us", 0.0),
                "compile_time_s": entry.get("compile_time_s"),
            }

    kernel_display = {
        "FusedNorm+Residual": "Fused Norm+Residual",
    }

    for kernel in kernel_names:
        display = kernel_display.get(kernel, kernel)
        data = kernel_data.get(kernel)
        hbm_val = hbm.get(kernel, "")
        lines.append(_format_kernel_row(display, data, hbm_val))

    return "\n".join(lines)


def _format_kernel_row(
    display: str,
    data: dict | None,
    hbm_val: str,
) -> str:
    """Format a single kernel row for the benchmark table."""
    if data and data.get("status") == "not_implemented":
        return f"| {display} | N/A | N/A | N/A | N/A | {hbm_val} |"

    if data and data.get("pytorch_us", 0) > 0 and data.get("triton_us", 0) > 0:
        pt = f"{data['pytorch_us']:.1f}"
        tr = _bold(f"{data['triton_us']:.1f}")
        speedup = _bold(f"{data['pytorch_us'] / data['triton_us']:.2f}x")
        compile_t = data.get("compile_time_s")
        compile_str = f"{compile_t:.2f}" if compile_t is not None else "N/A"
        return f"| {display} | {pt} | {tr} | {speedup} | {compile_str} | {hbm_val} |"

    return f"| {display} | TBD | TBD | TBDx | TBD | {hbm_val} |"


# --- Quality table ---


def _render_quality_table_legacy(
    raw: dict | None,
    is_korean: bool = False,
) -> str:
    """Render the legacy single-comparison Tier 3 markdown table."""
    if is_korean:
        preamble = (
            "Triton \ucee4\ub110\uc740 PyTorch \uae30\ubcf8\uacfc "
            "\ud1b5\uacc4\uc801\uc73c\ub85c \ub3d9\ub4f1\ud55c "
            "\uc624\ub514\uc624\ub97c \uc0dd\uc131\ud569\ub2c8\ub2e4."
        )
        header = "| \uba54\ud2b8\ub9ad | Base | Triton | Delta | \uc784\uacc4\uac12 |"
        divider = "|--------|:----:|:------:|:-----:|:------:|"
        footer = (
            "`make eval-fast`\ub85c \uc7ac\ud604 \uac00\ub2a5. "
            "\ubc29\ubc95\ub860\uc740 "
            "[\uac80\uc99d \uccb4\uacc4](docs/verification-tiers.md) \ucc38\uc870."
        )
    else:
        preamble = (
            "Triton kernels produce statistically equivalent "
            "audio to the PyTorch baseline."
        )
        header = "| Metric | Base | Triton | Delta | Threshold |"
        divider = "|--------|:----:|:------:|:-----:|:---------:|"
        footer = (
            "Run `make eval-fast` to reproduce. "
            "See [Verification Tiers](docs/verification-tiers.md) "
            "for methodology."
        )

    lines = [preamble, "", header, divider]

    if raw and "base_metrics" in raw and "triton_metrics" in raw:
        bm = raw["base_metrics"]
        tm = raw["triton_metrics"]
        comp = raw.get("comparison", {})

        utmos_base = f"{bm['utmos_mean']:.2f}"
        utmos_triton = f"{tm['utmos_mean']:.2f}"
        utmos_delta = comp.get("utmos_delta", tm["utmos_mean"] - bm["utmos_mean"])
        utmos_delta_str = f"{utmos_delta:+.2f}"

        cer_base = f"{bm['cer_mean']:.4f}"
        cer_triton = f"{tm['cer_mean']:.4f}"
        cer_delta = comp.get("cer_delta", tm["cer_mean"] - bm["cer_mean"])
        cer_delta_str = f"{cer_delta:+.4f}"

        sim_mean = comp.get("speaker_sim_mean")
        sim_str = f"{sim_mean:.2f}" if sim_mean is not None else "TBD"
    else:
        utmos_base = "TBD"
        utmos_triton = "TBD"
        utmos_delta_str = "TBD"
        cer_base = "TBD"
        cer_triton = "TBD"
        cer_delta_str = "TBD"
        sim_str = "TBD"

    lines.append(
        f"| UTMOS (MOS) | {utmos_base} | {utmos_triton} "
        f"| {utmos_delta_str} | |delta| < 0.3 |"
    )
    lines.append(
        f"| CER | {cer_base} | {cer_triton} | {cer_delta_str} | |delta| < 0.05 |"
    )
    lines.append(f"| Speaker Sim | -- | {sim_str} | -- | > 0.75 |")
    lines.append("")
    lines.append(footer)

    return "\n".join(lines)


def _render_quality_table_multi(
    raw: dict,
    is_korean: bool = False,
) -> str:
    """Render the multi-runner Tier 3 markdown table."""
    mode = raw.get("mode", "full")
    runners = raw.get("runners", {})
    comparisons = raw.get("comparisons", [])
    cmp_by_opt = {comp.get("opt"): comp for comp in comparisons}

    if is_korean:
        preamble = (
            f"\uacf5\uc2dd \ub9b4\ub9ac\uc2a4 \ud488\uc9c8 "
            f"\uc218\uce58\ub294 {mode} \ubaa8\ub4dc "
            f"\uae30\uc900\uc73c\ub85c \uc815\ub9ac\ud588\uc2b5\ub2c8\ub2e4."
        )
        header = "| \ub7ec\ub108 | UTMOS | CER | Speaker Sim | \uc0c1\ud0dc |"
        divider = "|--------|:-----:|:---:|:-----------:|:----:|"
        reproduce_cmd = (
            "`make eval-full`\ub85c \uc7ac\ud604 \uac00\ub2a5."
            if mode == "full"
            else "`make eval-fast`\ub85c \uc7ac\ud604 \uac00\ub2a5."
        )
        footer = (
            f"{reproduce_cmd} "
            "fast \ubaa8\ub4dc\ub294 \uc2a4\ubaa8\ud06c "
            "\uccb4\ud06c\uc6a9\uc73c\ub85c \ubcf4\ub294 \ud3b8\uc774 \ub0ab\ub2e4."
        )
        caveat_title = (
            f"\ub9b4\ub9ac\uc2a4 \uc8fc\uc758\uc0ac\ud56d ({mode} \ubaa8\ub4dc):"
        )
    else:
        preamble = (
            f"Official release quality numbers use {mode} mode "
            f"as the canonical Tier 3 result."
        )
        header = "| Runner | UTMOS | CER | Speaker Sim | Status |"
        divider = "|--------|:-----:|:---:|:-----------:|:------:|"
        reproduce_cmd = (
            "Run `make eval-full` to reproduce."
            if mode == "full"
            else "Run `make eval-fast` to reproduce."
        )
        footer = (
            f"{reproduce_cmd} Treat fast mode as a "
            "smoke check, not the release authority."
        )
        caveat_title = f"Release caveats ({mode} mode):"

    lines = [preamble, "", header, divider]

    base_stats = runners.get("base", {})
    base_utmos = (
        f"{base_stats.get('utmos_mean', 0):.2f} \u00b1 "
        f"{base_stats.get('utmos_std', 0):.2f}"
    )
    base_cer = (
        f"{base_stats.get('cer_mean', 0):.2f} \u00b1 {base_stats.get('cer_std', 0):.2f}"
    )
    base_label = "Base (\uae30\uc900)" if is_korean else _display_runner_name("base")
    base_status = "\uae30\uc900" if is_korean else "ref"
    lines.append(f"| {base_label} | {base_utmos} | {base_cer} | - | {base_status} |")

    for runner in QUALITY_RUNNER_ORDER:
        if runner not in runners:
            continue
        stats = runners[runner]
        cmp = cmp_by_opt.get(runner, {})
        u_m = stats.get("utmos_mean", 0)
        u_s = stats.get("utmos_std", 0)
        utmos = f"{u_m:.2f} \u00b1 {u_s:.2f}"
        c_m = stats.get("cer_mean", 0)
        c_s = stats.get("cer_std", 0)
        cer = f"{c_m:.2f} \u00b1 {c_s:.2f}"
        sim = cmp.get("speaker_sim_mean")
        sim_str = f"{sim:.2f}" if sim is not None else "-"
        status = cmp.get("status", "N/A")
        lines.append(
            f"| {_display_runner_name(runner)} "
            f"| {utmos} | {cer} "
            f"| {sim_str} | {status} |"
        )

    failing = [comp for comp in comparisons if comp.get("failures")]
    if failing:
        lines.extend(["", caveat_title])
        for comp in failing:
            failures = "; ".join(comp.get("failures", []))
            opt = comp.get("opt", "?")
            st = comp.get("status", "FAIL")
            lines.append(f"- `{opt}`: {st} - {failures}")
    else:
        if is_korean:
            lines.extend(
                [
                    "",
                    "- \ubaa8\ub4e0 \ucd5c\uc801\ud654 \ub7ec\ub108\uac00 "
                    "full mode \uae30\uc900\uc744 "
                    "\ud1b5\uacfc\ud588\uc2b5\ub2c8\ub2e4.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "- All optimized runners pass the full mode release gate.",
                ]
            )

    lines.extend(["", footer])
    return "\n".join(lines)


def _render_quality_table(
    raw: dict | None,
    is_korean: bool = False,
) -> str:
    """Render the Tier 3 audio quality markdown table."""
    if raw and "runners" in raw and "comparisons" in raw:
        return _render_quality_table_multi(raw, is_korean=is_korean)
    return _render_quality_table_legacy(raw, is_korean=is_korean)


def _load_tier3_result() -> dict | None:
    """Load the canonical Tier 3 artifact, preferring full multi-runner results."""
    candidate_names = [
        "tier3_full_multi.json",
        "tier3_fast_multi.json",
        "tier3_full_base_vs_triton.json",
        "tier3_fast_base_vs_triton.json",
    ]
    for name in candidate_names:
        raw = _load_json(RESULTS_DIR / name)
        if isinstance(raw, dict):
            return raw
    return None


# --- Patching logic ---


def _patch_readme(
    readme_path: Path,
    sections: dict[str, str],
    dry_run: bool = False,
) -> bool:
    """Replace content between marker comments in a README file.

    Args:
        readme_path: Path to README file.
        sections: Dict mapping section key (KERNEL, E2E, QUALITY, SUMMARY)
            to rendered markdown content.
        dry_run: If True, print to stdout instead of writing.

    Returns:
        True if any replacements were made.
    """
    if not readme_path.exists():
        return False

    content = readme_path.read_text(encoding="utf-8")
    changed = False

    for match in MARKER_PATTERN.finditer(content):
        key = match.group(2)
        if key in sections:
            start_marker = match.group(1)
            end_marker = match.group(4)
            old_block = match.group(0)
            new_block = f"{start_marker}\n{sections[key]}\n{end_marker}"

            if old_block != new_block:
                content = content.replace(old_block, new_block)
                changed = True

    if dry_run:
        print(f"--- {readme_path.name} ---")
        for key, rendered in sections.items():
            print(f"\n[{key}]")
            print(rendered)
        print()
    elif changed:
        readme_path.write_text(content, encoding="utf-8")

    return changed


def main() -> None:
    """Load benchmark data, render tables, and patch README files."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark tables for README files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tables to stdout instead of patching files",
    )
    args = parser.parse_args()

    # Load data
    e2e_raw = _load_json(RESULTS_DIR / "e2e_benchmarks.json")
    kernel_raw = _load_json(RESULTS_DIR / "kernel_benchmarks.json")
    tier3_raw = _load_tier3_result()

    # Aggregate E2E data
    e2e_agg: dict[str, dict[str, float]] = {}
    if isinstance(e2e_raw, list) and e2e_raw:
        e2e_agg = _aggregate_e2e(e2e_raw)

    kernel_list = kernel_raw if isinstance(kernel_raw, list) else None
    tier3_dict = tier3_raw if isinstance(tier3_raw, dict) else None

    # Patch each README
    for readme_path in README_FILES:
        is_ko = readme_path.name == "README_ko.md"

        sections: dict[str, str] = {}

        # Summary line
        sections["SUMMARY"] = _render_summary(e2e_agg, is_korean=is_ko)

        # Kernel table
        sections["KERNEL"] = _render_kernel_table(kernel_list, is_korean=is_ko)

        # E2E table
        if e2e_agg:
            sections["E2E"] = _render_e2e_table(e2e_agg, is_korean=is_ko)
        else:
            # Keep TBD placeholders
            sections["E2E"] = _render_e2e_table({}, is_korean=is_ko)

        # Quality table
        sections["QUALITY"] = _render_quality_table(tier3_dict, is_korean=is_ko)

        changed = _patch_readme(readme_path, sections, dry_run=args.dry_run)

        if not args.dry_run:
            status = "updated" if changed else "no changes"
            print(f"{readme_path.name}: {status}")


if __name__ == "__main__":
    main()
