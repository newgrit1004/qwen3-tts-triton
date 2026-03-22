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
RUNNER_ORDER: list[str] = ["Base", "Triton", "Faster", "Hybrid"]

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


# --- E2E table ---


def _aggregate_e2e(
    raw: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate E2E results across languages per runner.

    Returns dict[runner_name, {latency_s, rtf, vram}].
    For latency: average of per-language means (converted ms->s).
    For RTF: average of per-language means.
    For VRAM: max across languages.
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

        aggregated[runner] = {
            "latency_s": sum(latencies) / len(latencies) / 1000.0,
            "rtf": sum(rtfs) / len(rtfs),
            "vram": max(vrams),
        }
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
            "| \ubaa8\ub4dc | \ub808\uc774\ud134\uc2dc (s) | RTF "
            "| \ud53c\ud06c VRAM (GB) | vs Base |"
        )
    else:
        caption = (
            "> RTX 5090, bf16, 2 texts (ko + en), "
            "3 warmup + 20 runs each. Run `make bench-e2e` to reproduce."
        )
        header = "| Mode | Latency (s) | RTF | Peak VRAM (GB) | vs Base |"

    divider = "|------|:-----------:|:---:|:--------------:|:-------:|"

    # Collect values for bold-best computation
    present = [r for r in RUNNER_ORDER if r in aggregated]
    lat_vals = {r: aggregated[r]["latency_s"] for r in present}
    rtf_vals = {r: aggregated[r]["rtf"] for r in present}
    vram_vals = {r: aggregated[r]["vram"] for r in present}
    speedup_vals: dict[str, float] = {}
    for r in present:
        if r == "Base":
            speedup_vals[r] = 1.0
        elif base_latency > 0:
            speedup_vals[r] = base_latency / aggregated[r]["latency_s"]
        else:
            speedup_vals[r] = 0.0

    # Bold the best in each column (exclude Base from speedup best)
    lat_fmt = _bold_best(lat_vals, "{:.2f}", minimize=True)
    rtf_fmt = _bold_best(rtf_vals, "{:.2f}", minimize=False)
    vram_fmt = _bold_best(vram_vals, "{:.2f}", minimize=True)
    spd_fmt = _bold_best(speedup_vals, "{:.2f}x", minimize=False, exclude={"Base"})

    lines = [caption, "", header, divider]

    for runner in RUNNER_ORDER:
        if runner not in aggregated:
            continue

        lat = lat_fmt[runner]
        rtf = rtf_fmt[runner]
        vram = vram_fmt[runner]
        vs_base = spd_fmt[runner]

        # stable-fast style labels
        if runner == "Hybrid":
            label = "__Hybrid (Faster+Triton)__"
        elif runner == "Base":
            label = "Base (PyTorch)"
        else:
            label = runner

        lines.append(f"| {label} | {lat} | {rtf} | {vram} | {vs_base} |")

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


def _render_quality_table(
    raw: dict | None,
    is_korean: bool = False,
) -> str:
    """Render the Tier 3 audio quality markdown table."""
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
    tier3_raw = _load_json(RESULTS_DIR / "tier3_fast_base_vs_triton.json")

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
