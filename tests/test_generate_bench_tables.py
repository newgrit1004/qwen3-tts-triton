"""Tests for README benchmark table generation.

These tests verify that ``scripts/generate_bench_tables.py`` renders
release-facing tables correctly.  They do NOT validate actual benchmark
numbers — the input data comes from synthetic conftest fixtures whose
values are arbitrary.  The assertions check **structural** properties:

- All 7 inference modes appear in the correct release order.
- Column headers (Load Time, Latency, RTF, …) are present.
- Tier 3 artifact loading prefers ``full`` over ``fast`` mode.
- Quality caveats reference the correct mode label.

If real benchmark numbers change, these tests will NOT break.
They only break when the table schema or mode set changes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    """Load the standalone bench table generator script as a module."""
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "generate_bench_tables.py"
    )
    spec = importlib.util.spec_from_file_location("generate_bench_tables", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_e2e_table_includes_all_seven_modes_in_release_order(
    seven_mode_e2e_aggregated,
):
    """7-mode release table should include all runners in stable order.

    Feeds synthetic aggregated data (from conftest fixture) into the
    renderer and checks that the output contains the right columns and
    all mode labels in the expected sequence.
    """
    module = _load_module()

    rendered = module._render_e2e_table(seven_mode_e2e_aggregated)

    # Column headers must be present
    assert "Load Time" in rendered
    assert "Latency (ko)" in rendered
    assert "Latency (en)" in rendered
    assert "RTF (ko)" in rendered
    assert "RTF (en)" in rendered
    assert "Peak VRAM" in rendered
    # Spot-check a few formatted values from the fixture
    assert "10.0s" in rendered
    assert "4,000 ms" in rendered
    assert "4.7x" in rendered
    assert "[0, 24)" in rendered

    # Mode labels must appear in the canonical release order
    expected_order = [
        "Base (PyTorch)",
        "Base+TQ",
        "Triton",
        "Triton+TQ",
        "Faster",
        "__Hybrid (Faster+Triton)__",
        "Hybrid+TQ",
    ]

    cursor = -1
    for label in expected_order:
        idx = rendered.find(label)
        assert idx > cursor, f"{label} missing or out of order:\n{rendered}"
        cursor = idx


def test_load_tier3_result_prefers_full_multi_runner_artifact(tmp_path):
    """Release docs should prefer full multi-runner Tier 3 results.

    When both ``tier3_fast_multi.json`` and ``tier3_full_multi.json``
    exist, the loader must pick ``full`` — it has Mann-Whitney stats
    that ``fast`` mode omits.
    """
    module = _load_module()
    module.RESULTS_DIR = tmp_path

    (tmp_path / "tier3_fast_multi.json").write_text('{"mode":"fast","status":"FAIL"}')
    (tmp_path / "tier3_full_multi.json").write_text('{"mode":"full","status":"PASS"}')

    loaded = module._load_tier3_result()

    assert loaded is not None
    assert loaded["mode"] == "full"
    assert loaded["status"] == "PASS"


def test_render_quality_table_supports_full_multi_runner_release_format(
    seven_mode_quality_raw,
):
    """Quality section should render all optimized runners plus caveats.

    Uses the 7-mode quality fixture (full mode, with one FAIL runner)
    to verify that every runner appears in the output and failure
    details are surfaced.
    """
    module = _load_module()

    rendered = module._render_quality_table(seven_mode_quality_raw)

    for runner in ("base+tq", "triton", "triton+tq", "faster", "hybrid", "hybrid+tq"):
        assert runner in rendered

    assert "full mode" in rendered.lower()
    assert "Speaker sim 0.7100 < 0.75" in rendered


def test_render_quality_table_uses_actual_mode_in_release_caveat_title(
    fast_mode_quality_raw,
):
    """Fallback fast-mode rendering should not claim full-mode caveats.

    The fast-mode fixture has only 2 runners.  The caveat section must
    say "fast mode", not "full mode", so users know the statistical
    evidence is weaker.
    """
    module = _load_module()

    rendered = module._render_quality_table(fast_mode_quality_raw)

    assert "Release caveats (fast mode):" in rendered
    assert "Release caveats (full mode):" not in rendered
    assert "Run `make eval-fast` to reproduce." in rendered
