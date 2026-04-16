"""Regression tests for release-oriented Makefile targets."""

from __future__ import annotations

import re
from pathlib import Path

MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"


def _target_definition(target: str) -> tuple[str, str]:
    """Return the dependency list and recipe block for a Make target."""
    text = MAKEFILE.read_text()
    pattern = re.compile(
        rf"(?ms)^\.PHONY:\s*{re.escape(target)}\n"
        rf"^{re.escape(target)}:(?P<deps>[^\n]*)\n"
        rf"(?P<body>(?:^\t.*\n)*)",
    )
    match = pattern.search(text)
    assert match is not None, f"target {target!r} not found"
    return match.group("deps").strip(), match.group("body")


def test_verify_all_runs_full_tier3_before_report_generation():
    """`make verify-all` should execute full Tier 3 before loading the report."""
    deps, body = _target_definition("verify-all")

    assert deps.split() == ["eval-full", "verify"]
    assert body == ""


def test_verify_remains_report_generation_entrypoint():
    """`make verify` should still build the consolidated verification report."""
    deps, body = _target_definition("verify")

    assert deps == ""
    assert "benchmark.run_verification" in body


def test_makefile_version_tracks_v020_release_branch():
    """Release metadata in Makefile should match the v0.2.0 branch."""
    text = MAKEFILE.read_text()

    assert re.search(r"^VERSION=0\.2\.0$", text, re.MULTILINE)
