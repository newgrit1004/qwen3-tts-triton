"""Regression tests for public release artifacts and messaging."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


def test_verification_artifact_uses_current_tier1_total():
    """The canonical verification artifact should reflect the current test set."""
    report = json.loads(_read("benchmark/results/verification_report.json"))

    assert report["tier1"]["status"] == "PASS"
    assert report["tier1"]["total"] == 197
    assert report["tier1"]["passed"] == 197


def test_release_docs_match_current_tier1_total_and_test_layout():
    """Release-facing docs should match the current verification/test surface."""
    readme_ko = _read("README_ko.md")
    bench_en = _read("docs/benchmark_results_en.md")
    bench_ko = _read("docs/benchmark_results_ko.md")
    test_doc = _read("tests/TEST.md")

    assert "90 tests" not in readme_ko
    assert "`make test` (197 tests)" in readme_ko
    assert "make test          # Tier 1: 197 tests" in readme_ko

    assert "PASS (197 tests)" in bench_en
    assert "| **Total** | **197** | **PASS** |" in bench_en
    assert "PASS (197개 테스트)" in bench_ko
    assert "| **합계** | **197** | **PASS** |" in bench_ko

    for snippet in (
        "test_fused_dequant.py",
        "test_turboquant.py",
        "test_partial_patching.py",
        "test_generate_bench_tables.py",
        "test_makefile_targets.py",
        "test_eval_quality_defaults.py",
    ):
        assert snippet in test_doc


def test_readmes_use_mode_specific_turboquant_release_language():
    """TurboQuant copy should reflect the current per-mode release caveats."""
    readme_en = _read("README.md")
    readme_ko = _read("README_ko.md")

    assert "minimal quality impact" not in readme_en
    assert "maintaining comparable speed and quality" not in readme_en
    assert "Hybrid+TQ is the current release-grade TurboQuant path." in readme_en
    assert (
        "Base+TQ and Triton+TQ remain experimental"  # noqa: E501
        " until they pass the full Tier 3 gate." in readme_en
    )

    assert "Hybrid+TQ가 현재 공개 기준의 TurboQuant 경로" in readme_ko
    assert (
        "Base+TQ와 Triton+TQ는 full Tier 3 게이트를 통과하기 전까지 experimental로 보는 편이 맞습니다."  # noqa: E501
        in readme_ko
    )
