"""Shared pytest fixtures for qwen3-tts-triton tests."""

import pytest
import torch


@pytest.fixture
def device() -> str:
    """Return CUDA device if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Benchmark table rendering fixtures
#
# These are *synthetic* test inputs — NOT real benchmark numbers.
# Their only purpose is to feed the table rendering functions so we can
# assert structural properties (column presence, mode order, caveat text).
# Changing real benchmark results will never break these fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def seven_mode_e2e_aggregated() -> dict:
    """Synthetic E2E aggregated data covering all 7 inference modes.

    Used by ``test_generate_bench_tables.py`` to verify that the table
    renderer includes every mode in the correct release order and emits
    the expected column headers.  The numbers are arbitrary.
    """
    return {
        "Base": {
            "latency_s": 4.20,
            "rtf": 1.00,
            "vram": 4.00,
            "load_time_s": 10.0,
            "latency_ms_ko": 4000.0,
            "latency_ms_en": 4400.0,
            "rtf_ko": 1.00,
            "rtf_en": 0.95,
        },
        "Base+TQ": {
            "latency_s": 5.10,
            "rtf": 0.90,
            "vram": 3.90,
            "load_time_s": 8.5,
            "latency_ms_ko": 5100.0,
            "latency_ms_en": 5100.0,
            "rtf_ko": 0.90,
            "rtf_en": 0.89,
        },
        "Triton": {
            "latency_s": 3.20,
            "rtf": 1.30,
            "vram": 4.02,
            "load_time_s": 7.1,
            "latency_ms_ko": 3100.0,
            "latency_ms_en": 3300.0,
            "rtf_ko": 1.31,
            "rtf_en": 1.29,
        },
        "Triton+TQ": {
            "latency_s": 3.40,
            "rtf": 1.20,
            "vram": 3.92,
            "load_time_s": 7.2,
            "latency_ms_ko": 3500.0,
            "latency_ms_en": 3300.0,
            "rtf_ko": 1.18,
            "rtf_en": 1.22,
        },
        "Faster": {
            "latency_s": 1.20,
            "rtf": 3.50,
            "vram": 4.30,
            "load_time_s": 6.0,
            "latency_ms_ko": 1200.0,
            "latency_ms_en": 1200.0,
            "rtf_ko": 3.50,
            "rtf_en": 3.50,
        },
        "Hybrid": {
            "latency_s": 0.95,
            "rtf": 4.50,
            "vram": 4.28,
            "load_time_s": 6.5,
            "latency_ms_ko": 950.0,
            "latency_ms_en": 950.0,
            "rtf_ko": 4.50,
            "rtf_en": 4.50,
        },
        "Hybrid+TQ": {
            "latency_s": 0.90,
            "rtf": 4.60,
            "vram": 4.10,
            "load_time_s": 6.6,
            "latency_ms_ko": 900.0,
            "latency_ms_en": 900.0,
            "rtf_ko": 4.60,
            "rtf_en": 4.60,
        },
    }


@pytest.fixture
def seven_mode_quality_raw() -> dict:
    """Synthetic Tier 3 quality result with all 7 modes (full mode).

    Used to verify that the quality table renderer includes every runner,
    shows the correct mode label, and surfaces failure caveats.
    """
    return {
        "mode": "full",
        "status": "FAIL",
        "runners": {
            "base": {
                "utmos_mean": 3.20,
                "utmos_std": 0.40,
                "cer_mean": 0.10,
                "cer_std": 0.02,
            },
            "base+tq": {
                "utmos_mean": 3.18,
                "utmos_std": 0.42,
                "cer_mean": 0.11,
                "cer_std": 0.03,
            },
            "triton": {
                "utmos_mean": 3.22,
                "utmos_std": 0.35,
                "cer_mean": 0.11,
                "cer_std": 0.02,
            },
            "triton+tq": {
                "utmos_mean": 2.85,
                "utmos_std": 0.50,
                "cer_mean": 0.22,
                "cer_std": 0.08,
            },
            "faster": {
                "utmos_mean": 3.18,
                "utmos_std": 0.45,
                "cer_mean": 0.12,
                "cer_std": 0.03,
            },
            "hybrid": {
                "utmos_mean": 3.24,
                "utmos_std": 0.36,
                "cer_mean": 0.10,
                "cer_std": 0.02,
            },
            "hybrid+tq": {
                "utmos_mean": 3.21,
                "utmos_std": 0.38,
                "cer_mean": 0.10,
                "cer_std": 0.02,
            },
        },
        "comparisons": [
            {
                "ref": "base",
                "opt": "base+tq",
                "status": "PASS",
                "speaker_sim_mean": 0.79,
                "failures": [],
            },
            {
                "ref": "base",
                "opt": "triton",
                "status": "PASS",
                "speaker_sim_mean": 0.80,
                "failures": [],
            },
            {
                "ref": "base",
                "opt": "triton+tq",
                "status": "FAIL",
                "speaker_sim_mean": 0.71,
                "failures": ["Speaker sim 0.7100 < 0.75"],
            },
            {
                "ref": "base",
                "opt": "faster",
                "status": "PASS",
                "speaker_sim_mean": 0.78,
                "failures": [],
            },
            {
                "ref": "base",
                "opt": "hybrid",
                "status": "PASS",
                "speaker_sim_mean": 0.79,
                "failures": [],
            },
            {
                "ref": "base",
                "opt": "hybrid+tq",
                "status": "PASS",
                "speaker_sim_mean": 0.77,
                "failures": [],
            },
        ],
    }


@pytest.fixture
def fast_mode_quality_raw() -> dict:
    """Synthetic Tier 3 quality result with only 2 runners (fast mode).

    Intentionally minimal — tests that the renderer labels caveats as
    "fast mode" rather than "full mode" when the artifact was produced
    by ``make eval-fast``.
    """
    return {
        "mode": "fast",
        "status": "FAIL",
        "runners": {
            "base": {
                "utmos_mean": 3.10,
                "utmos_std": 0.30,
                "cer_mean": 0.10,
                "cer_std": 0.02,
            },
            "triton": {
                "utmos_mean": 3.05,
                "utmos_std": 0.32,
                "cer_mean": 0.11,
                "cer_std": 0.03,
            },
        },
        "comparisons": [
            {
                "ref": "base",
                "opt": "triton",
                "status": "FAIL",
                "speaker_sim_mean": 0.72,
                "failures": ["Speaker sim 0.7200 < 0.75"],
            },
        ],
    }
