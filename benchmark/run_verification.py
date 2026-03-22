"""3-Tier verification runner. Saves structured JSON for UI display.

Usage:
    python -m benchmark.run_verification              # All tiers
    python -m benchmark.run_verification --skip-tier3  # Tier 1+2 only
    python -m benchmark.run_verification --tier 1      # Tier 1 only
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
TIER2_ARTIFACT = RESULTS_DIR / "tier2_metrics.json"


# ────────────────────────────────────────────────────────────
# Tier 1: Kernel unit tests (via pytest subprocess)
# ────────────────────────────────────────────────────────────


def _parse_pytest_line(line: str) -> dict[str, Any] | None:
    """Parse a pytest verbose output line into test result dict."""
    # Pattern: tests/test_foo.py::test_name PASSED/FAILED [duration]
    match = re.match(
        r"^(tests/\S+::(\S+))\s+(PASSED|FAILED|SKIPPED|ERROR)",
        line.strip(),
    )
    if not match:
        return None
    return {
        "name": match.group(2),
        "fullname": match.group(1),
        "status": match.group(3),
    }


def run_tier1() -> dict[str, Any]:
    """Tier 1: Run pytest kernel tests (subprocess)."""
    t_start = time.perf_counter()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--ignore=tests/test_model_parity.py",
        "-v",
        "--tb=short",
        "-q",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    duration = round(time.perf_counter() - t_start, 2)

    # Parse test results from stdout
    tests: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parsed = _parse_pytest_line(line)
        if parsed:
            tests.append(parsed)

    passed = sum(1 for t in tests if t["status"] == "PASSED")
    failed = sum(1 for t in tests if t["status"] == "FAILED")
    skipped = sum(1 for t in tests if t["status"] == "SKIPPED")
    total = len(tests)

    status = "PASS" if failed == 0 and result.returncode == 0 else "FAIL"

    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "duration_s": duration,
        "tests": tests,
        "returncode": result.returncode,
    }


# ────────────────────────────────────────────────────────────
# Tier 2: Model parity (via pytest subprocess + JSON artifact)
# ────────────────────────────────────────────────────────────


def run_tier2() -> dict[str, Any]:
    """Tier 2: Run model parity tests via pytest subprocess.

    Delegates to tests/test_model_parity.py which writes a structured
    JSON artifact (tier2_metrics.json) with per-layer metrics, logits,
    and greedy divergence data. This ensures consistency between
    pytest and run_verification.py results.
    """
    t_start = time.perf_counter()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_model_parity.py",
        "-v",
        "--tb=short",
        "-q",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    duration = round(time.perf_counter() - t_start, 2)

    # Parse test results
    tests: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parsed = _parse_pytest_line(line)
        if parsed:
            tests.append(parsed)

    passed = sum(1 for t in tests if t["status"] == "PASSED")
    failed = sum(1 for t in tests if t["status"] == "FAILED")

    pytest_pass = failed == 0 and result.returncode == 0

    # Load structured metrics from the JSON artifact
    tier2_data = _load_tier2_artifact()

    if tier2_data:
        # Use artifact status, but override if pytest itself failed
        artifact_pass = tier2_data.get("status") == "PASS"
        status = "PASS" if (pytest_pass and artifact_pass) else "FAIL"
        result: dict[str, Any] = {
            "status": status,
            "duration_s": duration,
            "passed": passed,
            "failed": failed,
            "total": len(tests),
            "tests": tests,
        }
        # Multi-pair format (new): has "pairs" key
        if "pairs" in tier2_data:
            result["pairs"] = tier2_data["pairs"]
        else:
            # Single-pair format (legacy): layers/logits/greedy at top level
            result["layers"] = tier2_data.get("layers", {})
            result["logits"] = tier2_data.get("logits", {})
            result["greedy"] = tier2_data.get("greedy", {})
        return result

    # Fallback: artifact not found (tests may have been skipped)
    logger.warning("Tier 2 artifact not found at %s", TIER2_ARTIFACT)
    return {
        "status": "PASS" if pytest_pass else "FAIL",
        "duration_s": duration,
        "passed": passed,
        "failed": failed,
        "total": len(tests),
        "layers": {},
        "logits": {},
        "greedy": {},
        "tests": tests,
    }


def _load_tier2_artifact() -> dict[str, Any] | None:
    """Load tier2_metrics.json written by test_model_parity.py."""
    if not TIER2_ARTIFACT.exists():
        return None
    try:
        return json.loads(TIER2_ARTIFACT.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ────────────────────────────────────────────────────────────
# Tier 3: Load existing eval results (no execution)
# ────────────────────────────────────────────────────────────


def run_tier3_load() -> dict[str, Any] | None:
    """Tier 3: Load existing eval results JSON (does not run eval).

    Supports three formats:
    - Multi-runner (tier3_*_multi.json with ``comparisons`` list)
    - Single-pair (tier3_*.json with ``base_metrics``/``comparison``)
    - Legacy (eval_*.json with ``overall_verdict``)
    """
    if not RESULTS_DIR.exists():
        return None

    # Prefer new format (tier3_*), fall back to legacy (eval_*)
    candidates = sorted(RESULTS_DIR.glob("tier3_*.json"), reverse=True)
    if not candidates:
        candidates = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)

    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Multi-runner format: has comparisons list
        if "comparisons" in data and isinstance(data["comparisons"], list):
            return {
                "status": data.get("status", "UNKNOWN"),
                "source": candidate.name,
                "mode": data.get("mode"),
                "num_sentences": data.get("num_sentences"),
                "runners": data.get("runners", {}),
                "comparisons": data["comparisons"],
            }

        # Single-pair format: has base_metrics/comparison
        if "base_metrics" in data and "comparison" in data:
            return {
                "status": data.get("status", "UNKNOWN"),
                "source": candidate.name,
                "mode": data.get("mode"),
                "num_sentences": data.get("num_sentences"),
                "runners": {},
                "comparisons": [
                    {
                        "ref": data.get("ref_runner", "base"),
                        "opt": data.get("opt_runner", "triton"),
                        "status": data.get("status", "UNKNOWN"),
                        **data["comparison"],
                    }
                ],
            }

        # Legacy format: has overall_verdict
        return {
            "status": data.get("overall_verdict", "UNKNOWN"),
            "source": candidate.name,
            "mode": data.get("mode"),
            "num_sentences": data.get("num_sentences"),
            "runners": {},
            "comparisons": [
                {
                    "ref": "base",
                    "opt": "triton",
                    "status": data.get("overall_verdict", "UNKNOWN"),
                    "utmos_delta": data.get("utmos_delta_mean"),
                    "cer_delta": data.get("cer_delta_mean"),
                    "speaker_sim_mean": data.get("speaker_sim_mean"),
                }
            ],
        }

    return None


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entrypoint for 3-Tier verification."""
    parser = argparse.ArgumentParser(description="3-Tier verification runner")
    parser.add_argument(
        "--tier",
        type=str,
        default="1,2,3",
        help="Comma-separated tier numbers to run (default: 1,2,3)",
    )
    parser.add_argument(
        "--skip-tier3",
        action="store_true",
        help="Skip Tier 3 (equivalent to --tier 1,2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: benchmark/results/verification_report.json)",
    )
    args = parser.parse_args()

    tiers = {int(t.strip()) for t in args.tier.split(",")}
    if args.skip_tier3:
        tiers.discard(3)

    report: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Tier 1
    if 1 in tiers:
        logger.info("=" * 50)
        logger.info("Running Tier 1: Kernel unit tests...")
        logger.info("=" * 50)
        report["tier1"] = run_tier1()
        logger.info(
            "Tier 1: %s (%d/%d passed, %.1fs)",
            report["tier1"]["status"],
            report["tier1"]["passed"],
            report["tier1"]["total"],
            report["tier1"]["duration_s"],
        )

    # Tier 2
    if 2 in tiers:
        logger.info("=" * 50)
        logger.info("Running Tier 2: Model parity tests...")
        logger.info("=" * 50)
        try:
            report["tier2"] = run_tier2()
            logger.info(
                "Tier 2: %s (%.1fs)",
                report["tier2"]["status"],
                report["tier2"]["duration_s"],
            )
        except Exception as exc:
            logger.error("Tier 2 failed: %s", exc)
            report["tier2"] = {
                "status": "ERROR",
                "error": str(exc),
                "duration_s": 0,
            }

    # Tier 3
    if 3 in tiers:
        logger.info("=" * 50)
        logger.info("Loading Tier 3: E2E quality results...")
        logger.info("=" * 50)
        tier3 = run_tier3_load()
        if tier3:
            report["tier3"] = tier3
            logger.info("Tier 3: %s (source: %s)", tier3["status"], tier3["source"])
        else:
            logger.info("Tier 3: No eval results found. Run 'make eval-fast' first.")
            report["tier3"] = None

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / "verification_report.json")
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to %s", out_path)

    # Print summary
    _print_summary(report)


def _print_summary(report: dict[str, Any]) -> None:
    """Log a concise pass/fail summary for each tier in the report.

    Args:
        report: Verification report dict with optional 'tier1', 'tier2',
            and 'tier3' keys, each containing a 'status' field.
    """
    logger.info("=" * 50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 50)

    # Tier 1
    t1 = report.get("tier1")
    if t1 is None:
        logger.info("  TIER1: SKIPPED")
    else:
        logger.info("  TIER1 (Kernel Tests): %s", t1.get("status", "UNKNOWN"))

    # Tier 2 — show per-pair details if available
    t2 = report.get("tier2")
    if t2 is None:
        logger.info("  TIER2: SKIPPED")
    elif "pairs" in t2:
        logger.info("  TIER2 (Model Parity): %s", t2.get("status", "UNKNOWN"))
        for pair_name, pair_data in t2["pairs"].items():
            label = pair_name.replace("_", " ")
            logger.info("    %s: %s", label, pair_data.get("status", "?"))
    else:
        logger.info("  TIER2 (Model Parity): %s", t2.get("status", "UNKNOWN"))

    # Tier 3 — show per-comparison details if available
    t3 = report.get("tier3")
    if t3 is None:
        logger.info("  TIER3: SKIPPED")
    elif "comparisons" in t3:
        logger.info("  TIER3 (E2E Quality): %s", t3.get("status", "UNKNOWN"))
        for comp in t3["comparisons"]:
            ref = comp.get("ref", "base")
            opt = comp.get("opt", "?")
            logger.info("    %s vs %s: %s", ref, opt, comp.get("status", "?"))
    else:
        logger.info("  TIER3 (E2E Quality): %s", t3.get("status", "UNKNOWN"))

    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
