"""Tongue twister stress test: pronunciation robustness evaluation.

Compares Base vs Hybrid vs Hybrid+Patch on tongue twister sentences
to amplify pronunciation drift detected in PER analysis.

References:
    - Seed-TTS hard set (ByteDance 2024): 400 hard sentences with tongue twisters
    - MaskGCT (ICLR 2025): Appendix J tongue twister evaluation
    - EmergentTTS-Eval (NeurIPS 2025): Complex Pronunciation category

Usage:
    python -m benchmark.eval_tongue_twister --patch-range 0,24 --mode fast
    python -m benchmark.eval_tongue_twister \
        --patch-range 0,20 --patch-range 0,24 --mode full
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.eval_config import EVAL_CONFIG, TONGUE_TWISTER_SENTENCES
from benchmark.eval_quality import (
    _compute_verdict,
    _run_model_evaluation,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUTS_DIR = Path(__file__).parent / "output" / "eval_tongue_twister"


def _select_sentences(mode: str) -> list[dict[str, str]]:
    """Select tongue twister sentences based on mode."""
    sentences: list[dict[str, str]] = []
    for lang in EVAL_CONFIG["languages"]:
        lang_sents = TONGUE_TWISTER_SENTENCES.get(lang, [])
        if mode == "fast":
            sentences.extend(lang_sents[:5])
        else:
            sentences.extend(lang_sents)
    return sentences


def _make_base_factory() -> Any:
    """Factory for BaseRunner."""
    from qwen3_tts_triton.models.base_runner import BaseRunner

    return BaseRunner


def _make_hybrid_factory() -> Any:
    """Factory for TritonFasterRunner (full patching)."""
    from qwen3_tts_triton.models.triton_faster_runner import TritonFasterRunner

    return TritonFasterRunner


def _make_hybrid_patch_factory(patch_range: tuple[int, int]) -> Any:
    """Factory for TritonFasterRunner with partial patching."""
    from qwen3_tts_triton.models.triton_faster_runner import TritonFasterRunner

    def _factory() -> Any:
        return TritonFasterRunner(patch_range=patch_range)

    return _factory


def _print_summary(
    runners_stats: dict[str, dict[str, float]],
    comparisons: list[dict[str, Any]],
    mode: str,
    eval_time: float,
    num_sentences: int,
    runs_per_sentence: int,
    asr_model: str,
) -> None:
    """Print formatted evaluation summary."""
    sep = "=" * 72

    logger.info("")
    logger.info("TONGUE TWISTER STRESS TEST (%s mode)", mode)
    logger.info(sep)
    logger.info(
        "Sentences: %d | Runs/sent: %d | ASR: %s | Time: %.1fs",
        num_sentences,
        runs_per_sentence,
        asr_model,
        eval_time,
    )
    logger.info(sep)

    header = f"{'Runner':<20} {'UTMOS':<14} {'CER':<14} {'Spk Sim':<10} {'Status'}"
    logger.info(header)
    logger.info("-" * 72)

    # Reference row
    ref_stats = runners_stats.get("Base", {})
    u_m = ref_stats.get("utmos_mean", 0)
    u_s = ref_stats.get("utmos_std", 0)
    c_m = ref_stats.get("cer_mean", 0)
    c_s = ref_stats.get("cer_std", 0)
    logger.info(
        "%-20s %-14s %-14s %-10s %s",
        "Base",
        f"{u_m:.3f}\u00b1{u_s:.3f}",
        f"{c_m:.3f}\u00b1{c_s:.3f}",
        "-",
        "(ref)",
    )

    # Comparison rows
    cmp_map = {c["opt"]: c for c in comparisons}
    for name in runners_stats:
        if name == "Base":
            continue
        stats = runners_stats[name]
        cmp = cmp_map.get(name, {})
        logger.info(
            "%-20s %-14s %-14s %-10s %s",
            name,
            f"{stats.get('utmos_mean', 0):.3f}\u00b1{stats.get('utmos_std', 0):.3f}",
            f"{stats.get('cer_mean', 0):.3f}\u00b1{stats.get('cer_std', 0):.3f}",
            f"{cmp.get('speaker_sim_mean', 0):.3f}",
            cmp.get("status", "N/A"),
        )

    logger.info(sep)

    # Failures
    for cmp in comparisons:
        if cmp.get("failures"):
            logger.info("  %s failures:", cmp["opt"])
            for f in cmp["failures"]:
                logger.info("    - %s", f)

    # Key comparison
    logger.info("")
    logger.info("Key comparison (closer to Base = better):")
    for cmp in comparisons:
        logger.info(
            "  %s: UTMOS delta=%.4f, CER delta=%.4f, Speaker Sim=%.4f",
            cmp["opt"],
            cmp.get("utmos_delta", 0),
            cmp.get("cer_delta", 0),
            cmp.get("speaker_sim_mean", 0),
        )
    logger.info(sep)


def run_eval_tongue_twister(
    patch_ranges: list[tuple[int, int]],
    mode: str = "fast",
) -> dict[str, Any]:
    """Run tongue twister stress test: Base vs Hybrid vs Hybrid+Patch(N).

    Args:
        patch_ranges: List of (start, end) ranges to evaluate.
        mode: 'fast' or 'full'.

    Returns:
        Evaluation result dict.
    """
    tier3_cfg = EVAL_CONFIG["tier3"]
    warmup = int(EVAL_CONFIG["warmup_runs"])

    runs_per_sentence = (
        tier3_cfg["runs_per_sentence_fast"]
        if mode == "fast"
        else tier3_cfg["runs_per_sentence_full"]
    )
    asr_model = tier3_cfg["asr_model"]

    sentences = _select_sentences(mode)
    output_dir = OUTPUTS_DIR / f"twister_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build runner configs: Base + Hybrid + Hybrid+Patch(N)
    runner_configs: list[tuple[str, Any]] = [
        ("Base", _make_base_factory()),
        ("Hybrid", _make_hybrid_factory()),
    ]
    for pr in patch_ranges:
        name = f"Hybrid+P({pr[0]},{pr[1]})"
        runner_configs.append((name, _make_hybrid_patch_factory(pr)))

    runner_names = [name for name, _ in runner_configs]
    logger.info(
        "Tongue twister stress test (%s): Base vs %s (%d sentences, %d runs/sent)",
        mode,
        runner_names[1:],
        len(sentences),
        runs_per_sentence,
    )

    t_start = time.perf_counter()

    # Phase 1: Generate + evaluate all runners
    all_samples: dict[str, list[dict[str, Any]]] = {}
    for name, factory in runner_configs:
        logger.info("Generating with %s ...", name)
        all_samples[name] = _run_model_evaluation(
            factory,
            name,
            sentences,
            runs_per_sentence,
            output_dir,
            str(asr_model),
            warmup,
        )

    # Phase 2: Compare each non-Base runner against Base
    ref_results = all_samples["Base"]
    comparisons: list[dict[str, Any]] = []
    runners_stats: dict[str, dict[str, float]] = {}

    for name, samples in all_samples.items():
        utmos_vals = [r["utmos"] for r in samples]
        cer_vals = [r["cer"] for r in samples]
        runners_stats[name] = {
            "utmos_mean": round(float(np.mean(utmos_vals)), 4),
            "utmos_std": round(float(np.std(utmos_vals)), 4),
            "cer_mean": round(float(np.mean(cer_vals)), 4),
            "cer_std": round(float(np.std(cer_vals)), 4),
        }

        # Per-language breakdown
        languages = sorted({r["language"] for r in samples})
        per_lang: dict[str, dict[str, float]] = {}
        for lang in languages:
            lang_samples = [r for r in samples if r["language"] == lang]
            lang_utmos = [r["utmos"] for r in lang_samples]
            lang_cers = [r["cer"] for r in lang_samples]
            per_lang[lang] = {
                "utmos_mean": round(float(np.mean(lang_utmos)), 4),
                "utmos_std": round(float(np.std(lang_utmos)), 4),
                "cer_mean": round(float(np.mean(lang_cers)), 4),
                "cer_std": round(float(np.std(lang_cers)), 4),
                "n_samples": len(lang_samples),
            }
        runners_stats[name]["per_language"] = per_lang  # type: ignore[assignment]

        if name == "Base":
            continue

        verdict = _compute_verdict(ref_results, samples, "Base", name, mode)
        comparisons.append(
            {
                "ref": "Base",
                "opt": name,
                "status": verdict["status"],
                "utmos_delta": verdict["utmos_delta"],
                "cer_delta": verdict["cer_delta"],
                "speaker_sim_mean": verdict["speaker_sim_mean"],
                "failures": verdict["failures"],
            }
        )

    eval_time = round(time.perf_counter() - t_start, 2)

    _print_summary(
        runners_stats,
        comparisons,
        mode,
        eval_time,
        len(sentences),
        runs_per_sentence,
        str(asr_model),
    )

    result: dict[str, Any] = {
        "mode": mode,
        "sentence_set": "tongue_twister",
        "patch_ranges": [list(pr) for pr in patch_ranges],
        "num_sentences": len(sentences),
        "runs_per_sentence": runs_per_sentence,
        "asr_model": str(asr_model),
        "eval_time_s": eval_time,
        "runners": runners_stats,
        "comparisons": comparisons,
    }
    return result


def _parse_patch_range(value: str) -> tuple[int, int]:
    """Parse 'start,end' string into a (start, end) tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'start,end' format, got '{value}'")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            f"Must satisfy 0 <= start < end, got ({start}, {end})"
        )
    return (start, end)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tongue twister pronunciation stress test"
    )
    parser.add_argument(
        "--patch-range",
        type=_parse_patch_range,
        action="append",
        required=True,
        dest="patch_ranges",
        help="Layer range (repeatable), e.g. --patch-range 0,24.",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast (CI) or full (PR gate).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.",
    )
    args = parser.parse_args()

    result = run_eval_tongue_twister(
        patch_ranges=args.patch_ranges,
        mode=args.mode,
    )

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    range_tag = "_".join(f"{s}-{e}" for s, e in args.patch_ranges)
    out_path = args.output or str(
        RESULTS_DIR / f"eval_twister_{args.mode}_{range_tag}.json"
    )
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
