"""TTS quality evaluation: independent benchmark distribution comparison.

Evaluates Base vs Triton by generating audio independently with each model,
computing per-sample metrics (CER, UTMOS), then comparing distributions.
No pair-level waveform comparison (PESQ/STOI/MCD) — inappropriate for
stochastic autoregressive TTS where identical models produce different waveforms.

Methodology follows vLLM/TensorRT-LLM pattern:
- Each model generates independently on the same test set
- Task-level metrics (CER, UTMOS) are computed per sample
- Distribution equivalence is verified via mean delta + Mann-Whitney U test

Usage:
    # Fast CI evaluation (~5 min, whisper-small, 1 run/sentence)
    python -m benchmark.eval_quality --mode fast

    # Full PR gate evaluation (~30 min, whisper-large-v3, 3 runs/sentence)
    python -m benchmark.eval_quality --mode full
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from benchmark.eval_config import EVAL_CONFIG, EVAL_SENTENCES

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUTS_DIR = Path(__file__).parent / "output" / "eval"

# Cache for expensive model loads
_whisper_model_cache: dict[str, Any] = {}
_utmos_predictor_cache: list[Any] = []
_voice_encoder_cache: list[Any] = []


# ────────────────────────────────────────────────────────────
# Model caching helpers
# ────────────────────────────────────────────────────────────


def _get_whisper_model(model_size: str) -> Any:
    """Load and cache Whisper ASR model."""
    if model_size not in _whisper_model_cache:
        import whisper

        _whisper_model_cache[model_size] = whisper.load_model(model_size)
    return _whisper_model_cache[model_size]


def _get_utmos_predictor() -> Any:
    """Load and cache UTMOS predictor."""
    if not _utmos_predictor_cache:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        _utmos_predictor_cache.append(predictor)
    return _utmos_predictor_cache[0]


def _get_voice_encoder() -> Any:
    """Load and cache Resemblyzer voice encoder."""
    if not _voice_encoder_cache:
        from resemblyzer import VoiceEncoder

        _voice_encoder_cache.append(VoiceEncoder())
    return _voice_encoder_cache[0]


# ────────────────────────────────────────────────────────────
# Per-sample metric computation
# ────────────────────────────────────────────────────────────


def compute_cer(
    wav_path: str | Path,
    ground_truth: str,
    asr_model: str = "small",
) -> dict[str, Any]:
    """Compute CER for a single WAV against ground truth text."""
    from jiwer import cer

    model = _get_whisper_model(asr_model)
    result = model.transcribe(str(wav_path), language=None)
    transcript = result["text"].strip()

    cer_value = cer(ground_truth, transcript)
    return {
        "cer": float(cer_value),
        "transcript": transcript,
    }


def compute_utmos(wav_path: str | Path) -> float:
    """Predict MOS score using UTMOS."""
    import librosa

    predictor = _get_utmos_predictor()
    wave, sr = librosa.load(str(wav_path), sr=None, mono=True)
    tensor = torch.from_numpy(wave).unsqueeze(0)
    score = predictor(tensor, sr)
    return float(score.mean().item())


def compute_speaker_similarity(wav_a: str | Path, wav_b: str | Path) -> float:
    """Compute speaker embedding cosine similarity between two WAVs."""
    from resemblyzer import preprocess_wav

    encoder = _get_voice_encoder()
    audio_a = preprocess_wav(Path(wav_a))
    audio_b = preprocess_wav(Path(wav_b))

    embed_a = encoder.embed_utterance(audio_a)
    embed_b = encoder.embed_utterance(audio_b)

    similarity = float(
        np.dot(embed_a, embed_b) / (np.linalg.norm(embed_a) * np.linalg.norm(embed_b))
    )
    return similarity


# ────────────────────────────────────────────────────────────
# Independent generation + evaluation
# ────────────────────────────────────────────────────────────


def generate_and_evaluate(
    runner: Any,
    sentence: dict[str, str],
    output_dir: Path,
    runner_name: str,
    idx: int,
    run: int,
    asr_model: str = "small",
) -> dict[str, Any]:
    """Generate audio with one model and evaluate independently.

    Args:
        runner: Loaded TTS runner instance.
        sentence: Dict with 'text' and 'language' keys.
        output_dir: Directory to save WAV files.
        runner_name: Name for file naming (e.g., 'base', 'triton').
        idx: Sentence index.
        run: Run number (for multi-run modes).
        asr_model: Whisper model size for CER computation.

    Returns:
        Dict with wav_path, cer, utmos, and metadata.
    """
    output = runner.generate(text=sentence["text"], language=sentence["language"])
    audio = output.get("audio")
    sr = output.get("sample_rate", 24000)

    wav_path = output_dir / f"{runner_name}_{idx:03d}_r{run}.wav"
    sf.write(str(wav_path), audio, sr)

    cer_result = compute_cer(wav_path, sentence["text"], asr_model)
    utmos_score = compute_utmos(wav_path)

    return {
        "wav_path": str(wav_path),
        "sentence_idx": idx,
        "run": run,
        "text": sentence["text"][:60],
        "language": sentence["language"],
        "cer": cer_result["cer"],
        "transcript": cer_result["transcript"],
        "utmos": utmos_score,
    }


def _run_model_evaluation(
    runner_cls: type,
    runner_name: str,
    sentences: list[dict[str, str]],
    runs_per_sentence: int,
    output_dir: Path,
    asr_model: str,
    warmup_runs: int,
) -> list[dict[str, Any]]:
    """Generate and evaluate all sentences with one model.

    Args:
        runner_cls: Runner class to instantiate.
        runner_name: Name identifier for the runner.
        sentences: List of sentence dicts.
        runs_per_sentence: Number of generations per sentence.
        output_dir: Output directory for WAV files.
        asr_model: Whisper model size.
        warmup_runs: Number of warmup generations.

    Returns:
        List of per-sample result dicts.
    """
    runner = runner_cls()
    results: list[dict[str, Any]] = []

    try:
        runner.load_model()

        # Warmup
        for i in range(warmup_runs):
            logger.info("[%s] Warmup %d/%d", runner_name, i + 1, warmup_runs)
            runner.generate(
                text=sentences[0]["text"], language=sentences[0]["language"]
            )

        # Generate and evaluate
        total = len(sentences) * runs_per_sentence
        count = 0
        for run in range(runs_per_sentence):
            for idx, sent in enumerate(sentences):
                count += 1
                logger.info(
                    "[%s] Generating %d/%d (sent %d, run %d)",
                    runner_name,
                    count,
                    total,
                    idx,
                    run,
                )
                result = generate_and_evaluate(
                    runner,
                    sent,
                    output_dir,
                    runner_name,
                    idx,
                    run,
                    asr_model,
                )
                results.append(result)
    finally:
        runner.unload_model()

    return results


# ────────────────────────────────────────────────────────────
# Distribution comparison
# ────────────────────────────────────────────────────────────


def _compute_distribution_stats(
    values: list[float],
) -> dict[str, float]:
    """Compute mean/std/min/max for a list of values."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _compute_verdict(
    ref_results: list[dict[str, Any]],
    opt_results: list[dict[str, Any]],
    ref_name: str,
    opt_name: str,
    mode: str,
) -> dict[str, Any]:
    """Compare distributions and produce PASS/FAIL verdict.

    Args:
        ref_results: Per-sample results from reference model.
        opt_results: Per-sample results from optimized model.
        ref_name: Name of the reference runner.
        opt_name: Name of the optimized runner.
        mode: 'fast' or 'full' (affects Mann-Whitney requirement).

    Returns:
        Dict with ref_metrics, opt_metrics, comparison, and verdict.
    """
    tier3_cfg = EVAL_CONFIG["tier3"]

    # Extract metric arrays
    ref_utmos = [r["utmos"] for r in ref_results]
    opt_utmos = [r["utmos"] for r in opt_results]
    ref_cers = [r["cer"] for r in ref_results]
    opt_cers = [r["cer"] for r in opt_results]

    # Distribution stats
    ref_metrics = {
        "utmos": _compute_distribution_stats(ref_utmos),
        "cer": _compute_distribution_stats(ref_cers),
    }
    opt_metrics = {
        "utmos": _compute_distribution_stats(opt_utmos),
        "cer": _compute_distribution_stats(opt_cers),
    }

    # Deltas
    utmos_delta = abs(ref_metrics["utmos"]["mean"] - opt_metrics["utmos"]["mean"])
    cer_delta = abs(ref_metrics["cer"]["mean"] - opt_metrics["cer"]["mean"])

    # Speaker similarity (compare matched sentence pairs)
    speaker_sims = _compute_speaker_similarities(ref_results, opt_results)
    speaker_sim_mean = float(np.mean(speaker_sims)) if speaker_sims else 0.0

    # Mann-Whitney U test (full mode only)
    mann_whitney_result = None
    if mode == "full" and len(ref_utmos) >= 3:
        from scipy.stats import mannwhitneyu

        stat, p_value = mannwhitneyu(ref_utmos, opt_utmos, alternative="two-sided")
        mann_whitney_result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "equivalent": float(p_value) > tier3_cfg["mann_whitney_alpha"],
        }

    comparison = {
        "utmos_delta": round(utmos_delta, 4),
        "cer_delta": round(cer_delta, 4),
        "speaker_sim_mean": round(speaker_sim_mean, 4),
    }
    if mann_whitney_result:
        comparison["mann_whitney_utmos"] = mann_whitney_result

    # Verdict
    failures: list[str] = []

    if utmos_delta > tier3_cfg["utmos_delta_max"]:
        failures.append(
            f"UTMOS delta {utmos_delta:.4f} > {tier3_cfg['utmos_delta_max']}"
        )
    if ref_metrics["utmos"]["mean"] < tier3_cfg["utmos_floor"]:
        failures.append(
            f"{ref_name} UTMOS mean {ref_metrics['utmos']['mean']:.3f}"
            f" < {tier3_cfg['utmos_floor']}"
        )
    if opt_metrics["utmos"]["mean"] < tier3_cfg["utmos_floor"]:
        failures.append(
            f"{opt_name} UTMOS mean {opt_metrics['utmos']['mean']:.3f}"
            f" < {tier3_cfg['utmos_floor']}"
        )
    if cer_delta > tier3_cfg["cer_delta_max"]:
        failures.append(f"CER delta {cer_delta:.4f} > {tier3_cfg['cer_delta_max']}")
    if speaker_sim_mean < tier3_cfg["speaker_sim_min"]:
        failures.append(
            f"Speaker sim {speaker_sim_mean:.4f} < {tier3_cfg['speaker_sim_min']}"
        )
    if mode == "full" and mann_whitney_result:
        if not mann_whitney_result["equivalent"]:
            failures.append(
                f"Mann-Whitney p={mann_whitney_result['p_value']:.4f}"
                f" < {tier3_cfg['mann_whitney_alpha']}"
            )

    for f in failures:
        logger.warning("FAIL: %s", f)

    status = "FAIL" if failures else "PASS"

    return {
        "ref": ref_name,
        "opt": opt_name,
        "status": status,
        "failures": failures,
        "ref_metrics": {
            "utmos_mean": round(ref_metrics["utmos"]["mean"], 4),
            "utmos_std": round(ref_metrics["utmos"]["std"], 4),
            "cer_mean": round(ref_metrics["cer"]["mean"], 4),
            "cer_std": round(ref_metrics["cer"]["std"], 4),
        },
        "opt_metrics": {
            "utmos_mean": round(opt_metrics["utmos"]["mean"], 4),
            "utmos_std": round(opt_metrics["utmos"]["std"], 4),
            "cer_mean": round(opt_metrics["cer"]["mean"], 4),
            "cer_std": round(opt_metrics["cer"]["std"], 4),
        },
        "comparison": comparison,
        "utmos_delta": round(utmos_delta, 4),
        "cer_delta": round(cer_delta, 4),
        "speaker_sim_mean": round(speaker_sim_mean, 4),
    }


def _compute_speaker_similarities(
    base_results: list[dict[str, Any]],
    triton_results: list[dict[str, Any]],
) -> list[float]:
    """Compute speaker similarity for matched sentence pairs (run 0 only).

    Compares base run-0 vs triton run-0 for each sentence index.
    """
    # Group by sentence_idx, take run=0 only
    base_by_idx: dict[int, str] = {}
    for r in base_results:
        if r["run"] == 0:
            base_by_idx[r["sentence_idx"]] = r["wav_path"]

    triton_by_idx: dict[int, str] = {}
    for r in triton_results:
        if r["run"] == 0:
            triton_by_idx[r["sentence_idx"]] = r["wav_path"]

    sims: list[float] = []
    for idx in sorted(base_by_idx.keys()):
        if idx not in triton_by_idx:
            continue
        sim = compute_speaker_similarity(base_by_idx[idx], triton_by_idx[idx])
        sims.append(sim)
        logger.info(
            "Speaker sim sentence %d: %.4f",
            idx,
            sim,
        )

    return sims


# ────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ────────────────────────────────────────────────────────────


def run_tier3(
    mode: str = "fast",
    ref_runner: str = "base",
    opt_runners: list[str] | None = None,
) -> dict[str, Any]:
    """Run Tier 3 quality evaluation: independent distribution comparison.

    Generates reference samples once, then compares each opt runner against
    the reference independently.

    Args:
        mode: 'fast' (1 run, whisper-small) or 'full' (3 runs, whisper-large-v3).
        ref_runner: Reference runner name (default: 'base').
        opt_runners: Optimized runner names (default: all available runners).

    Returns:
        Complete evaluation result dict with per-runner comparisons.
    """
    from qwen3_tts_triton.models import get_runner_class

    if opt_runners is None:
        opt_runners = ["triton", "faster", "hybrid"]

    tier3_cfg = EVAL_CONFIG["tier3"]
    warmup = int(EVAL_CONFIG["warmup_runs"])

    runs_per_sentence = (
        tier3_cfg["runs_per_sentence_fast"]
        if mode == "fast"
        else tier3_cfg["runs_per_sentence_full"]
    )
    asr_model = (
        tier3_cfg["asr_model_fast"] if mode == "fast" else tier3_cfg["asr_model_full"]
    )

    sentences = _select_sentences(mode)
    output_dir = OUTPUTS_DIR / f"multi_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Tier 3 %s: %s vs %s (%d sentences, %d runs/sent, ASR=%s)",
        mode,
        ref_runner,
        opt_runners,
        len(sentences),
        runs_per_sentence,
        asr_model,
    )

    t_start = time.perf_counter()

    # Phase 1: Generate reference samples once
    logger.info("Phase 1: Generating with %s (reference)...", ref_runner)
    ref_results = _run_model_evaluation(
        get_runner_class(ref_runner),
        ref_runner,
        sentences,
        runs_per_sentence,
        output_dir,
        str(asr_model),
        warmup,
    )

    # Phase 1b: Generate opt runner samples
    opt_samples: dict[str, list[dict[str, Any]]] = {}
    for opt_name in opt_runners:
        logger.info("Phase 1: Generating with %s...", opt_name)
        opt_samples[opt_name] = _run_model_evaluation(
            get_runner_class(opt_name),
            opt_name,
            sentences,
            runs_per_sentence,
            output_dir,
            str(asr_model),
            warmup,
        )

    # Phase 2: Distribution comparison + verdict per opt runner
    logger.info("Phase 2: Comparing distributions...")
    comparisons: list[dict[str, Any]] = []
    runners_stats: dict[str, dict[str, float]] = {}

    ref_utmos = [r["utmos"] for r in ref_results]
    ref_cers = [r["cer"] for r in ref_results]
    runners_stats[ref_runner] = {
        "utmos_mean": round(float(np.mean(ref_utmos)), 4),
        "utmos_std": round(float(np.std(ref_utmos)), 4),
        "cer_mean": round(float(np.mean(ref_cers)), 4),
        "cer_std": round(float(np.std(ref_cers)), 4),
    }

    for opt_name, opt_results in opt_samples.items():
        verdict = _compute_verdict(ref_results, opt_results, ref_runner, opt_name, mode)
        comparisons.append(
            {
                "ref": ref_runner,
                "opt": opt_name,
                "status": verdict["status"],
                "utmos_delta": verdict["utmos_delta"],
                "cer_delta": verdict["cer_delta"],
                "speaker_sim_mean": verdict["speaker_sim_mean"],
                "failures": verdict["failures"],
            }
        )
        opt_utmos = [r["utmos"] for r in opt_results]
        opt_cers = [r["cer"] for r in opt_results]
        runners_stats[opt_name] = {
            "utmos_mean": round(float(np.mean(opt_utmos)), 4),
            "utmos_std": round(float(np.std(opt_utmos)), 4),
            "cer_mean": round(float(np.mean(opt_cers)), 4),
            "cer_std": round(float(np.std(opt_cers)), 4),
        }

    eval_time = round(time.perf_counter() - t_start, 2)
    overall_status = (
        "PASS" if all(c["status"] == "PASS" for c in comparisons) else "FAIL"
    )

    result = {
        "status": overall_status,
        "mode": mode,
        "ref_runner": ref_runner,
        "opt_runners": opt_runners,
        "num_sentences": len(sentences),
        "runs_per_sentence": runs_per_sentence,
        "asr_model": str(asr_model),
        "eval_time_s": eval_time,
        "runners": runners_stats,
        "comparisons": comparisons,
        "ref_samples": ref_results,
        "opt_samples": opt_samples,
    }

    _print_summary(result)
    return result


def _select_sentences(mode: str) -> list[dict[str, str]]:
    """Select evaluation sentences based on mode.

    In fast mode, takes the first 5 sentences per language.
    In full mode, uses all sentences per language.

    Args:
        mode: Either 'fast' (CI subset) or 'full' (complete set).

    Returns:
        List of sentence dicts with 'text' and 'language' keys.
    """
    sentences: list[dict[str, str]] = []
    for lang in EVAL_CONFIG["languages"]:
        lang_sents = EVAL_SENTENCES.get(lang, [])
        if mode == "fast":
            sentences.extend(lang_sents[:5])
        else:
            sentences.extend(lang_sents)
    return sentences


def _print_summary(result: dict[str, Any]) -> None:
    """Log a formatted summary of Tier 3 evaluation results.

    Prints a table with one row per runner: UTMOS, CER, Speaker Sim, Status.

    Args:
        result: Complete evaluation result dict as returned by run_tier3.
    """
    mode = result["mode"]
    ref_runner = result["ref_runner"]
    opt_runners = result.get("opt_runners", [])
    runners_stats = result.get("runners", {})
    comparisons = result.get("comparisons", [])

    sep = "\u2550" * 62
    logger.info("")
    logger.info("TIER 3 EVALUATION (%s mode)", mode)
    logger.info(sep)
    logger.info(
        "Sentences: %d | Runs/sent: %d | ASR: %s | Time: %.1fs",
        result["num_sentences"],
        result["runs_per_sentence"],
        result["asr_model"],
        result["eval_time_s"],
    )
    logger.info(sep)

    # Table header
    header = f"{'Runner':<12}{'UTMOS':<16}{'CER':<16}{'Speaker Sim':<14}{'Status'}"
    logger.info(header)
    logger.info("-" * 62)

    # Reference row
    ref_stats = runners_stats.get(ref_runner, {})
    ref_utmos = (
        f"{ref_stats.get('utmos_mean', 0):.2f}\u00b1{ref_stats.get('utmos_std', 0):.2f}"
    )
    ref_cer = (
        f"{ref_stats.get('cer_mean', 0):.2f}\u00b1{ref_stats.get('cer_std', 0):.2f}"
    )
    logger.info(f"{ref_runner:<12}{ref_utmos:<16}{ref_cer:<16}{'-':<14}(ref)")

    # Comparison map for quick lookup
    cmp_by_opt = {c["opt"]: c for c in comparisons}

    # Opt runner rows
    for opt_name in opt_runners:
        stats = runners_stats.get(opt_name, {})
        utmos_str = (
            f"{stats.get('utmos_mean', 0):.2f}\u00b1{stats.get('utmos_std', 0):.2f}"
        )
        cer_str = f"{stats.get('cer_mean', 0):.2f}\u00b1{stats.get('cer_std', 0):.2f}"
        cmp = cmp_by_opt.get(opt_name, {})
        sim_str = f"{cmp.get('speaker_sim_mean', 0):.2f}"
        status_str = cmp.get("status", "N/A")
        logger.info(
            f"{opt_name:<12}{utmos_str:<16}{cer_str:<16}{sim_str:<14}{status_str}"
        )

    logger.info(sep)
    logger.info("Overall: %s", result["status"])

    # Print failures per comparison
    for cmp in comparisons:
        if cmp.get("failures"):
            logger.info("  %s failures:", cmp["opt"])
            for f in cmp["failures"]:
                logger.info("    - %s", f)

    logger.info(sep)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entrypoint for Tier 3 quality evaluation.

    Parses --mode, --runners, and --output arguments, runs the independent
    distribution comparison, and saves the JSON report to disk.
    """
    parser = argparse.ArgumentParser(
        description="TTS quality evaluation: independent distribution comparison"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="Evaluation mode (fast=CI, full=PR gate)",
    )
    parser.add_argument(
        "--runners",
        nargs="*",
        default=None,
        metavar="OPT",
        help="Optimized runner names to compare against base (default: all available)",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="base",
        help="Reference runner name (default: base)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/tier3_<mode>_multi.json)",
    )
    args = parser.parse_args()

    result = run_tier3(args.mode, ref_runner=args.ref, opt_runners=args.runners)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / f"tier3_{args.mode}_multi.json")
    # Strip wav_path from samples to keep JSON clean
    for sample in result.get("ref_samples", []):
        sample.pop("wav_path", None)
    for opt_samples in result.get("opt_samples", {}).values():
        for sample in opt_samples:
            sample.pop("wav_path", None)

    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
