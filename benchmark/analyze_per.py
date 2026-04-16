"""Phoneme Error Rate (PER) analysis on existing eval WAVs.

Reads WAV files from eval_partial output, runs Cohere Transcribe ASR, applies
language-specific G2P, and computes PER to detect fine-grained
pronunciation differences that CER may miss.

Usage:
    python -m benchmark.analyze_per \
        --eval-dir benchmark/output/eval_partial/partial_full \
        --asr-model CohereLabs/cohere-transcribe-03-2026
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.eval_config import (
    EVAL_CONFIG,
    EVAL_SENTENCES,
    TONGUE_TWISTER_SENTENCES,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def _get_sentences(sentence_set: str = "standard") -> list[dict[str, str]]:
    """Get all eval sentences in generation order.

    Args:
        sentence_set: 'standard' for EVAL_SENTENCES,
            'tongue_twister' for TONGUE_TWISTER_SENTENCES.
    """
    source = (
        TONGUE_TWISTER_SENTENCES if sentence_set == "tongue_twister" else EVAL_SENTENCES
    )
    sentences: list[dict[str, str]] = []
    for lang in EVAL_CONFIG["languages"]:
        sentences.extend(source.get(lang, []))
    return sentences


# ── G2P per language ─────────────────────────────────────────


def _g2p_chinese(text: str) -> list[str]:
    """Convert Chinese text to pinyin phoneme sequence."""
    from pypinyin import Style, lazy_pinyin

    return lazy_pinyin(text, style=Style.TONE3)


def _g2p_korean(text: str) -> list[str]:
    """Convert Korean text to jamo (decomposed) phoneme sequence."""
    from jamo import h2j

    # Remove non-Korean characters
    korean_only = re.sub(r"[^\uac00-\ud7a3]", "", text)
    if not korean_only:
        return []
    jamo_str = h2j(korean_only)
    return list(jamo_str)


def _g2p_english(text: str) -> list[str]:
    """Convert English text to ARPAbet phoneme sequence."""
    from g2p_en import G2p

    g2p = G2p()
    phonemes = g2p(text.lower())
    # Filter out spaces and punctuation
    return [p for p in phonemes if p.strip() and not re.match(r"^[^\w]$", p)]


def text_to_phonemes(text: str, language: str) -> list[str]:
    """Convert text to phoneme sequence using language-specific G2P."""
    if language == "zh":
        return _g2p_chinese(text)
    elif language == "ko":
        return _g2p_korean(text)
    elif language == "en":
        return _g2p_english(text)
    else:
        logger.warning("Unknown language %s, falling back to character list", language)
        return list(text)


# ── Edit distance ────────────────────────────────────────────


def _edit_distance(seq1: list[str], seq2: list[str]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_per(expected_phonemes: list[str], actual_phonemes: list[str]) -> float:
    """Compute Phoneme Error Rate = edit_distance / len(expected)."""
    if not expected_phonemes:
        return 0.0
    dist = _edit_distance(expected_phonemes, actual_phonemes)
    return dist / len(expected_phonemes)


# ── ASR ──────────────────────────────────────────────────────


def _load_cohere_asr(model_id: str) -> tuple[Any, Any]:
    """Load Cohere Transcribe ASR model."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Cohere Transcribe %s ...", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model


def _transcribe(asr_model: tuple[Any, Any], wav_path: str, language: str) -> str:
    """Transcribe a WAV file."""
    processor, model = asr_model
    texts = model.transcribe(
        processor=processor,
        audio_files=[wav_path],
        language=language,
    )
    return texts[0].strip()


# ── Main analysis ────────────────────────────────────────────


def _discover_runners(eval_dir: Path) -> list[str]:
    """Discover runner names from WAV filenames."""
    runners: set[str] = set()
    for wav in eval_dir.glob("*.wav"):
        # Pattern: RunnerName_NNN_rN.wav
        parts = wav.stem.rsplit("_", 2)
        if len(parts) >= 3:
            runners.add(parts[0])
    return sorted(runners)


def analyze_per(
    eval_dir: Path,
    asr_model: str = "CohereLabs/cohere-transcribe-03-2026",
    sentence_set: str = "standard",
) -> dict[str, Any]:
    """Analyze PER for all runners in eval_dir.

    Args:
        eval_dir: Directory with WAV files from eval_partial.
        asr_model: ASR model ID.
        sentence_set: 'standard' or 'tongue_twister'.

    Returns:
        Result dict with per-runner, per-language PER stats.
    """
    sentences = _get_sentences(sentence_set)
    runners = _discover_runners(eval_dir)
    logger.info("Runners found: %s", runners)
    logger.info("Sentences: %d", len(sentences))

    cohere_model = _load_cohere_asr(asr_model)

    # Detect runs per sentence from files
    sample_wav = list(eval_dir.glob(f"{runners[0]}_000_r*.wav"))
    runs_per = len(sample_wav)
    logger.info("Runs per sentence: %d", runs_per)

    results: dict[str, list[dict[str, Any]]] = {r: [] for r in runners}

    for runner in runners:
        logger.info("Processing %s ...", runner)
        for sent_idx, sent in enumerate(sentences):
            text = sent["text"]
            lang = sent["language"]
            expected_phonemes = text_to_phonemes(text, lang)

            for run_idx in range(runs_per):
                wav_path = eval_dir / f"{runner}_{sent_idx:03d}_r{run_idx}.wav"
                if not wav_path.exists():
                    logger.warning("Missing: %s", wav_path)
                    continue

                transcript = _transcribe(cohere_model, str(wav_path), lang)
                actual_phonemes = text_to_phonemes(transcript, lang)
                per = compute_per(expected_phonemes, actual_phonemes)

                results[runner].append(
                    {
                        "sent_idx": sent_idx,
                        "run_idx": run_idx,
                        "language": lang,
                        "text": text,
                        "transcript": transcript,
                        "expected_phonemes": expected_phonemes[
                            :10
                        ],  # truncate for JSON
                        "actual_phonemes": actual_phonemes[:10],
                        "n_expected": len(expected_phonemes),
                        "n_actual": len(actual_phonemes),
                        "edit_distance": _edit_distance(
                            expected_phonemes, actual_phonemes
                        ),
                        "per": per,
                    }
                )

    # Aggregate stats
    summary: dict[str, Any] = {}
    for runner, samples in results.items():
        per_vals = [s["per"] for s in samples]
        per_acc = [1.0 - s["per"] for s in samples]

        # Per-language
        languages = sorted({s["language"] for s in samples})
        per_lang: dict[str, dict[str, float]] = {}
        for lang in languages:
            lang_samples = [s for s in samples if s["language"] == lang]
            lang_per = [s["per"] for s in lang_samples]
            per_lang[lang] = {
                "per_mean": round(float(np.mean(lang_per)), 4),
                "per_std": round(float(np.std(lang_per)), 4),
                "accuracy_mean": round(1.0 - float(np.mean(lang_per)), 4),
                "n_samples": len(lang_samples),
            }

        summary[runner] = {
            "per_mean": round(float(np.mean(per_vals)), 4),
            "per_std": round(float(np.std(per_vals)), 4),
            "accuracy_mean": round(float(np.mean(per_acc)), 4),
            "n_samples": len(samples),
            "per_language": per_lang,
        }

    # Print summary
    _print_summary(summary)

    return {"summary": summary, "details": results}


def _print_summary(summary: dict[str, Any]) -> None:
    """Print PER summary table."""
    sep = "=" * 76
    logger.info("")
    logger.info("PHONEME ERROR RATE (PER) ANALYSIS")
    logger.info(sep)

    header = f"{'Runner':<20} {'PER (overall)':<16} {'Accuracy':<12} {'N':<6}"
    logger.info(header)
    logger.info("-" * 76)
    for runner, stats in summary.items():
        logger.info(
            "%-20s %-16s %-12s %-6d",
            runner,
            f"{stats['per_mean']:.4f}\u00b1{stats['per_std']:.4f}",
            f"{stats['accuracy_mean']:.4f}",
            stats["n_samples"],
        )
    logger.info(sep)

    # Per-language breakdown
    logger.info("")
    logger.info("Per-language PER (lower = better pronunciation):")
    logger.info("-" * 76)

    # Collect all languages
    all_langs: set[str] = set()
    for stats in summary.values():
        all_langs.update(stats["per_language"].keys())
    langs = sorted(all_langs)

    header2 = f"{'Runner':<20} " + " ".join(f"{lg:>18}" for lg in langs)
    logger.info(header2)
    for runner, stats in summary.items():
        pl = stats["per_language"]
        vals = " ".join(
            f"{pl.get(lg, {}).get('per_mean', 0):.4f}\u00b1"
            f"{pl.get(lg, {}).get('per_std', 0):.4f}"
            if lg in pl
            else f"{'N/A':>18}"
            for lg in langs
        )
        logger.info("%-20s %s", runner, vals)
    logger.info(sep)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="PER analysis on eval WAVs")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="benchmark/output/eval_partial/partial_full",
        help="Directory with eval WAV files.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="CohereLabs/cohere-transcribe-03-2026",
        help="ASR model ID (default: CohereLabs/cohere-transcribe-03-2026).",
    )
    parser.add_argument(
        "--sentence-set",
        choices=["standard", "tongue_twister"],
        default="standard",
        help="Sentence set: standard or tongue_twister.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.",
    )
    args = parser.parse_args()

    result = analyze_per(
        eval_dir=Path(args.eval_dir),
        asr_model=args.asr_model,
        sentence_set=args.sentence_set,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / "per_analysis.json")
    # Save summary only (details too large)
    Path(out_path).write_text(json.dumps(result["summary"], indent=2, default=str))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
