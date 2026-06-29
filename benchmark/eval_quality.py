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
    # Fast CI evaluation (~15 min, Cohere Transcribe, 1 run/sentence)
    python -m benchmark.eval_quality --mode fast

    # Full PR gate evaluation (~80 min, Cohere Transcribe, 3 runs/sentence)
    python -m benchmark.eval_quality --mode full
"""

import argparse
import gc
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
DEFAULT_TIER3_OPT_RUNNERS = [
    "base+tq",
    "triton",
    "triton+tq",
    "faster",
    "hybrid",
    "hybrid+tq",
]

# Cache for expensive model loads
_cohere_asr_cache: dict[str, tuple[Any, Any]] = {}  # {model_id: (processor, model)}
_utmos_predictor_cache: list[Any] = []
_voice_encoder_cache: list[Any] = []


# ────────────────────────────────────────────────────────────
# Model caching helpers
# ────────────────────────────────────────────────────────────


def _get_cohere_asr(model_id: str) -> tuple[Any, Any]:
    """Load and cache Cohere Transcribe ASR model."""
    if model_id not in _cohere_asr_cache:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        _cohere_asr_cache[model_id] = (processor, model)
    return _cohere_asr_cache[model_id]


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
    asr_model: str,
    language: str,
) -> dict[str, Any]:
    """Compute CER for a single WAV against ground truth text."""
    from jiwer import cer

    processor, model = _get_cohere_asr(asr_model)
    texts = model.transcribe(
        processor=processor,
        audio_files=[str(wav_path)],
        language=language,
    )
    transcript = texts[0].strip()

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


def generate_sample(
    runner: Any,
    sentence: dict[str, str],
    output_dir: Path,
    runner_name: str,
    idx: int,
    run: int,
) -> dict[str, Any]:
    """Generate audio with one model and record latency/VRAM metadata.

    Args:
        runner: Loaded TTS runner instance.
        sentence: Dict with 'text' and 'language' keys.
        output_dir: Directory to save WAV files.
        runner_name: Name for file naming (e.g., 'base', 'triton').
        idx: Sentence index.
        run: Run number (for multi-run modes).

    Returns:
        Dict with wav_path and generation metadata.
    """
    baseline_vram_gb = (
        torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    wall_start = time.perf_counter()
    output = runner.generate(text=sentence["text"], language=sentence["language"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - wall_start
    audio = output.get("audio")
    sr = int(output.get("sample_rate", 24000) or 24000)
    audio_samples = len(audio) if audio is not None else 0
    audio_duration_s = audio_samples / sr if sr > 0 else 0.0
    runner_time_s = float(output.get("time_s", wall_time_s))
    peak_vram_gb = float(
        output.get(
            "peak_vram_gb",
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0,
        )
    )
    rtf = audio_duration_s / runner_time_s if runner_time_s > 0 else 0.0

    wav_path = output_dir / f"{runner_name}_{idx:03d}_r{run}.wav"
    sf.write(str(wav_path), audio, sr)

    return {
        "wav_path": str(wav_path),
        "sentence_idx": idx,
        "run": run,
        "text": sentence["text"],
        "language": sentence["language"],
        "runner_time_s": round(runner_time_s, 4),
        "wall_time_s": round(wall_time_s, 4),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "baseline_vram_gb": round(baseline_vram_gb, 4),
        "inference_delta_gb": round(peak_vram_gb - baseline_vram_gb, 4),
        "audio_duration_s": round(audio_duration_s, 4),
        "audio_samples": int(audio_samples),
        "rtf": round(rtf, 4),
    }


def _reset_cuda_memory_measurement() -> None:
    """Reset Python and CUDA allocator state between quality runner loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def score_generated_sample(
    result: dict[str, Any],
    asr_model: str,
) -> dict[str, Any]:
    """Attach CER transcript and UTMOS metrics to an existing WAV result."""
    cer_result = compute_cer(
        result["wav_path"],
        result["text"],
        asr_model,
        result["language"],
    )
    result["cer"] = cer_result["cer"]
    result["transcript"] = cer_result["transcript"]
    result["utmos"] = compute_utmos(result["wav_path"])
    return result


def _score_generated_results(
    results: list[dict[str, Any]],
    asr_model: str,
) -> list[dict[str, Any]]:
    """Score generated WAVs after all TTS runner VRAM measurements are done."""
    return [score_generated_sample(result, asr_model) for result in results]


def generate_and_evaluate(
    runner: Any,
    sentence: dict[str, str],
    output_dir: Path,
    runner_name: str,
    idx: int,
    run: int,
    asr_model: str = "small",
) -> dict[str, Any]:
    """Generate audio with one model and evaluate independently."""
    result = generate_sample(runner, sentence, output_dir, runner_name, idx, run)
    return score_generated_sample(result, asr_model)


def _run_model_evaluation(
    runner_factory: type,
    runner_name: str,
    sentences: list[dict[str, str]],
    runs_per_sentence: int,
    output_dir: Path,
    asr_model: str,
    warmup_runs: int,
    score_samples: bool = True,
) -> list[dict[str, Any]]:
    """Generate and evaluate all sentences with one model.

    Args:
        runner_factory: Callable that returns a runner instance (class or
            zero-argument factory).
        runner_name: Name identifier for the runner.
        sentences: List of sentence dicts.
        runs_per_sentence: Number of generations per sentence.
        output_dir: Output directory for WAV files.
        asr_model: ASR model ID.
        warmup_runs: Number of warmup generations.

    Returns:
        List of per-sample result dicts.
    """
    _reset_cuda_memory_measurement()
    runner = runner_factory()
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
                result = generate_sample(
                    runner,
                    sent,
                    output_dir,
                    runner_name,
                    idx,
                    run,
                )
                results.append(result)
    finally:
        runner.unload_model()
        _reset_cuda_memory_measurement()

    if score_samples:
        return _score_generated_results(results, asr_model)
    return results


# ────────────────────────────────────────────────────────────
# Batched generation (generate_batch) adapter
# ────────────────────────────────────────────────────────────


def _group_indices_by_language(
    sentences: list[dict[str, str]],
) -> dict[str, list[int]]:
    """Group sentence indices by language (``generate_batch`` takes one language)."""
    groups: dict[str, list[int]] = {}
    for i, sent in enumerate(sentences):
        groups.setdefault(sent["language"], []).append(i)
    return groups


def _batched_sample_result(
    sent: dict[str, str],
    audio: np.ndarray,
    sr: int,
    idx: int,
    run: int,
    wav_path: Path,
    per_sample_time_s: float,
    peak_vram_gb: float,
) -> dict[str, Any]:
    """Build a per-sample result dict matching ``generate_sample``'s schema.

    Timing is amortised across the batch (per-sample = batch wall / batch size);
    only UTMOS/CER/speaker-sim gate the verdict, so amortised timing is for
    reporting only.
    """
    samples = int(len(audio))
    dur = samples / sr if sr > 0 else 0.0
    return {
        "wav_path": str(wav_path),
        "sentence_idx": idx,
        "run": run,
        "text": sent["text"],
        "language": sent["language"],
        "runner_time_s": round(per_sample_time_s, 4),
        "wall_time_s": round(per_sample_time_s, 4),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "baseline_vram_gb": 0.0,
        "inference_delta_gb": 0.0,
        "audio_duration_s": round(dur, 4),
        "audio_samples": samples,
        "rtf": round(dur / per_sample_time_s, 4) if per_sample_time_s > 0 else 0.0,
    }


def _run_batched_model_evaluation(
    runner_factory: type,
    runner_name: str,
    sentences: list[dict[str, str]],
    runs_per_sentence: int,
    output_dir: Path,
    asr_model: str,
    warmup_runs: int,
    score_samples: bool = True,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Generate every sentence via ``generate_batch`` (one batch per language).

    Unlike ``_run_model_evaluation`` (per-sentence ``generate``), this exercises
    the real batched path: all sentences of a language are synthesised in a
    single batch, so left-padding, the shared CUDA graph, and per-row sampling
    are all in effect — exactly the behaviour whose quality we want to validate.
    Any runner works: ``base``/``triton`` batch via the HF list path,
    ``faster``/``hybrid`` via a captured ``B``-batched CUDA graph.
    """
    _reset_cuda_memory_measurement()
    runner = runner_factory()
    groups = _group_indices_by_language(sentences)
    results: list[dict[str, Any]] = []
    try:
        runner.load_model()
        first_lang, first_idx = next(iter(groups.items()))
        warm_texts = [sentences[i]["text"] for i in first_idx]
        for w in range(warmup_runs):
            logger.info("[%s] Warmup batch %d/%d", runner_name, w + 1, warmup_runs)
            runner.generate_batch(
                warm_texts, language=first_lang, batch_size=batch_size
            )
        for run in range(runs_per_sentence):
            for lang, idx_group in groups.items():
                texts = [sentences[i]["text"] for i in idx_group]
                logger.info(
                    "[%s] Batch %s x%d (run %d)", runner_name, lang, len(texts), run
                )
                out = runner.generate_batch(texts, language=lang, batch_size=batch_size)
                per_sample = out["wall_s"] / max(len(idx_group), 1)
                vram = float(out.get("peak_vram_gb", 0.0))
                for local, i in enumerate(idx_group):
                    res = out["results"][local]
                    audio = np.asarray(res["audio"])
                    sr = int(res.get("sample_rate", 24000) or 24000)
                    wav_path = output_dir / f"{runner_name}_{i:03d}_r{run}.wav"
                    sf.write(str(wav_path), audio, sr)
                    results.append(
                        _batched_sample_result(
                            sentences[i], audio, sr, i, run, wav_path, per_sample, vram
                        )
                    )
    finally:
        runner.unload_model()
        _reset_cuda_memory_measurement()

    if score_samples:
        return _score_generated_results(results, asr_model)
    return results


def _evaluate_runner(
    runner_name: str,
    sentences: list[dict[str, str]],
    runs_per_sentence: int,
    output_dir: Path,
    asr_model: str,
    warmup_runs: int,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Route to the batched or per-sentence evaluation path.

    ``batch_size > 1`` selects the batched ``generate_batch`` adapter for any
    runner; otherwise each sentence is generated individually via ``generate``.
    """
    factory = _make_runner_factory(runner_name)
    if batch_size > 1:
        return _run_batched_model_evaluation(
            factory,
            runner_name,
            sentences,
            runs_per_sentence,
            output_dir,
            asr_model,
            warmup_runs,
            score_samples=False,
            batch_size=batch_size,
        )
    return _run_model_evaluation(
        factory,
        runner_name,
        sentences,
        runs_per_sentence,
        output_dir,
        asr_model,
        warmup_runs,
        score_samples=False,
    )


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


def _values(results: list[dict[str, Any]], key: str) -> list[float]:
    """Extract numeric values for a key from per-sample results."""
    return [float(r[key]) for r in results if r.get(key) is not None]


def _compute_runner_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute quality, latency, and VRAM summary for one runner."""
    utmos_vals = _values(results, "utmos")
    cer_vals = _values(results, "cer")
    runner_time_vals = _values(results, "runner_time_s")
    wall_time_vals = _values(results, "wall_time_s")
    peak_vram_vals = _values(results, "peak_vram_gb")
    rtf_vals = _values(results, "rtf")

    stats: dict[str, float] = {}
    if utmos_vals:
        stats["utmos_mean"] = round(float(np.mean(utmos_vals)), 4)
        stats["utmos_std"] = round(float(np.std(utmos_vals)), 4)
    if cer_vals:
        stats["cer_mean"] = round(float(np.mean(cer_vals)), 4)
        stats["cer_std"] = round(float(np.std(cer_vals)), 4)
    if runner_time_vals:
        stats["runner_time_s_mean"] = round(float(np.mean(runner_time_vals)), 4)
        stats["runner_time_s_p50"] = round(
            float(np.percentile(runner_time_vals, 50)),
            4,
        )
    if wall_time_vals:
        stats["wall_time_s_mean"] = round(float(np.mean(wall_time_vals)), 4)
        stats["wall_time_s_p50"] = round(float(np.percentile(wall_time_vals, 50)), 4)
    if peak_vram_vals:
        stats["peak_vram_gb_mean"] = round(float(np.mean(peak_vram_vals)), 4)
        stats["peak_vram_gb_max"] = round(float(np.max(peak_vram_vals)), 4)
    if rtf_vals:
        stats["rtf_mean"] = round(float(np.mean(rtf_vals)), 4)
    return stats


def _compute_perf_comparison(
    ref_results: list[dict[str, Any]],
    opt_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute speedup and VRAM delta between reference and opt samples."""
    ref_time = _values(ref_results, "runner_time_s")
    opt_time = _values(opt_results, "runner_time_s")
    ref_vram = _values(ref_results, "peak_vram_gb")
    opt_vram = _values(opt_results, "peak_vram_gb")

    comparison: dict[str, float] = {}
    if ref_time and opt_time:
        ref_mean = float(np.mean(ref_time))
        opt_mean = float(np.mean(opt_time))
        comparison["runner_time_s_delta"] = round(opt_mean - ref_mean, 4)
        comparison["latency_speedup"] = (
            round(ref_mean / opt_mean, 4) if opt_mean else 0.0
        )
    if ref_vram and opt_vram:
        comparison["peak_vram_gb_delta"] = round(
            float(np.max(opt_vram) - np.max(ref_vram)),
            4,
        )
    return comparison


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


def _make_runner_factory(name: str) -> type:
    """Return a zero-argument callable that constructs a runner by name.

    Supports TurboQuant variants (``'triton+tq'``, ``'hybrid+tq'``) via
    ``create_runner``.  Plain names use the runner class directly so the
    existing ``runner_cls()`` call pattern is preserved.

    Args:
        name: Runner name, e.g. ``'base'``, ``'triton+tq'``.

    Returns:
        A zero-argument callable returning a runner instance.
    """
    from qwen3_tts_triton.models import create_runner

    def _factory() -> object:
        return create_runner(name)

    return _factory  # type: ignore[return-value]


def run_tier3(
    mode: str = "fast",
    ref_runner: str = "base",
    opt_runners: list[str] | None = None,
    languages: list[str] | None = None,
    sentences_per_language: int | None = None,
    warmup_runs: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run Tier 3 quality evaluation: independent distribution comparison.

    Generates reference samples once, then compares each opt runner against
    the reference independently.

    Args:
        mode: 'fast' (1 run) or 'full' (3 runs, Mann-Whitney).
        ref_runner: Reference runner name (default: 'base').
        opt_runners: Optimized runner names (default: all release runners
            except the reference runner).
        languages: Optional language filter. Defaults to configured languages.
        sentences_per_language: Optional per-language sentence cap.
        warmup_runs: Optional warmup override for smoke tests.

    Returns:
        Complete evaluation result dict with per-runner comparisons.
    """
    if opt_runners is None:
        opt_runners = [name for name in DEFAULT_TIER3_OPT_RUNNERS if name != ref_runner]

    tier3_cfg = EVAL_CONFIG["tier3"]
    warmup = (
        int(EVAL_CONFIG["warmup_runs"]) if warmup_runs is None else int(warmup_runs)
    )

    runs_per_sentence = (
        tier3_cfg["runs_per_sentence_fast"]
        if mode == "fast"
        else tier3_cfg["runs_per_sentence_full"]
    )
    asr_model = tier3_cfg["asr_model"]

    sentences = _select_sentences(
        mode,
        languages=languages,
        sentences_per_language=sentences_per_language,
    )
    wav_subdir = f"multi_{mode}" if batch_size <= 1 else f"multi_{mode}_b{batch_size}"
    output_dir = OUTPUTS_DIR / wav_subdir
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
    ref_results = _evaluate_runner(
        ref_runner,
        sentences,
        runs_per_sentence,
        output_dir,
        str(asr_model),
        warmup,
        batch_size=batch_size,
    )

    # Phase 1b: Generate opt runner samples before loading quality models.
    opt_samples: dict[str, list[dict[str, Any]]] = {}
    for opt_name in opt_runners:
        logger.info("Phase 1: Generating with %s...", opt_name)
        opt_samples[opt_name] = _evaluate_runner(
            opt_name,
            sentences,
            runs_per_sentence,
            output_dir,
            str(asr_model),
            warmup,
            batch_size=batch_size,
        )

    # Phase 2: Score generated WAVs.  Keep this after all runner generation so
    # ASR/UTMOS GPU caches do not contaminate TTS VRAM measurements.
    logger.info("Phase 2: Scoring generated WAVs...")
    ref_results = _score_generated_results(ref_results, str(asr_model))
    for opt_name, opt_results in opt_samples.items():
        opt_samples[opt_name] = _score_generated_results(opt_results, str(asr_model))

    # Phase 3: Distribution comparison + verdict per opt runner
    logger.info("Phase 3: Comparing distributions...")
    comparisons: list[dict[str, Any]] = []
    runners_stats: dict[str, dict[str, float]] = {}

    runners_stats[ref_runner] = _compute_runner_stats(ref_results)

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
                **_compute_perf_comparison(ref_results, opt_results),
            }
        )
        runners_stats[opt_name] = _compute_runner_stats(opt_results)

    eval_time = round(time.perf_counter() - t_start, 2)
    overall_status = (
        "PASS" if all(c["status"] == "PASS" for c in comparisons) else "FAIL"
    )

    result = {
        "status": overall_status,
        "mode": mode,
        "ref_runner": ref_runner,
        "opt_runners": opt_runners,
        "batch_size": batch_size,
        "wav_dir": str(output_dir),
        "languages": sorted({sentence["language"] for sentence in sentences}),
        "sentences_per_language": sentences_per_language,
        "warmup_runs": warmup,
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


def _select_sentences(
    mode: str,
    languages: list[str] | None = None,
    sentences_per_language: int | None = None,
) -> list[dict[str, str]]:
    """Select evaluation sentences based on mode.

    In fast mode, takes the first 5 sentences per language.
    In full mode, uses all sentences per language.

    Args:
        mode: Either 'fast' (CI subset) or 'full' (complete set).
        languages: Optional language filter.
        sentences_per_language: Optional cap applied after mode defaults.

    Returns:
        List of sentence dicts with 'text' and 'language' keys.
    """
    sentences: list[dict[str, str]] = []
    selected_languages = languages or list(EVAL_CONFIG["languages"])
    default_limit = 5 if mode == "fast" else None
    limit = (
        sentences_per_language if sentences_per_language is not None else default_limit
    )

    for lang in selected_languages:
        lang_sents = EVAL_SENTENCES.get(lang, [])
        sentences.extend(lang_sents[:limit] if limit is not None else lang_sents)
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

    sep = "\u2550" * 104
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
    header = (
        f"{'Runner':<22}{'UTMOS':<14}{'CER':<14}{'Latency':<12}"
        f"{'Speedup':<10}{'VRAM':<10}{'Speaker':<10}{'Status'}"
    )
    logger.info(header)
    logger.info("-" * 104)

    # Reference row
    ref_stats = runners_stats.get(ref_runner, {})
    ref_utmos = (
        f"{ref_stats.get('utmos_mean', 0):.2f}\u00b1{ref_stats.get('utmos_std', 0):.2f}"
    )
    ref_cer = (
        f"{ref_stats.get('cer_mean', 0):.2f}\u00b1{ref_stats.get('cer_std', 0):.2f}"
    )
    ref_latency = f"{ref_stats.get('runner_time_s_mean', 0):.2f}s"
    ref_vram = f"{ref_stats.get('peak_vram_gb_max', 0):.2f}GB"
    logger.info(
        f"{ref_runner:<22}{ref_utmos:<14}{ref_cer:<14}{ref_latency:<12}"
        f"{'-':<10}{ref_vram:<10}{'-':<10}(ref)"
    )

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
        latency_str = f"{stats.get('runner_time_s_mean', 0):.2f}s"
        speedup_str = f"{cmp.get('latency_speedup', 0):.2f}x"
        vram_str = f"{stats.get('peak_vram_gb_max', 0):.2f}GB"
        sim_str = f"{cmp.get('speaker_sim_mean', 0):.2f}"
        status_str = cmp.get("status", "N/A")
        logger.info(
            f"{opt_name:<22}{utmos_str:<14}{cer_str:<14}{latency_str:<12}"
            f"{speedup_str:<10}{vram_str:<10}{sim_str:<10}{status_str}"
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


def _parse_cli_list(values: list[str] | None) -> list[str] | None:
    """Parse nargs values that may also contain comma-separated entries."""
    if not values:
        return None
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


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
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language filter, e.g. --languages ko en",
    )
    parser.add_argument(
        "--sentences-per-language",
        type=int,
        default=None,
        help="Optional per-language sentence cap for smoke runs",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="Override warmup generation count",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch many sentences per generate_batch call (>1 = batched serving "
        "path; works for any runner). Default 1 = per-sentence generate().",
    )
    args = parser.parse_args()

    languages = _parse_cli_list(args.languages)
    result = run_tier3(
        args.mode,
        ref_runner=args.ref,
        opt_runners=args.runners,
        languages=languages,
        sentences_per_language=args.sentences_per_language,
        warmup_runs=args.warmup_runs,
        batch_size=args.batch_size,
    )

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
