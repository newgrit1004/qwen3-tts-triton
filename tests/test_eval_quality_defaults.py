"""Regression tests for Tier 3 runner defaults."""

from __future__ import annotations

from pathlib import Path

import benchmark.eval_quality as eval_quality


def _patch_fast_fake_eval(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Replace expensive generation/evaluation with one deterministic sample."""
    monkeypatch.setattr(
        eval_quality,
        "_select_sentences",
        lambda mode, languages=None, sentences_per_language=None: [
            {"text": "hello", "language": "en"}
        ],
    )
    monkeypatch.setattr(eval_quality, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(eval_quality, "_print_summary", lambda result: None)

    def fake_run_model_evaluation(
        runner_factory: type,
        runner_name: str,
        sentences: list[dict[str, str]],
        runs_per_sentence: int,
        output_dir: Path,
        asr_model: str,
        warmup_runs: int,
        score_samples: bool = True,
        batch_size: int = 1,
    ) -> list[dict[str, object]]:
        runner_time_s = 2.0 if runner_name == "base" else 1.0
        peak_vram_gb = 12.0 if runner_name == "base" else 11.5
        return [
            {
                "wav_path": str(output_dir / f"{runner_name}.wav"),
                "sentence_idx": 0,
                "run": 0,
                "text": sentences[0]["text"],
                "language": sentences[0]["language"],
                "cer": 0.01,
                "transcript": sentences[0]["text"],
                "utmos": 3.5,
                "runner_time_s": runner_time_s,
                "wall_time_s": runner_time_s + 0.1,
                "peak_vram_gb": peak_vram_gb,
                "audio_duration_s": 3.0,
                "audio_samples": 72000,
                "rtf": 3.0 / runner_time_s,
            }
        ]

    # Both the per-sentence and batched evaluation paths are stubbed so the
    # opt-in batched runners can be exercised without a GPU.
    monkeypatch.setattr(
        eval_quality,
        "_run_model_evaluation",
        fake_run_model_evaluation,
    )
    monkeypatch.setattr(
        eval_quality,
        "_run_batched_model_evaluation",
        fake_run_model_evaluation,
    )
    monkeypatch.setattr(
        eval_quality,
        "_score_generated_results",
        lambda results, asr_model: results,
    )
    monkeypatch.setattr(
        eval_quality,
        "_compute_verdict",
        lambda ref, opt, ref_name, opt_name, mode: {
            "ref": ref_name,
            "opt": opt_name,
            "status": "PASS",
            "utmos_delta": 0.0,
            "cer_delta": 0.0,
            "speaker_sim_mean": 0.8,
            "failures": [],
        },
    )


def test_run_tier3_defaults_include_base_tq_and_all_release_runners(
    monkeypatch,
    tmp_path: Path,
):
    """Default Tier 3 runs should cover the full 7-mode release matrix."""
    _patch_fast_fake_eval(monkeypatch, tmp_path)

    result = eval_quality.run_tier3(mode="fast")

    assert result["opt_runners"] == [
        "base+tq",
        "triton",
        "triton+tq",
        "faster",
        "hybrid",
        "hybrid+tq",
    ]


def test_run_tier3_batch_size_routes_through_batched_adapter(
    monkeypatch,
    tmp_path: Path,
):
    """``batch_size > 1`` routes every runner through the batched adapter."""
    _patch_fast_fake_eval(monkeypatch, tmp_path)

    seen: dict[str, int] = {}
    faked = eval_quality._run_batched_model_evaluation  # stub installed above

    def spy(*args: object, **kwargs: object) -> object:
        seen[str(args[1])] = int(kwargs.get("batch_size", 1))  # args[1]=runner_name
        return faked(*args, **kwargs)

    monkeypatch.setattr(eval_quality, "_run_batched_model_evaluation", spy)

    result = eval_quality.run_tier3(mode="fast", opt_runners=["hybrid"], batch_size=8)

    # Batched serving is a capability flag, not a runner name; the 7-mode names
    # are unchanged and both ref + opt runners ran the batched adapter at B=8.
    assert result["opt_runners"] == ["hybrid"]
    assert result["batch_size"] == 8
    assert seen == {"base": 8, "hybrid": 8}


def test_run_tier3_records_latency_and_vram_stats(monkeypatch, tmp_path: Path):
    """Tier 3 reports should include performance evidence, not only quality."""
    _patch_fast_fake_eval(monkeypatch, tmp_path)

    result = eval_quality.run_tier3(
        mode="fast",
        opt_runners=["hybrid"],
    )

    assert result["runners"]["base"]["runner_time_s_mean"] == 2.0
    assert result["runners"]["hybrid"]["peak_vram_gb_max"] == 11.5
    assert result["comparisons"][0]["latency_speedup"] == 2.0
    assert result["comparisons"][0]["peak_vram_gb_delta"] == -0.5


def test_run_tier3_passes_language_sentence_and_warmup_filters(
    monkeypatch,
    tmp_path: Path,
):
    """Smoke evaluations should be able to narrow language, size, and warmup."""
    _patch_fast_fake_eval(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def fake_select_sentences(
        mode: str,
        languages: list[str] | None = None,
        sentences_per_language: int | None = None,
    ) -> list[dict[str, str]]:
        captured["mode"] = mode
        captured["languages"] = languages
        captured["sentences_per_language"] = sentences_per_language
        return [{"text": "안녕하세요", "language": "ko"}]

    monkeypatch.setattr(eval_quality, "_select_sentences", fake_select_sentences)

    result = eval_quality.run_tier3(
        mode="fast",
        opt_runners=["hybrid"],
        languages=["ko"],
        sentences_per_language=1,
        warmup_runs=0,
    )

    assert captured == {
        "mode": "fast",
        "languages": ["ko"],
        "sentences_per_language": 1,
    }
    assert result["languages"] == ["ko"]
    assert result["warmup_runs"] == 0
