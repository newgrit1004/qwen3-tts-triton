"""Regression tests for Tier 3 runner defaults."""

from __future__ import annotations

from pathlib import Path

import benchmark.eval_quality as eval_quality


def test_run_tier3_defaults_include_base_tq_and_all_release_runners(
    monkeypatch,
    tmp_path: Path,
):
    """Default Tier 3 runs should cover the full 7-mode release matrix."""
    monkeypatch.setattr(
        eval_quality,
        "_select_sentences",
        lambda mode: [{"text": "hello", "language": "en"}],
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
    ) -> list[dict[str, object]]:
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
            }
        ]

    monkeypatch.setattr(
        eval_quality,
        "_run_model_evaluation",
        fake_run_model_evaluation,
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

    result = eval_quality.run_tier3(mode="fast")

    assert result["opt_runners"] == [
        "base+tq",
        "triton",
        "triton+tq",
        "faster",
        "hybrid",
        "hybrid+tq",
    ]
