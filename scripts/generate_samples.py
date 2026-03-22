#!/usr/bin/env python3
"""Batch-generate TTS voice samples for all runner modes.

Generates 10 custom voice texts (5 Korean + 5 English) + voice cloning samples
per mode into assets/audio_samples/<mode>/ with a metadata.json index.

Usage::

    uv run python scripts/generate_samples.py
    uv run python scripts/generate_samples.py --modes base triton
    uv run python scripts/generate_samples.py \\
        --speaker vivian --output-dir assets/audio_samples
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen3_tts_triton.models.base_runner import BaseRunner

import soundfile as sf
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_AUDIO = PROJECT_ROOT / "assets" / "reference_audio" / "ljspeech_sample.wav"
REF_TEXT = "so subversive of meditation, so disturbing to the thoughts;"

SAMPLE_TEXTS: dict[str, list[tuple[str, str]]] = {
    "ko": [
        ("conversation", "안녕하세요, 오늘 하루도 좋은 하루 보내세요."),
        (
            "news",
            "정부는 오늘 새로운 경제 정책을 발표하며,"
            " 하반기 성장률 목표를 상향 조정했습니다.",
        ),
        (
            "technical",
            "트리톤 커널 퓨전은 RMSNorm과 잔차 연결을"
            " 하나의 GPU 커널로 합쳐 메모리 대역폭을 절약합니다.",
        ),
        (
            "emotional",
            "오랜 시간이 흘러 다시 찾은 고향의 골목길에서,"
            " 어린 시절의 추억이 물밀듯 밀려왔습니다.",
        ),
        (
            "descriptive",
            "대규모 언어 모델 기반의 음성 합성 시스템은"
            " 텍스트를 입력받아 자연스러운 음성을 생성하며,"
            " 다양한 화자와 스타일을 지원합니다.",
        ),
    ],
    "en": [
        ("conversation", "Hi there! How are you doing today?"),
        (
            "news",
            "The Federal Reserve announced a quarter-point rate cut on Wednesday,"
            " signaling a shift in monetary policy.",
        ),
        (
            "technical",
            "Triton kernel fusion merges RMSNorm and"
            " residual addition into a single GPU kernel,"
            " reducing memory bandwidth by eliminating"
            " intermediate tensor allocations.",
        ),
        (
            "emotional",
            "She stood at the edge of the cliff, watching the sunset paint the sky"
            " in shades of gold and crimson, feeling the weight of the world slowly"
            " lift from her shoulders.",
        ),
        (
            "descriptive",
            "Modern text-to-speech systems leverage"
            " transformer architectures to generate"
            " high-fidelity audio that captures subtle"
            " prosodic variations, emotional nuances,"
            " and speaker characteristics.",
        ),
    ],
}

CLONE_TEXTS: dict[str, list[tuple[str, str]]] = {
    "ko": [
        ("conversation", "안녕하세요, 클론된 목소리로 인사드립니다."),
        (
            "technical",
            "음성 클로닝 기술은 참조 오디오의 화자 특성을"
            " 추출하여 새로운 텍스트에 적용합니다.",
        ),
    ],
    "en": [
        ("conversation", "Hello, this voice was cloned from a reference audio sample."),
        (
            "technical",
            "Voice cloning extracts speaker characteristics from reference audio"
            " and applies them to new text.",
        ),
    ],
}

LANG_DISPLAY: dict[str, str] = {
    "ko": "Korean",
    "en": "English",
}

ALL_MODES = ["base", "triton", "faster", "hybrid"]


def _generate_for_mode(
    mode: str,
    speaker: str,
    output_dir: Path,
    *,
    skip_clone: bool = False,
) -> list[dict]:
    """Load one runner, generate custom voice + clone samples, unload."""
    from qwen3_tts_triton.models import get_runner_class

    logger.info("=== Mode: %s ===", mode)
    cls = get_runner_class(mode)
    runner = cls(device="cuda")
    runner.load_model()

    samples: list[dict] = []
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    # --- Custom voice samples ---
    for lang_code, texts in SAMPLE_TEXTS.items():
        lang_name = LANG_DISPLAY[lang_code]
        for idx, (style, text) in enumerate(texts, 1):
            filename = f"{lang_code}_{idx:02d}.wav"
            filepath = mode_dir / filename

            logger.info(
                "  [%s] %s/%s_%02d: %.40s...",
                mode,
                lang_code,
                style,
                idx,
                text,
            )

            t0 = time.perf_counter()
            result = runner.generate(
                text=text,
                language=lang_name,
                speaker=speaker,
            )
            gen_time = time.perf_counter() - t0

            audio = result["audio"]
            sr = result["sample_rate"]
            sf.write(str(filepath), audio, sr)

            duration_s = len(audio) / sr
            samples.append(
                {
                    "file": str(filepath.relative_to(output_dir)),
                    "mode": mode,
                    "type": "custom",
                    "language": lang_code,
                    "language_name": lang_name,
                    "style": style,
                    "text": text,
                    "duration_s": round(duration_s, 2),
                    "generation_time_s": round(gen_time, 2),
                }
            )
            logger.info(
                "    -> %.1fs audio, generated in %.2fs",
                duration_s,
                gen_time,
            )

    # --- Voice cloning samples ---
    if not skip_clone:
        samples.extend(_generate_clone_for_mode(runner, mode, output_dir, mode_dir))

    runner.unload_model()
    torch.cuda.empty_cache()
    logger.info("=== %s complete (%d samples) ===", mode, len(samples))
    return samples


def _generate_clone_for_mode(
    runner: "BaseRunner",
    mode: str,
    output_dir: Path,
    mode_dir: Path,
) -> list[dict]:
    """Generate voice cloning samples using LJSpeech reference audio."""
    if not REF_AUDIO.exists():
        logger.warning("Reference audio not found: %s — skipping clone", REF_AUDIO)
        return []

    samples: list[dict] = []

    for lang_code, texts in CLONE_TEXTS.items():
        lang_name = LANG_DISPLAY[lang_code]
        for idx, (style, text) in enumerate(texts, 1):
            filename = f"clone_{lang_code}_{idx:02d}.wav"
            filepath = mode_dir / filename

            logger.info(
                "  [%s] clone/%s_%02d: %.40s...",
                mode,
                lang_code,
                idx,
                text,
            )

            t0 = time.perf_counter()
            try:
                result = runner.generate_voice_clone(
                    text=text,
                    language=lang_name,
                    ref_audio=str(REF_AUDIO),
                    ref_text=REF_TEXT,
                )
            except (NotImplementedError, AttributeError, RuntimeError, ValueError) as e:
                logger.warning(
                    "    -> clone not supported for %s: %s — skipping",
                    mode,
                    e,
                )
                return samples
            gen_time = time.perf_counter() - t0

            audio = result["audio"]
            sr = result["sample_rate"]
            sf.write(str(filepath), audio, sr)

            duration_s = len(audio) / sr
            samples.append(
                {
                    "file": str(filepath.relative_to(output_dir)),
                    "mode": mode,
                    "type": "clone",
                    "language": lang_code,
                    "language_name": lang_name,
                    "style": style,
                    "text": text,
                    "ref_audio": str(REF_AUDIO.relative_to(PROJECT_ROOT)),
                    "ref_text": REF_TEXT,
                    "duration_s": round(duration_s, 2),
                    "generation_time_s": round(gen_time, 2),
                }
            )
            logger.info(
                "    -> %.1fs audio, generated in %.2fs",
                duration_s,
                gen_time,
            )

    return samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TTS voice samples for all runner modes",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ALL_MODES,
        choices=ALL_MODES,
        help="Runner modes to generate (default: all 4)",
    )
    parser.add_argument(
        "--speaker",
        default="sohee",
        help="Speaker name (default: sohee)",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/audio_samples",
        help="Output directory (default: assets/audio_samples)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip voice cloning samples",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for batch sample generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []
    total_start = time.perf_counter()

    for mode in args.modes:
        mode_samples = _generate_for_mode(
            mode, args.speaker, output_dir, skip_clone=args.skip_clone
        )
        all_samples.extend(mode_samples)

    total_elapsed = time.perf_counter() - total_start

    # Write metadata
    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "speaker": args.speaker,
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "sample_rate": 24000,
        "ref_audio": str(REF_AUDIO.relative_to(PROJECT_ROOT)),
        "ref_text": REF_TEXT,
        "total_samples": len(all_samples),
        "total_generation_time_s": round(total_elapsed, 1),
        "modes": args.modes,
        "samples": all_samples,
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    logger.info(
        "Done! %d samples in %.1fs. Metadata: %s",
        len(all_samples),
        total_elapsed,
        meta_path,
    )


if __name__ == "__main__":
    main()
