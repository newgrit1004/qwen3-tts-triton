"""TTS Playground tab for the Streamlit dashboard.

Audio-first layout: large audio players with minimal metrics (total_s, RTF).
Full metric cards are in the Benchmarks tab.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from ui.i18n import t
from ui.utils import calculate_rtf, reset_vram_stats

logger = logging.getLogger(__name__)

_DISPLAY_TO_KEY: dict[str, str] = {
    "Base": "base",
    "Triton": "triton",
    "Faster": "faster",
    "Hybrid": "hybrid",
}

LANGUAGES = [
    "English",
    "Korean",
    "Japanese",
    "Chinese",
    "French",
    "Spanish",
    "German",
    "Portuguese",
]

LANG_CODES: dict[str, str] = {
    "English": "english",
    "Korean": "korean",
    "Japanese": "japanese",
    "Chinese": "chinese",
    "French": "french",
    "Spanish": "spanish",
    "German": "german",
    "Portuguese": "portuguese",
}

SAMPLE_TEXTS: dict[str, str] = {
    "English": "Hello, this is a Qwen3 TTS Triton demo.",
    "Korean": "안녕하세요, Qwen3 TTS Triton 데모입니다.",
    "Japanese": "こんにちは、Qwen3 TTS Tritonのデモです。",
    "Chinese": "你好，这是Qwen3 TTS Triton的演示。",
    "French": "Bonjour, ceci est une demo de Qwen3 TTS Triton.",
    "Spanish": "Hola, esta es una demo de Qwen3 TTS Triton.",
    "German": "Hallo, dies ist eine Qwen3 TTS Triton Demo.",
    "Portuguese": "Ola, esta e uma demo do Qwen3 TTS Triton.",
}

ALL_RUNNER_NAMES = ["Base", "Triton", "Faster", "Hybrid"]
DEFAULT_RUNNERS = ["Base", "Triton", "Faster", "Hybrid"]
DEFAULT_SPEAKERS = ["sohee", "vivian"]


def _render_controls() -> tuple[str, str, list[str], str, bytes | None]:
    """Render TTS controls inline and return user inputs."""
    col_left, col_right = st.columns(2)

    with col_left:

        def _on_language_change() -> None:
            lang = st.session_state.language_select
            st.session_state.synth_text_area = SAMPLE_TEXTS.get(lang, "")

        if "synth_text_area" not in st.session_state:
            st.session_state.synth_text_area = SAMPLE_TEXTS["English"]

        language = st.selectbox(
            t("sidebar.language"),
            LANGUAGES,
            index=0,
            key="language_select",
            on_change=_on_language_change,
        )
        lang_code = LANG_CODES.get(language, "en")

        text = st.text_area(
            t("sidebar.text_to_synth"),
            height=120,
            key="synth_text_area",
        )

    with col_right:
        selected_runners = st.multiselect(
            t("sidebar.runners"),
            ALL_RUNNER_NAMES,
            default=DEFAULT_RUNNERS,
        )

        voice_clone = st.toggle(t("sidebar.voice_clone"), value=False)
        ref_audio: bytes | None = None
        speaker = DEFAULT_SPEAKERS[0]

        if voice_clone:
            uploaded = st.file_uploader(
                t("sidebar.ref_audio"), type=["wav", "mp3", "flac"]
            )
            if uploaded is not None:
                ref_audio = uploaded.read()
        else:
            speaker = st.selectbox(
                t("sidebar.speaker"),
                DEFAULT_SPEAKERS,
                index=0,
                key="speaker_select",
            )

    return text, lang_code, selected_runners, speaker, ref_audio


def render_inference_tab() -> None:
    """Render the TTS Playground tab with inline controls."""
    # ── Controls ──
    text, lang_code, selected_runners, speaker, ref_audio = _render_controls()

    # ── Generate button ──
    if st.button(t("inference.run"), type="primary", width="stretch"):
        if not selected_runners:
            st.warning(t("inference.select_runner"))
            return
        st.toast(t("inference.toast_started"))
        _run_comparison(selected_runners, text, lang_code, speaker, ref_audio)

    st.markdown("---")

    # Show previous results if available
    results = st.session_state.get("inference_results")
    if results:
        _display_results(results, selected_runners)
    else:
        st.info(t("inference.prompt"))


def _run_comparison(
    selected_runners: list[str],
    text: str,
    lang_code: str,
    speaker: str = "sohee",
    ref_audio: bytes | None = None,
) -> None:
    """Execute inference for each runner with progress feedback."""
    results: dict[str, dict[str, Any]] = {}
    total = len(selected_runners)
    bar = st.progress(0, text=t("inference.starting"))

    for i, name in enumerate(selected_runners):
        bar.progress(
            i / total,
            text=t("inference.running_n", idx=i + 1, total=total, name=name),
        )

        with st.status(
            t("inference.status_running", name=name),
            expanded=True,
        ) as status:
            # Phase 1: Get runner instance
            runner = _get_runner(name)
            if runner is None:
                results[name] = {
                    "error": t("inference.unavailable", name=name),
                }
                status.update(
                    label=t("inference.status_error", name=name),
                    state="error",
                )
                continue

            try:
                # Phase 2: Load model (+ CUDA graph warmup for Faster/Hybrid)
                key = _DISPLAY_TO_KEY.get(name, name.lower())
                if key in ("faster", "hybrid"):
                    st.write(t("inference.loading_warmup", name=name))
                else:
                    st.write(t("inference.loading_model", name=name))
                reset_vram_stats()
                t_load = time.perf_counter()
                runner.load_model()
                load_s = time.perf_counter() - t_load

                # Phase 3: Generate audio
                st.write(t("inference.generating"))
                result = _generate(runner, text, lang_code, speaker, ref_audio)
                result["load_s"] = round(load_s, 2)
                results[name] = result

                elapsed = result.get("total_s", 0)
                rtf = result.get("rtf", 0)
                st.write(
                    t(
                        "inference.status_done_detail",
                        time=f"{elapsed:.1f}",
                        rtf=f"{rtf:.2f}",
                    )
                )
                status.update(
                    label=t(
                        "inference.status_done",
                        name=name,
                        time=f"{elapsed:.1f}",
                    ),
                    state="complete",
                    expanded=False,
                )
                st.toast(t("inference.toast_done", name=name, time=f"{elapsed:.1f}"))

            except Exception as e:
                results[name] = {"error": str(e)}
                st.write(str(e))
                status.update(
                    label=t("inference.status_error", name=name),
                    state="error",
                    expanded=True,
                )
            finally:
                runner.unload_model()
                reset_vram_stats()

    bar.progress(1.0, text=t("inference.all_done", total=total))
    bar.empty()

    st.session_state["inference_results"] = results


def _display_results(
    results: dict[str, dict[str, Any]],
    selected_runners: list[str],
) -> None:
    """Display audio-first results layout."""
    runner_names = [r for r in selected_runners if r in results]
    if not runner_names:
        runner_names = list(results.keys())

    cols = st.columns(len(runner_names)) if runner_names else []

    for col, name in zip(cols, runner_names):
        result = results.get(name, {"error": t("inference.not_run")})
        col.subheader(name)

        if result.get("error"):
            col.error(result["error"])
            continue

        # Large audio player
        audio = result.get("audio")
        sr = result.get("sample_rate", 24000)
        if audio is not None:
            col.audio(audio, sample_rate=sr)

        # Minimal metrics below audio
        total = result.get("total_s", 0)
        rtf = result.get("rtf", 0)
        col.caption(t("inference.metrics", total=f"{total:.3f}", rtf=f"{rtf:.2f}"))


def _get_runner(name: str) -> Any:
    """Import a runner class by display name with graceful fallback."""
    try:
        from qwen3_tts_triton.models import get_runner_class

        key = _DISPLAY_TO_KEY.get(name, name.lower())
        cls = get_runner_class(key)
        return cls() if cls else None
    except (ImportError, KeyError):
        return None


def _generate(
    runner: Any,
    text: str,
    language: str,
    speaker: str = "sohee",
    ref_audio: bytes | None = None,
) -> dict[str, Any]:
    """Run generation (model must already be loaded).

    Uses generate_voice_clone() when ref_audio is provided,
    otherwise uses generate() with speaker name.
    """
    t_start = time.perf_counter()

    if ref_audio is not None:
        # Voice cloning: save bytes to temp file, call generate_voice_clone
        output = _generate_voice_clone(runner, text, language, ref_audio)
    else:
        # Custom voice: use speaker name
        output = runner.generate(text=text, language=language, speaker=speaker)

    t_total = time.perf_counter() - t_start

    audio = output.get("audio")
    sr = output.get("sample_rate", 24000)
    ttfa = output.get("time_s", t_total)
    peak_vram = output.get("peak_vram_gb", 0.0)

    audio_len = len(audio) if audio is not None else 0
    rtf = calculate_rtf(audio_len, sr, t_total)

    return {
        "audio": audio,
        "sample_rate": sr,
        "ttfa_s": ttfa,
        "rtf": rtf,
        "total_s": t_total,
        "peak_vram_gb": peak_vram,
    }


def _generate_voice_clone(
    runner: Any,
    text: str,
    language: str,
    ref_audio_bytes: bytes,
) -> dict:  # type: ignore[type-arg]
    """Save ref_audio bytes to temp file and call generate_voice_clone."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(ref_audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        return runner.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=tmp_path,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
