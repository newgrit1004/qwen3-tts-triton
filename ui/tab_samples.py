"""Audio Samples tab for side-by-side comparison across inference modes.

Loads pre-generated WAV files from assets/audio_samples/ with metadata
and displays them in a filterable, side-by-side comparison layout.
"""

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.i18n import t

logger = logging.getLogger(__name__)

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio_samples"
_METADATA_PATH = _SAMPLES_DIR / "metadata.json"

_MODES = ["base", "triton", "faster", "hybrid"]
_MODE_LABELS = {
    "base": "Base (PyTorch)",
    "triton": "Triton",
    "faster": "Faster (CUDA Graph)",
    "hybrid": "Hybrid (Faster+Triton)",
}


def render_samples_tab() -> None:
    """Render the Audio Samples comparison tab."""
    metadata = _load_metadata()
    if metadata is None:
        st.info(t("samples.no_data"))
        return

    samples = metadata.get("samples", [])
    if not samples:
        st.info(t("samples.no_data"))
        return

    # --- Filters ---
    col_lang, col_type = st.columns(2)
    with col_lang:
        lang_filter = st.selectbox(
            t("samples.filter_language"),
            ["All", "Korean", "English"],
        )
    with col_type:
        type_filter = st.selectbox(
            t("samples.filter_type"),
            ["All", "Custom Voice", "Voice Cloning"],
        )

    # --- Group by text (same utterance across modes) ---
    groups = _group_by_utterance(samples, lang_filter, type_filter)

    if not groups:
        st.warning(t("samples.no_matches"))
        return

    st.caption(
        t("samples.total_count", count=len(groups)),
    )

    for idx, group in enumerate(groups):
        _render_sample_group(idx, group)


def _render_sample_group(idx: int, group: dict[str, Any]) -> None:
    """Render one utterance with all 4 modes side by side."""
    text = group["text"]
    lang = group["language_name"]
    style = group["style"]
    sample_type = group["type"]

    type_label = (
        t("samples.type_clone") if sample_type == "clone" else t("samples.type_custom")
    )
    header = f"**{idx + 1}. [{lang}] {style.capitalize()}** — {type_label}"
    st.markdown(header)

    with st.expander(f'"{text}"', expanded=idx == 0):
        cols = st.columns(len(group["modes"]))
        for col, mode in zip(cols, _MODES):
            sample = group["modes"].get(mode)
            if sample is None:
                continue
            with col:
                st.markdown(f"**{_MODE_LABELS.get(mode, mode)}**")
                audio_path = _SAMPLES_DIR / sample["file"]
                if audio_path.exists():
                    st.audio(str(audio_path))
                else:
                    st.warning(t("common.file_missing"))
                dur = sample.get("duration_s", 0)
                gen = sample.get("generation_time_s", 0)
                rtf = dur / gen if gen > 0 else 0
                st.caption(f"{gen:.1f}s | RTF {rtf:.1f}x")


def _group_by_utterance(
    samples: list[dict],
    lang_filter: str,
    type_filter: str,
) -> list[dict[str, Any]]:
    """Group samples by text so the same utterance appears once with all modes."""
    type_map = {"Custom Voice": "custom", "Voice Cloning": "clone"}
    lang_map = {"Korean": "ko", "English": "en"}

    filtered_lang = lang_map.get(lang_filter)
    filtered_type = type_map.get(type_filter)

    # Build groups keyed by (text, type, language)
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for s in samples:
        if filtered_lang and s.get("language") != filtered_lang:
            continue
        if filtered_type and s.get("type") != filtered_type:
            continue

        key = (s["text"], s["type"], s["language"])
        if key not in groups:
            groups[key] = {
                "text": s["text"],
                "type": s["type"],
                "language": s["language"],
                "language_name": s.get("language_name", s["language"]),
                "style": s.get("style", ""),
                "modes": {},
            }
        groups[key]["modes"][s["mode"]] = s

    return list(groups.values())


def _load_metadata() -> dict[str, Any] | None:
    """Load metadata.json for audio samples."""
    if not _METADATA_PATH.exists():
        return None
    try:
        return json.loads(_METADATA_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load %s", _METADATA_PATH)
        return None
