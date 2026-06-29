"""Tier 3 Audio tab — listen to the WAVs generated during Tier 3 evaluation.

Unlike the curated *Audio Samples* tab (``assets/audio_samples/``), this tab
plays back the exact audio produced by ``make eval-fast`` / ``make eval-full``
(stored under ``benchmark/output/eval/``) and pairs every clip with its
measured CER / UTMOS / ASR transcript from the matching ``tier3_*_multi.json``.

Audio for a result is resolved from the ``wav_dir`` recorded in the JSON, with
a fallback to the conventional ``multi_<mode>[_b<batch_size>]`` directory so
older result files still work.
"""

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.i18n import t

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _PROJECT_ROOT / "benchmark" / "results"
OUTPUTS_DIR = _PROJECT_ROOT / "benchmark" / "output" / "eval"

# CER at/above which a clip is flagged as a likely catastrophic generation.
_HIGH_CER = 0.3


def render_tier3_audio_tab() -> None:
    """Render the Tier 3 generated-audio playback tab."""
    results = _discover_results()
    if not results:
        st.info(t("tier3_audio.no_data"))
        return

    st.caption(t("tier3_audio.intro"))

    labels = [r["label"] for r in results]
    choice = st.selectbox(
        t("tier3_audio.select_result"),
        list(range(len(results))),
        format_func=lambda i: labels[i],
    )
    _render_result(results[choice])


# ── Discovery ──────────────────────────────────────────────────────


def _discover_results() -> list[dict[str, Any]]:
    """Find ``tier3_*_multi.json`` files that carry per-clip sample data."""
    found: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("tier3_*_multi.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "ref_samples" not in data or "ref_runner" not in data:
            continue
        found.append({"path": path, "data": data, "label": _label_for(path, data)})
    return found


def _label_for(path: Path, data: dict[str, Any]) -> str:
    """Human-readable label for a result file in the selector."""
    mode = data.get("mode", "?")
    batch = data.get("batch_size") or 1
    kind = f"Batched B={batch}" if batch > 1 else "Single-clip"
    n_sent = data.get("num_sentences", "?")
    runs = data.get("runs_per_sentence", "?")
    return f"{kind} · {mode} (n={n_sent}×{runs}) — {path.name}"


# ── Result rendering ───────────────────────────────────────────────


def _render_result(item: dict[str, Any]) -> None:
    """Render controls and the per-sentence audio grid for one result."""
    data = item["data"]
    st.caption(t("tier3_audio.result_note"))

    index = _index_samples(data)
    ref_runner = data["ref_runner"]
    runners = [ref_runner] + list(data.get("opt_runners", []))
    wav_root = _wav_root(data)
    run, lang, only_high = _render_controls(data)

    sentence_ids = sorted({key[0] for key in index.get(ref_runner, {})})
    shown = 0
    for sent_id in sentence_ids:
        ref_sample = index.get(ref_runner, {}).get((sent_id, run))
        if ref_sample is None:
            continue
        if lang != "All" and ref_sample.get("language") != lang:
            continue
        if only_high and not _any_high_cer(index, runners, sent_id, run):
            continue
        _render_sentence(sent_id, run, ref_sample, runners, index, wav_root, ref_runner)
        shown += 1

    if shown == 0:
        st.warning(t("tier3_audio.no_matches"))


def _render_controls(data: dict[str, Any]) -> tuple[int, str, bool]:
    """Render run / language / high-CER controls; return their values."""
    runs = int(data.get("runs_per_sentence", 1) or 1)
    languages = ["All", *data.get("languages", [])]
    col_run, col_lang, col_cer = st.columns(3)
    with col_run:
        run = st.selectbox(
            t("tier3_audio.select_run"),
            list(range(runs)),
            format_func=lambda r: f"r{r}",
        )
    with col_lang:
        lang = st.selectbox(t("tier3_audio.filter_language"), languages)
    with col_cer:
        only_high = st.checkbox(t("tier3_audio.only_high_cer", threshold=_HIGH_CER))
    return int(run), str(lang), bool(only_high)


def _render_sentence(
    sent_id: int,
    run: int,
    ref_sample: dict[str, Any],
    runners: list[str],
    index: dict[str, dict[tuple[int, int], dict[str, Any]]],
    wav_root: Path,
    ref_runner: str,
) -> None:
    """Render one sentence with every runner's audio side by side."""
    lang = ref_sample.get("language", "")
    text = ref_sample.get("text", "")
    st.markdown(f"**{t('tier3_audio.sentence', idx=sent_id)}** · `{lang}` — {text}")

    cols = st.columns(len(runners))
    for col, name in zip(cols, runners):
        with col:
            sample = index.get(name, {}).get((sent_id, run))
            suffix = f" ({t('tier3_audio.reference')})" if name == ref_runner else ""
            st.markdown(f"**{name}**{suffix}")
            if sample is None:
                st.caption(t("common.n_a"))
                continue
            wav_path = wav_root / f"{name}_{sent_id:03d}_r{run}.wav"
            if wav_path.exists():
                st.audio(str(wav_path))
            else:
                st.caption(t("tier3_audio.missing_audio"))
            _render_metrics(sample)
    st.markdown("---")


def _render_metrics(sample: dict[str, Any]) -> None:
    """Render CER / UTMOS / transcript caption for one clip."""
    cer = sample.get("cer")
    utmos = sample.get("utmos")
    warn = "⚠️ " if cer is not None and cer >= _HIGH_CER else ""
    cer_str = f"{cer:.3f}" if cer is not None else "—"
    utmos_str = f"{utmos:.2f}" if utmos is not None else "—"
    st.caption(f"{warn}CER {cer_str} · UTMOS {utmos_str}")
    transcript = sample.get("transcript")
    if transcript:
        st.caption(t("tier3_audio.transcript", text=transcript[:80]))


# ── Helpers ────────────────────────────────────────────────────────


def _index_samples(
    data: dict[str, Any],
) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    """Index every runner's samples by ``(sentence_idx, run)``."""
    index: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    ref_runner = data["ref_runner"]
    index[ref_runner] = {
        (s["sentence_idx"], s["run"]): s for s in data.get("ref_samples", [])
    }
    for name, samples in data.get("opt_samples", {}).items():
        index[name] = {(s["sentence_idx"], s["run"]): s for s in samples}
    return index


def _wav_root(data: dict[str, Any]) -> Path:
    """Resolve the directory holding this result's WAV files."""
    recorded = data.get("wav_dir")
    if recorded and Path(recorded).is_dir():
        return Path(recorded)
    mode = data.get("mode", "fast")
    batch = data.get("batch_size") or 1
    subdir = f"multi_{mode}" if batch <= 1 else f"multi_{mode}_b{batch}"
    return OUTPUTS_DIR / subdir


def _any_high_cer(
    index: dict[str, dict[tuple[int, int], dict[str, Any]]],
    runners: list[str],
    sent_id: int,
    run: int,
) -> bool:
    """True if any runner's clip for this sentence/run has high CER."""
    for name in runners:
        sample = index.get(name, {}).get((sent_id, run))
        if sample and (cer := sample.get("cer")) is not None and cer >= _HIGH_CER:
            return True
    return False
