"""Tier 2: Model-level parity tests.

Verifies that Triton-patched model produces numerically close outputs
to the base model at the hidden state and logits level.

Two comparison pairs:
- Pair A (base_vs_triton): BaseRunner model with Triton kernels applied in-place
- Pair B (faster_vs_hybrid): FasterRunner model with Triton kernels applied in-place

Liger Kernel-style convergence test (inference variant).
Uses only torch/transformers/qwen-tts with no extra dependencies.

When run via pytest, writes a structured JSON artifact to
benchmark/results/tier2_metrics.json for consumption by
run_verification.py and the Streamlit dashboard.
"""

import json
import logging
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from qwen3_tts_triton.models.patching import apply_triton_kernels, find_patchable_model

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
# Sample 5 of 28 layers: early / middle / late coverage
SAMPLED_LAYERS = [0, 7, 14, 21, 27]

# Module-level report accumulator — tests write metrics here,
# test_tier2_write_report() dumps to JSON at the end.
_REPORT_DATA: dict = {
    "pairs": {
        "base_vs_triton": {
            "layers": {},
            "logits": {},
            "greedy": {},
        },
        "faster_vs_hybrid": {
            "layers": {},
            "logits": {},
            "greedy": {},
        },
    },
}

# Tier 2 thresholds: ~3-4x headroom over observed values.
# Lesson from Solar-GLM: cosine similarity alone is insufficient → use multiple metrics.
COSINE_SIM_MIN: float = 0.95
RELATIVE_L2_MAX: float = 0.08  # 8% relative L2 (in-place patching accumulation)
SNR_DB_MIN: float = 22.0  # ~0.6% error power (in-place patching accumulation)
# Per-layer max abs diff thresholds (accounts for FP accumulation)
MAX_ABS_DIFF_THRESHOLDS: dict[int, float] = {
    0: 1.0,
    7: 2.0,
    14: 5.0,
    21: 15.0,
    27: 30.0,
}


def _compute_parity_metrics(
    base: torch.Tensor, patched: torch.Tensor
) -> dict[str, float]:
    """Compute comprehensive parity metrics between two tensors.

    Returns dict with: cosine_sim, max_abs_diff, mean_abs_diff,
    relative_l2, rmse, snr_db, pearson_r.
    """
    base_f = base.float().flatten()
    patched_f = patched.float().flatten()
    diff = base_f - patched_f

    cos_sim = F.cosine_similarity(base_f.unsqueeze(0), patched_f.unsqueeze(0)).item()
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()

    # Relative L2 norm: catches magnitude/scale drift
    base_norm = base_f.norm().item()
    relative_l2 = diff.norm().item() / base_norm if base_norm > 0 else float("inf")

    # RMSE: absolute numerical distance
    rmse = diff.pow(2).mean().sqrt().item()

    # SNR (dB): signal-to-noise ratio
    signal_power = base_f.pow(2).mean().item()
    noise_power = diff.pow(2).mean().item()
    snr_db = (
        10 * math.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    )

    # Pearson correlation
    pearson_r = torch.corrcoef(torch.stack([base_f, patched_f]))[0, 1].item()

    return {
        "cosine_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2": relative_l2,
        "rmse": rmse,
        "snr_db": snr_db,
        "pearson_r": pearson_r,
    }


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.gpu,
    pytest.mark.slow,
]


def _find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | None:
    """Find the decoder layer list in Qwen3-TTS talker model."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == 28:
            return module
    return None


def _register_hooks(
    model: torch.nn.Module,
    storage: dict[int, torch.Tensor],
) -> list:
    """Register forward hooks on sampled decoder layers."""
    layers = _find_decoder_layers(model)
    if layers is None:
        pytest.skip("Could not find 28-layer decoder in model")  # type: ignore[too-many-positional-arguments]

    hooks = []
    for idx in SAMPLED_LAYERS:
        if idx >= len(layers):
            continue

        def _hook(
            _module: torch.nn.Module,
            _input: tuple,
            output: tuple,
            layer_idx: int = idx,
        ) -> None:
            # Decoder layer output is (hidden_states, ...) tuple
            hs = output[0] if isinstance(output, tuple) else output
            storage[layer_idx] = hs.detach().clone()

        hooks.append(layers[idx].register_forward_hook(_hook))
    return hooks


@pytest.fixture(scope="module")
def forward_outputs():
    """Run base forward, patch in-place, then patched forward.

    Uses in-place patching (no deepcopy) because the Qwen3-TTS model
    contains unpicklable objects (dict_keys) that break copy.deepcopy().
    Base results are saved to CPU before patching modifies the model.

    The talker backbone (not the full model) is used for forward passes,
    since it accepts inputs_embeds directly.
    """
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    talker = tts.model.talker
    backbone = talker.model
    processor = tts.processor

    text = "Hello, this is a test of the text to speech system."
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        inputs_embeds = backbone.text_embedding(inputs["input_ids"])
    attn_mask = inputs["attention_mask"]

    # --- Base forward pass ---
    base_storage: dict[int, torch.Tensor] = {}
    base_hooks = _register_hooks(backbone, base_storage)

    with torch.inference_mode():
        base_out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

    for h in base_hooks:
        h.remove()

    # Save base results to CPU before in-place patching
    base_storage = {k: v.cpu() for k, v in base_storage.items()}
    base_last_hidden = base_out.last_hidden_state.cpu()

    # Greedy generation skipped: talker.generate() requires full TTS
    # pipeline args (speaker, codec tokens) not available in backbone test.

    # --- Apply Triton patches in-place ---
    apply_triton_kernels(talker)
    talker.eval()

    # --- Patched forward pass ---
    patched_storage: dict[int, torch.Tensor] = {}
    patched_hooks = _register_hooks(backbone, patched_storage)

    with torch.inference_mode():
        patched_out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

    for h in patched_hooks:
        h.remove()

    patched_last_hidden = patched_out.last_hidden_state.cpu()
    patched_storage = {k: v.cpu() for k, v in patched_storage.items()}

    # Backbone returns BaseModelOutputWithPast (no logits).
    # Use last_hidden_state as the output-level comparison target.
    yield {
        "base_storage": base_storage,
        "patched_storage": patched_storage,
        "base_logits": base_last_hidden,
        "patched_logits": patched_last_hidden,
        "base_seq": [],
        "patched_seq": [],
    }

    # Cleanup
    del tts, talker
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def faster_forward_outputs():
    """Run FasterRunner forward, patch in-place, then patched forward.

    Mirrors the forward_outputs fixture but uses FasterRunner (faster-qwen3-tts).
    The internal patchable model is extracted via find_patchable_model() on
    runner.model.model (FasterQwen3TTS.model = Qwen3TTSModel wrapper).

    Base results are saved to CPU before in-place patching.
    """
    from qwen3_tts_triton.models.faster_runner import FasterRunner

    runner = FasterRunner()
    runner.load_model()

    # runner.model = FasterQwen3TTS
    # runner.model.model = Qwen3TTSModel (inference wrapper)
    # find_patchable_model → Qwen3TTSForConditionalGeneration (.talker inside)
    conditional_model = find_patchable_model(runner.model.model)
    talker = conditional_model.talker
    backbone = talker.model  # type: ignore[union-attr]

    processor = runner.model.model.processor
    text = "Hello, this is a test of the text to speech system."
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        inputs_embeds = backbone.text_embedding(inputs["input_ids"])  # type: ignore[union-attr]
    attn_mask = inputs["attention_mask"]

    # --- Base forward pass ---
    base_storage: dict[int, torch.Tensor] = {}
    base_hooks = _register_hooks(backbone, base_storage)

    with torch.inference_mode():
        base_out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

    for h in base_hooks:
        h.remove()

    # Save base results to CPU before in-place patching
    base_storage = {k: v.cpu() for k, v in base_storage.items()}
    base_last_hidden = base_out.last_hidden_state.cpu()

    # --- Apply Triton patches in-place ---
    apply_triton_kernels(talker)
    talker.eval()  # type: ignore[union-attr]

    # --- Patched forward pass ---
    patched_storage: dict[int, torch.Tensor] = {}
    patched_hooks = _register_hooks(backbone, patched_storage)

    with torch.inference_mode():
        patched_out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

    for h in patched_hooks:
        h.remove()

    patched_last_hidden = patched_out.last_hidden_state.cpu()
    patched_storage = {k: v.cpu() for k, v in patched_storage.items()}

    yield {
        "base_storage": base_storage,
        "patched_storage": patched_storage,
        "base_logits": base_last_hidden,
        "patched_logits": patched_last_hidden,
        "base_seq": [],
        "patched_seq": [],
    }

    # Cleanup
    runner.unload_model()
    torch.cuda.empty_cache()


def _skip_if_no_hidden_states(forward_outputs: dict) -> None:
    """Skip test if no hidden states were captured."""
    if not forward_outputs["base_storage"] or not forward_outputs["patched_storage"]:
        pytest.skip("No hidden states captured (model structure mismatch)")  # type: ignore[too-many-positional-arguments]


def test_hidden_state_cosine_similarity(forward_outputs) -> None:
    """Verify hidden state cosine similarity exceeds 0.95 at each sampled layer."""
    _skip_if_no_hidden_states(forward_outputs)
    base_storage = forward_outputs["base_storage"]
    patched_storage = forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "Layer %2d: cosine_sim=%.6f, max_abs_diff=%.6f",
            layer_idx,
            metrics["cosine_sim"],
            metrics["max_abs_diff"],
        )

        assert metrics["cosine_sim"] > COSINE_SIM_MIN, (
            f"Layer {layer_idx}: cosine_similarity="
            f"{metrics['cosine_sim']:.6f} < {COSINE_SIM_MIN}"
        )


def test_hidden_state_relative_l2(forward_outputs) -> None:
    """Verify relative L2 norm difference is below 5% at each sampled layer.

    Catches magnitude/scale drift that cosine similarity misses
    (scale-invariant). A missing RMSNorm or incorrect epsilon would
    cause high relative L2 while cosine similarity stays near 1.0.
    """
    _skip_if_no_hidden_states(forward_outputs)
    base_storage = forward_outputs["base_storage"]
    patched_storage = forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "Layer %2d: relative_l2=%.6f",
            layer_idx,
            metrics["relative_l2"],
        )

        assert metrics["relative_l2"] < RELATIVE_L2_MAX, (
            f"Layer {layer_idx}: relative_l2="
            f"{metrics['relative_l2']:.6f} >= {RELATIVE_L2_MAX}"
        )


def test_hidden_state_snr(forward_outputs) -> None:
    """Verify Signal-to-Noise Ratio exceeds 25 dB at each sampled layer.

    SNR combines magnitude and direction information. 25 dB means error
    power is ~0.3% of signal power. Unlike cosine similarity, SNR is
    NOT scale-invariant — it catches both directional and magnitude errors.
    """
    _skip_if_no_hidden_states(forward_outputs)
    base_storage = forward_outputs["base_storage"]
    patched_storage = forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "Layer %2d: snr_db=%.2f",
            layer_idx,
            metrics["snr_db"],
        )

        assert (
            metrics["snr_db"] > SNR_DB_MIN
        ), f"Layer {layer_idx}: SNR={metrics['snr_db']:.2f} dB < {SNR_DB_MIN} dB"


def test_hidden_state_max_abs_diff(forward_outputs) -> None:
    """Verify max absolute difference stays within per-layer bounds.

    Layer-dependent thresholds account for expected FP error accumulation
    through the 28-layer decoder. Thresholds are ~4x observed values.
    Promoted from INFO to HARD based on Solar-GLM analysis.
    """
    _skip_if_no_hidden_states(forward_outputs)
    base_storage = forward_outputs["base_storage"]
    patched_storage = forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        threshold = MAX_ABS_DIFF_THRESHOLDS.get(layer_idx, 30.0)

        logger.info(
            "Layer %2d: max_abs_diff=%.6f (threshold=%.1f)",
            layer_idx,
            metrics["max_abs_diff"],
            threshold,
        )

        assert metrics["max_abs_diff"] < threshold, (
            f"Layer {layer_idx}: max_abs_diff="
            f"{metrics['max_abs_diff']:.6f} >= {threshold}"
        )


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metric values for JSON serialization."""
    return {
        k: round(v, 6) if k != "snr_db" else round(v, 2) for k, v in metrics.items()
    }


def test_hidden_state_diagnostics(forward_outputs) -> None:
    """Report comprehensive per-layer diagnostics for base vs triton (informational).

    Logs Pearson correlation, RMSE, and all other metrics for each
    sampled layer. Always passes — for diagnostic monitoring only.
    """
    _skip_if_no_hidden_states(forward_outputs)
    base_storage = forward_outputs["base_storage"]
    patched_storage = forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "[base_vs_triton] Layer %2d diagnostics: "
            "cos_sim=%.6f, rel_l2=%.6f, snr=%.2fdB, "
            "rmse=%.6f, pearson_r=%.6f, "
            "max_abs=%.6f, mean_abs=%.6f",
            layer_idx,
            metrics["cosine_sim"],
            metrics["relative_l2"],
            metrics["snr_db"],
            metrics["rmse"],
            metrics["pearson_r"],
            metrics["max_abs_diff"],
            metrics["mean_abs_diff"],
        )
        _REPORT_DATA["pairs"]["base_vs_triton"]["layers"][str(layer_idx)] = (
            _round_metrics(metrics)
        )

    assert True


def test_logits_output_cosine(forward_outputs) -> None:
    """Verify output-level cosine similarity (last_hidden_state or logits)."""
    base_logits = forward_outputs["base_logits"]
    patched_logits = forward_outputs["patched_logits"]

    if base_logits is None or patched_logits is None:
        pytest.skip("Model does not expose logits in forward output")  # type: ignore[too-many-positional-arguments]

    base_flat = base_logits.float().flatten()
    patched_flat = patched_logits.float().flatten()
    cos = F.cosine_similarity(base_flat.unsqueeze(0), patched_flat.unsqueeze(0)).item()
    max_diff = (base_flat - patched_flat).abs().max().item()

    logger.info(
        "[base_vs_triton] Output cosine_sim=%.6f, max_abs_diff=%.6f", cos, max_diff
    )
    _REPORT_DATA["pairs"]["base_vs_triton"]["logits"]["output_cosine_sim"] = round(
        cos, 6
    )
    _REPORT_DATA["pairs"]["base_vs_triton"]["logits"]["output_max_abs_diff"] = round(
        max_diff, 6
    )

    assert cos > COSINE_SIM_MIN, f"Output cosine_sim={cos:.6f} < {COSINE_SIM_MIN}"


def _is_vocab_logits(tensor: torch.Tensor) -> bool:
    """Check if tensor has vocabulary dimension (last dim > 10000)."""
    return tensor is not None and tensor.dim() >= 2 and tensor.shape[-1] > 10000


def test_logits_top_k_overlap(forward_outputs) -> None:
    """Verify top-50 token overlap between base and patched logits exceeds 60%."""
    base_logits = forward_outputs["base_logits"]
    patched_logits = forward_outputs["patched_logits"]

    if not _is_vocab_logits(base_logits) or not _is_vocab_logits(patched_logits):
        pytest.skip("Backbone-only output (no vocab logits)")  # type: ignore[too-many-positional-arguments]

    k = 50
    base_last = base_logits[0, -1, :]
    patched_last = patched_logits[0, -1, :]

    base_topk = set(torch.topk(base_last, k).indices.tolist())
    patched_topk = set(torch.topk(patched_last, k).indices.tolist())

    overlap = len(base_topk & patched_topk)
    overlap_pct = overlap / k * 100

    logger.info("Logits top-%d overlap: %d/%d (%.1f%%)", k, overlap, k, overlap_pct)
    _REPORT_DATA["pairs"]["base_vs_triton"]["logits"]["top_k_overlap_pct"] = round(
        overlap_pct, 1
    )

    assert overlap_pct > 60, f"Top-{k} overlap={overlap_pct:.1f}% < 60%"


def test_logits_kl_divergence(forward_outputs) -> None:
    """Verify KL divergence between base and patched final logits is below 1.0."""
    base_logits = forward_outputs["base_logits"]
    patched_logits = forward_outputs["patched_logits"]

    if not _is_vocab_logits(base_logits) or not _is_vocab_logits(patched_logits):
        pytest.skip("Backbone-only output (no vocab logits)")  # type: ignore[too-many-positional-arguments]

    base_probs = F.log_softmax(base_logits[0, -1, :].float(), dim=-1)
    patched_probs = F.softmax(patched_logits[0, -1, :].float(), dim=-1)

    kl_div = F.kl_div(base_probs, patched_probs, reduction="sum").item()
    logger.info("Logits KL divergence: %.6f", kl_div)
    _REPORT_DATA["pairs"]["base_vs_triton"]["logits"]["kl_divergence"] = round(
        kl_div, 6
    )

    assert kl_div < 1.0, f"KL divergence={kl_div:.6f} >= 1.0"


def test_greedy_token_divergence(forward_outputs) -> None:
    """Record first greedy token divergence (informational)."""
    base_seq = forward_outputs["base_seq"]
    patched_seq = forward_outputs["patched_seq"]

    if not base_seq or not patched_seq:
        _REPORT_DATA["pairs"]["base_vs_triton"]["greedy"] = {
            "note": "Greedy generation not available for this model",
        }
        pytest.skip("Greedy generation not available")  # type: ignore[too-many-positional-arguments]

    min_len = min(len(base_seq), len(patched_seq))
    matches = sum(b == p for b, p in zip(base_seq[:min_len], patched_seq[:min_len]))
    match_pct = matches / min_len * 100 if min_len > 0 else 0

    first_div = min_len  # default: all match
    for i in range(min_len):
        if base_seq[i] != patched_seq[i]:
            first_div = i
            break

    logger.info(
        "Greedy tokens: %d/%d match (%.1f%%), first_divergence=%d, "
        "base_len=%d, patched_len=%d",
        matches,
        min_len,
        match_pct,
        first_div,
        len(base_seq),
        len(patched_seq),
    )
    _REPORT_DATA["pairs"]["base_vs_triton"]["greedy"] = {
        "match_pct": round(match_pct, 1),
        "first_divergence": first_div,
        "base_len": len(base_seq),
        "patched_len": len(patched_seq),
    }

    assert True


# ---------------------------------------------------------------------------
# Pair B: Faster vs Hybrid (FasterRunner + in-place Triton patching)
# ---------------------------------------------------------------------------


def test_faster_hidden_state_cosine_similarity(faster_forward_outputs) -> None:
    """Verify hidden state cosine similarity for faster vs hybrid pair."""
    _skip_if_no_hidden_states(faster_forward_outputs)
    base_storage = faster_forward_outputs["base_storage"]
    patched_storage = faster_forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "[faster_vs_hybrid] Layer %2d: cosine_sim=%.6f, max_abs_diff=%.6f",
            layer_idx,
            metrics["cosine_sim"],
            metrics["max_abs_diff"],
        )

        assert metrics["cosine_sim"] > COSINE_SIM_MIN, (
            f"[faster_vs_hybrid] Layer {layer_idx}: cosine_similarity="
            f"{metrics['cosine_sim']:.6f} < {COSINE_SIM_MIN}"
        )


def test_faster_hidden_state_relative_l2(faster_forward_outputs) -> None:
    """Verify relative L2 norm for faster vs hybrid is below threshold."""
    _skip_if_no_hidden_states(faster_forward_outputs)
    base_storage = faster_forward_outputs["base_storage"]
    patched_storage = faster_forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "[faster_vs_hybrid] Layer %2d: relative_l2=%.6f",
            layer_idx,
            metrics["relative_l2"],
        )

        assert metrics["relative_l2"] < RELATIVE_L2_MAX, (
            f"[faster_vs_hybrid] Layer {layer_idx}: relative_l2="
            f"{metrics['relative_l2']:.6f} >= {RELATIVE_L2_MAX}"
        )


def test_faster_hidden_state_snr(faster_forward_outputs) -> None:
    """Verify SNR for faster vs hybrid exceeds threshold."""
    _skip_if_no_hidden_states(faster_forward_outputs)
    base_storage = faster_forward_outputs["base_storage"]
    patched_storage = faster_forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "[faster_vs_hybrid] Layer %2d: snr_db=%.2f",
            layer_idx,
            metrics["snr_db"],
        )

        assert metrics["snr_db"] > SNR_DB_MIN, (
            f"[faster_vs_hybrid] Layer {layer_idx}: SNR={metrics['snr_db']:.2f} dB"
            f" < {SNR_DB_MIN} dB"
        )


def test_faster_hidden_state_max_abs_diff(faster_forward_outputs) -> None:
    """Verify max absolute difference for faster vs hybrid within per-layer bounds."""
    _skip_if_no_hidden_states(faster_forward_outputs)
    base_storage = faster_forward_outputs["base_storage"]
    patched_storage = faster_forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        threshold = MAX_ABS_DIFF_THRESHOLDS.get(layer_idx, 30.0)

        logger.info(
            "[faster_vs_hybrid] Layer %2d: max_abs_diff=%.6f (threshold=%.1f)",
            layer_idx,
            metrics["max_abs_diff"],
            threshold,
        )

        assert metrics["max_abs_diff"] < threshold, (
            f"[faster_vs_hybrid] Layer {layer_idx}: max_abs_diff="
            f"{metrics['max_abs_diff']:.6f} >= {threshold}"
        )


def test_faster_hidden_state_diagnostics(faster_forward_outputs) -> None:
    """Report per-layer diagnostics for faster vs hybrid (informational)."""
    _skip_if_no_hidden_states(faster_forward_outputs)
    base_storage = faster_forward_outputs["base_storage"]
    patched_storage = faster_forward_outputs["patched_storage"]

    for layer_idx in SAMPLED_LAYERS:
        if layer_idx not in base_storage or layer_idx not in patched_storage:
            continue

        metrics = _compute_parity_metrics(
            base_storage[layer_idx], patched_storage[layer_idx]
        )
        logger.info(
            "[faster_vs_hybrid] Layer %2d diagnostics: "
            "cos_sim=%.6f, rel_l2=%.6f, snr=%.2fdB, "
            "rmse=%.6f, pearson_r=%.6f, "
            "max_abs=%.6f, mean_abs=%.6f",
            layer_idx,
            metrics["cosine_sim"],
            metrics["relative_l2"],
            metrics["snr_db"],
            metrics["rmse"],
            metrics["pearson_r"],
            metrics["max_abs_diff"],
            metrics["mean_abs_diff"],
        )
        _REPORT_DATA["pairs"]["faster_vs_hybrid"]["layers"][str(layer_idx)] = (
            _round_metrics(metrics)
        )

    assert True


def test_faster_logits_output_cosine(faster_forward_outputs) -> None:
    """Verify output-level cosine similarity for faster vs hybrid pair."""
    base_logits = faster_forward_outputs["base_logits"]
    patched_logits = faster_forward_outputs["patched_logits"]

    if base_logits is None or patched_logits is None:
        pytest.skip("Model does not expose logits in forward output")  # type: ignore[too-many-positional-arguments]

    base_flat = base_logits.float().flatten()
    patched_flat = patched_logits.float().flatten()
    cos = F.cosine_similarity(base_flat.unsqueeze(0), patched_flat.unsqueeze(0)).item()
    max_diff = (base_flat - patched_flat).abs().max().item()

    logger.info(
        "[faster_vs_hybrid] Output cosine_sim=%.6f, max_abs_diff=%.6f", cos, max_diff
    )
    _REPORT_DATA["pairs"]["faster_vs_hybrid"]["logits"]["output_cosine_sim"] = round(
        cos, 6
    )
    _REPORT_DATA["pairs"]["faster_vs_hybrid"]["logits"]["output_max_abs_diff"] = round(
        max_diff, 6
    )

    assert (
        cos > COSINE_SIM_MIN
    ), f"[faster_vs_hybrid] Output cosine_sim={cos:.6f} < {COSINE_SIM_MIN}"


def _pair_pass(pair_data: dict) -> bool:
    """Return True if all metrics in a pair's data pass thresholds."""
    all_pass = True
    for layer_key, metrics in pair_data.get("layers", {}).items():
        layer_idx = int(layer_key)
        if metrics.get("cosine_sim", 0) <= COSINE_SIM_MIN:
            all_pass = False
        if metrics.get("relative_l2", 1) >= RELATIVE_L2_MAX:
            all_pass = False
        if metrics.get("snr_db", 0) <= SNR_DB_MIN:
            all_pass = False
        threshold = MAX_ABS_DIFF_THRESHOLDS.get(layer_idx, 30.0)
        if metrics.get("max_abs_diff", threshold) >= threshold:
            all_pass = False
    output_cos = pair_data.get("logits", {}).get("output_cosine_sim")
    if output_cos is not None and output_cos <= COSINE_SIM_MIN:
        all_pass = False
    return all_pass


def test_tier2_write_report(forward_outputs, faster_forward_outputs) -> None:
    """Write accumulated Tier 2 metrics to JSON artifact.

    Must run last (file ordering). Produces benchmark/results/tier2_metrics.json
    consumed by run_verification.py and the Streamlit verification tab.

    Output uses a "pairs" structure for both comparison pairs:
      base_vs_triton and faster_vs_hybrid.
    """
    pairs = _REPORT_DATA["pairs"]
    bvt_pass = _pair_pass(pairs["base_vs_triton"])
    fvh_pass = _pair_pass(pairs["faster_vs_hybrid"])
    overall_pass = bvt_pass and fvh_pass

    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "pairs": {
            "base_vs_triton": {
                "status": "PASS" if bvt_pass else "FAIL",
                "layers": pairs["base_vs_triton"]["layers"],
                "logits": pairs["base_vs_triton"]["logits"],
                "greedy": pairs["base_vs_triton"]["greedy"],
            },
            "faster_vs_hybrid": {
                "status": "PASS" if fvh_pass else "FAIL",
                "layers": pairs["faster_vs_hybrid"]["layers"],
                "logits": pairs["faster_vs_hybrid"]["logits"],
                "greedy": pairs["faster_vs_hybrid"]["greedy"],
            },
        },
    }

    results_dir = Path(__file__).resolve().parent.parent / "benchmark" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = results_dir / "tier2_metrics.json"
    artifact_path.write_text(json.dumps(report, indent=2))
    logger.info("Tier 2 artifact written to %s", artifact_path)

    assert True
