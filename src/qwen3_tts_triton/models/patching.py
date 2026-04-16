"""Monkey-patch Qwen3-TTS to replace PyTorch ops with Triton kernels."""

import logging
import types

import torch.nn as nn

from qwen3_tts_triton.kernels.fused_norm_residual import triton_fused_add_rms_norm
from qwen3_tts_triton.kernels.rms_norm import TritonRMSNorm
from qwen3_tts_triton.kernels.swiglu import triton_swiglu_forward

logger = logging.getLogger(__name__)


def _get_layer_index(name: str) -> int | None:
    """Extract layer index from a dotted module name.

    Args:
        name: Dotted module path such as ``"model.layers.5.input_layernorm"``.

    Returns:
        The integer layer index, or ``None`` if the module is not inside a
        numbered ``layers`` collection (e.g. final norm).

    Examples:
        >>> _get_layer_index("model.layers.5.input_layernorm")
        5
        >>> _get_layer_index("model.layers.27.mlp.gate_proj")
        27
        >>> _get_layer_index("model.norm")  # final norm
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _should_patch(name: str, patch_range: tuple[int, int] | None) -> bool:
    """Check whether a module should be patched based on its layer index.

    Args:
        name: Dotted module path.
        patch_range: ``(start, end)`` half-open range of layer indices to
            patch.  ``None`` means patch everything (default behaviour).

    Returns:
        ``True`` if the module should receive a Triton patch.
    """
    if patch_range is None:
        return True
    layer_idx = _get_layer_index(name)
    if layer_idx is None:
        return False  # outside any numbered layer (e.g. final norm)
    return patch_range[0] <= layer_idx < patch_range[1]


def _get_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Resolve a dotted module path to its parent module and attribute name.

    Args:
        model: Root nn.Module to traverse.
        dotted_name: Dot-separated path such as ``"layers.0.mlp"``.

    Returns:
        A tuple of ``(parent_module, child_attr_name)`` where
        ``getattr(parent_module, child_attr_name)`` yields the target module.
    """
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    return model.get_submodule(parts[0]), parts[1]


def _replace_rms_norm(model: nn.Module, name: str, old: nn.Module) -> None:
    """Swap an RMSNorm module in-place for TritonRMSNorm, reusing weights.

    The replacement shares the original weight parameter directly so no
    extra memory is allocated.

    Args:
        model: Root nn.Module that owns the submodule.
        name: Dotted name of the submodule to replace
            (e.g. ``"layers.0.input_layernorm"``).
        old: Existing RMSNorm module whose weight and epsilon will be reused.
    """
    parent, attr = _get_parent(model, name)
    hidden_size = old.weight.shape[0]
    eps = getattr(old, "variance_epsilon", getattr(old, "eps", 1e-6))

    new_norm = TritonRMSNorm(hidden_size, eps=eps)
    new_norm.weight = old.weight  # share parameter, no copy
    setattr(parent, attr, new_norm)


def _patch_mlp_forward(mlp: nn.Module) -> None:
    """Replace MLP forward method to use the fused Triton SwiGLU kernel.

    Monkey-patches ``mlp.forward`` with a closure that calls
    ``triton_swiglu_forward(gate, up)`` instead of the default
    ``silu(gate) * up`` sequence, eliminating the intermediate tensor.

    Args:
        mlp: MLP module with ``gate_proj``, ``up_proj``, and ``down_proj``
            linear layers.
    """

    def _forward(self: nn.Module, x):  # type: ignore[override]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(triton_swiglu_forward(gate, up))

    mlp.forward = types.MethodType(_forward, mlp)


def _patch_decoder_layer_forward(layer: nn.Module) -> None:
    """Replace decoder layer forward to fuse residual add and post-attention norm.

    Monkey-patches ``layer.forward`` so that the residual addition and
    ``post_attention_layernorm`` are executed as a single fused Triton kernel
    (``triton_fused_add_rms_norm``), reducing memory traffic versus two
    separate operations.

    Args:
        layer: Transformer decoder layer with ``input_layernorm``,
            ``post_attention_layernorm``, ``self_attn``, and ``mlp``
            attributes.
    """

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # FUSED: residual add + post_attention_layernorm (1 kernel)
        norm_weight = self.post_attention_layernorm.weight
        norm_eps = getattr(
            self.post_attention_layernorm,
            "eps",
            getattr(self.post_attention_layernorm, "variance_epsilon", 1e-6),
        )
        hidden_states, residual = triton_fused_add_rms_norm(
            hidden_states, residual, norm_weight, norm_eps
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    layer.forward = types.MethodType(_forward, layer)


def _is_rms_norm(module: nn.Module) -> bool:
    """Check if module is an RMSNorm variant with patchable weight."""
    return "RMSNorm" in type(module).__name__ and hasattr(module, "weight")


def _is_swiglu_mlp(module: nn.Module) -> bool:
    """Check if module is a SwiGLU MLP (gate/up/down projections)."""
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
    )


def _is_decoder_layer(module: nn.Module) -> bool:
    """Check if module is a decoder layer with fusible norm+residual."""
    return (
        hasattr(module, "input_layernorm")
        and hasattr(module, "post_attention_layernorm")
        and hasattr(module, "self_attn")
        and hasattr(module, "mlp")
    )


def _validate_patch_range(patch_range: tuple[int, int] | None) -> None:
    """Validate patch_range argument."""
    if patch_range is not None:
        start, end = patch_range
        if start < 0 or end <= start:
            raise ValueError(
                f"patch_range must satisfy 0 <= start < end, got ({start}, {end})"
            )


def _log_patch_summary(
    patch_range: tuple[int, int] | None,
    norm_count: int,
    mlp_count: int,
    fused_count: int,
) -> None:
    """Log patching summary."""
    if patch_range is None:
        logger.info(
            "Triton patching: %d RMSNorm, %d MLP, %d FusedNormResidual",
            norm_count,
            mlp_count,
            fused_count,
        )
    else:
        logger.info(
            "[Partial] Layers [%d, %d): %d RMSNorm, %d MLP, %d FusedNormResidual",
            patch_range[0],
            patch_range[1],
            norm_count,
            mlp_count,
            fused_count,
        )


def apply_triton_kernels(
    model: nn.Module,
    enable_fused_norm: bool = True,
    patch_range: tuple[int, int] | None = None,
) -> None:
    """Replace PyTorch ops with Triton kernels in Qwen3-TTS model.

    Patches applied:
      1. RMSNorm → TritonRMSNorm
      2. MLP activation → fused SwiGLU kernel
      3. residual + post_attn_norm → fused Triton kernel (if enable_fused_norm)

    Args:
        model: The nn.Module to patch.
        enable_fused_norm: Enable fused norm+residual kernel. Default True.
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch.  ``None`` (default) patches **all** layers,
            preserving the original behaviour.  For example,
            ``(0, 20)`` patches layers 0–19 with Triton and leaves layers
            20–27 (plus the final norm) as original PyTorch.

    Raises:
        ValueError: If *patch_range* is invalid (start >= end or negative).
    """
    _validate_patch_range(patch_range)

    norm_count = 0
    mlp_count = 0
    fused_count = 0
    modules = list(model.named_modules())

    for name, module in modules:
        if _is_rms_norm(module) and _should_patch(name, patch_range):
            _replace_rms_norm(model, name, module)
            norm_count += 1
        if _is_swiglu_mlp(module) and _should_patch(name, patch_range):
            _patch_mlp_forward(module)
            mlp_count += 1

    if enable_fused_norm:
        for _name, module in modules:
            if _is_decoder_layer(module) and _should_patch(_name, patch_range):
                _patch_decoder_layer_forward(module)
                fused_count += 1

    _log_patch_summary(patch_range, norm_count, mlp_count, fused_count)


def find_patchable_model(model: object) -> nn.Module:
    """Find the underlying nn.Module from a model wrapper.

    Qwen3TTSModel and FasterQwen3TTS wrap a transformers model internally.
    This searches for the nn.Module that contains patchable layers.

    Args:
        model: A model object (may or may not be an nn.Module).

    Returns:
        The underlying nn.Module suitable for Triton patching.

    Raises:
        RuntimeError: If no nn.Module can be found inside the wrapper.
    """
    if isinstance(model, nn.Module):
        return model

    # Search common attribute names for the internal model
    candidates = ["model", "transformer", "talker", "_model", "llm"]
    for attr in candidates:
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            logger.info("Found patchable model at .%s", attr)
            return inner

    # Walk all attributes looking for nn.Module
    for attr in dir(model):
        if attr.startswith("_"):
            continue
        val = getattr(model, attr, None)
        if isinstance(val, nn.Module):
            logger.info("Found patchable model at .%s", attr)
            return val

    raise RuntimeError("Cannot find nn.Module inside the model wrapper.")
