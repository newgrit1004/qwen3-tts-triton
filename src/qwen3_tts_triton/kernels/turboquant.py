"""TurboQuant KV cache quantization (Google, ICLR 2026).

Calibration-free, data-oblivious KV cache compression using PolarQuant:
random orthogonal rotation + Lloyd-Max scalar quantization.

Reference: arXiv:2504.19874 — TurboQuant: Online KV Cache Quantization
Implementation: tonbistudio/turboquant-pytorch (pure PyTorch, zero CUDA)
"""

import logging
from typing import Any, cast

import torch
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Pre-computed Lloyd-Max centroids for N(0,1).
# Symmetric around zero — only positive half stored, negative mirrored.
# Computed via iterative Lloyd-Max algorithm to convergence (tol=1e-7).
_LLOYD_MAX_CACHE: dict[int, torch.Tensor] = {}


def _compute_centroids(boundaries, n_levels):
    """Compute centroids from boundaries for N(0,1) Lloyd-Max."""
    import numpy as np

    full_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
    centroids = np.zeros(n_levels)
    for i in range(n_levels):
        a, b = full_bounds[i], full_bounds[i + 1]
        prob = norm.cdf(b) - norm.cdf(a)
        if prob < 1e-15:
            centroids[i] = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else a
        else:
            centroids[i] = (norm.pdf(a) - norm.pdf(b)) / prob
    return centroids


def _lloyd_max_codebook_np(bits: int, max_iter: int = 200, tol: float = 1e-7):
    """Compute Lloyd-Max optimal codebook for N(0,1) via scipy.

    Uses alternating boundary-centroid optimization until convergence.

    Args:
        bits: Number of quantization bits (2, 3, or 4).
        max_iter: Maximum iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        Sorted numpy array of 2**bits centroids.
    """
    import numpy as np

    n_levels = 2**bits

    # Initialize with uniform quantiles
    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    boundaries = norm.ppf(quantiles)

    for _ in range(max_iter):
        centroids = _compute_centroids(boundaries, n_levels)

        # Boundary step: midpoints between adjacent centroids
        new_boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        if np.max(np.abs(new_boundaries - boundaries)) < tol:
            boundaries = new_boundaries
            break
        boundaries = new_boundaries

    centroids = _compute_centroids(boundaries, n_levels)
    return centroids, boundaries


def lloyd_max_codebook(bits: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Get or compute Lloyd-Max codebook for N(0,1).

    Cached globally — computed once per bit-width.

    Args:
        bits: Quantization bits (2, 3, or 4).
        device: Target device.

    Returns:
        Sorted centroid tensor of shape ``(2**bits,)``.
    """
    if bits not in _LLOYD_MAX_CACHE:
        centroids_np, _ = _lloyd_max_codebook_np(bits)
        _LLOYD_MAX_CACHE[bits] = torch.from_numpy(centroids_np).float()
    return _LLOYD_MAX_CACHE[bits].to(device)


def lloyd_max_boundaries(
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Compute decision boundaries from codebook centroids.

    Boundaries are midpoints between adjacent centroids.

    Args:
        codebook: Sorted centroids ``(2**bits,)``.

    Returns:
        Boundaries tensor ``(2**bits - 1,)``.
    """
    return (codebook[:-1] + codebook[1:]) / 2.0


def generate_rotation_matrix(
    dim: int,
    layer_idx: int,
    head_idx: int,
    seed: int = 42,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic random orthogonal matrix via QR decomposition.

    Each (layer, head) pair gets a unique but reproducible rotation.

    Args:
        dim: Vector dimension (head_dim, typically 128).
        layer_idx: Decoder layer index.
        head_idx: KV head index.
        seed: Base random seed.
        device: Target device.
        dtype: Output dtype.

    Returns:
        Orthogonal matrix Q of shape ``(dim, dim)``.
    """
    combined_seed = seed + layer_idx * 1000 + head_idx
    gen = torch.Generator().manual_seed(combined_seed)
    g = torch.randn(dim, dim, generator=gen)
    q, _ = torch.linalg.qr(g)
    return q.to(device=device, dtype=dtype)


def quantize_vectors(
    x: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize vectors using PolarQuant.

    Steps: normalize → rotate → scalar quantize each coordinate.

    Args:
        x: Input tensor ``(..., dim)`` in any float dtype.
        rotation: Orthogonal matrix ``(dim, dim)``.
        codebook: Lloyd-Max centroids ``(n_levels,)``.
        boundaries: Decision boundaries ``(n_levels - 1,)``.

    Returns:
        ``(indices, scales)`` where indices is ``(..., dim)`` uint8
        and scales is ``(..., 1)`` float.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    dim = x_f.shape[-1]
    sqrt_dim = dim**0.5

    # Scale = L2 norm per vector
    scales = x_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normalized = x_f / scales

    # Rotate: x_rot = x_normalized @ R^T  (each vector rotated)
    # After rotation, coordinates of a unit vector follow
    # Beta((d-1)/2, (d-1)/2) with std ≈ 1/sqrt(d).
    # Scale by sqrt(d) so coordinates ≈ N(0,1) for the codebook.
    x_rotated = (x_normalized @ rotation.T) * sqrt_dim

    # Scalar quantize via boundary search (searchsorted)
    indices = torch.searchsorted(boundaries, x_rotated).to(torch.uint8)

    scales = scales.to(orig_dtype)
    return indices, scales


def dequantize_vectors(
    indices: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Dequantize from indices back to float vectors.

    Steps: codebook lookup → inverse rotate → rescale.

    Args:
        indices: Quantized indices ``(..., dim)`` uint8.
        scales: Per-vector scales ``(..., 1)`` float.
        rotation: Orthogonal matrix ``(dim, dim)``.
        codebook: Lloyd-Max centroids ``(n_levels,)``.

    Returns:
        Dequantized tensor ``(..., dim)`` in scales' dtype.
    """
    out_dtype = scales.dtype
    dim = indices.shape[-1]
    sqrt_dim = dim**0.5

    # Gather centroids and undo the sqrt(d) scaling from quantization
    reconstructed = codebook[indices.long()] / sqrt_dim  # (..., dim), float32

    # Inverse rotate: R^T^T = R
    reconstructed = reconstructed @ rotation

    # Rescale
    return (reconstructed * scales.float()).to(out_dtype)


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices (0–15) into 4-bit packed format.

    Two adjacent values share one byte: ``packed = (even << 4) | odd``.

    Args:
        indices: ``(..., dim)`` uint8, values in [0, 15]. dim must be even.

    Returns:
        ``(..., dim // 2)`` uint8 packed tensor.
    """
    even = indices[..., 0::2]
    odd = indices[..., 1::2]
    return ((even.to(torch.int16) << 4) | odd.to(torch.int16)).to(torch.uint8)


def unpack_4bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack 4-bit packed tensor back to uint8 indices.

    Args:
        packed: ``(..., dim // 2)`` uint8.
        dim: Original dimension (must be even).

    Returns:
        ``(..., dim)`` uint8.
    """
    even = (packed >> 4) & 0x0F
    odd = packed & 0x0F
    result = torch.empty(
        *packed.shape[:-1], dim, dtype=torch.uint8, device=packed.device
    )
    result[..., 0::2] = even
    result[..., 1::2] = odd
    return result


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices (0–7) into 3-bit packed format.

    Groups of 8 values (24 bits) are packed into 3 bytes.

    Args:
        indices: ``(..., dim)`` uint8, values in [0, 7]. dim must be
            divisible by 8.

    Returns:
        ``(..., dim * 3 // 8)`` uint8 packed tensor.
    """
    shape = indices.shape
    grouped = indices.reshape(*shape[:-1], -1, 8).to(torch.int32)
    v = grouped

    byte0 = ((v[..., 0] << 5) | (v[..., 1] << 2) | (v[..., 2] >> 1)).to(torch.uint8)
    byte1 = (
        ((v[..., 2] & 1) << 7) | (v[..., 3] << 4) | (v[..., 4] << 1) | (v[..., 5] >> 2)
    ).to(torch.uint8)
    byte2 = (((v[..., 5] & 3) << 6) | (v[..., 6] << 3) | v[..., 7]).to(torch.uint8)

    packed = torch.stack([byte0, byte1, byte2], dim=-1)
    return packed.reshape(*shape[:-1], -1)


def unpack_3bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack 3-bit packed tensor back to uint8 indices.

    Args:
        packed: ``(..., dim * 3 // 8)`` uint8.
        dim: Original dimension (must be divisible by 8).

    Returns:
        ``(..., dim)`` uint8.
    """
    shape = packed.shape
    grouped = packed.reshape(*shape[:-1], -1, 3).to(torch.int32)
    b = grouped

    v0 = (b[..., 0] >> 5) & 0x07
    v1 = (b[..., 0] >> 2) & 0x07
    v2 = ((b[..., 0] & 0x03) << 1) | ((b[..., 1] >> 7) & 0x01)
    v3 = (b[..., 1] >> 4) & 0x07
    v4 = (b[..., 1] >> 1) & 0x07
    v5 = ((b[..., 1] & 0x01) << 2) | ((b[..., 2] >> 6) & 0x03)
    v6 = (b[..., 2] >> 3) & 0x07
    v7 = b[..., 2] & 0x07

    result = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return result.reshape(*shape[:-1], dim).to(torch.uint8)


class TurboQuantKVCache:
    """KV cache with PolarQuant compression.

    Drop-in replacement for HuggingFace's ``DynamicCache``.
    Stores keys and values as quantized uint8 indices + fp16 scales,
    dequantizes on read for attention computation.

    Args:
        bits: Quantization bit-width (3 or 4). Default 4.
        num_layers: Number of decoder layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per head.
        device: Target device.
        dtype: Original model dtype for dequantized output.
    """

    def __init__(
        self,
        bits: int = 4,
        num_layers: int = 28,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bits = bits
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Codebook and boundaries (shared across all layers/heads)
        self.codebook = lloyd_max_codebook(bits, device=device)
        self.boundaries = lloyd_max_boundaries(self.codebook).to(device)

        # Rotation matrices: one per (layer, head) pair
        self.rotations: list[list[torch.Tensor]] = []
        for layer in range(num_layers):
            layer_rots = []
            for head in range(num_kv_heads):
                r = generate_rotation_matrix(
                    head_dim, layer, head, device=device, dtype=torch.float32
                )
                layer_rots.append(r)
            self.rotations.append(layer_rots)

        # Pre-stacked rotations per layer: (H, D, D) for Triton kernel
        self._rotations_stacked: list[torch.Tensor] = [
            torch.stack(self.rotations[layer]) for layer in range(num_layers)
        ]

        # Per-layer quantized storage (grown dynamically)
        _none_list = cast(list[torch.Tensor | None], [None] * num_layers)
        self._key_indices = _none_list[:]
        self._key_scales = _none_list[:]
        self._val_indices = _none_list[:]
        self._val_scales = _none_list[:]
        self._seq_lengths: list[int] = [0] * num_layers

        # Decompressed running buffers (incremental dequant — avoids O(S²))
        self._deq_keys = _none_list[:]
        self._deq_values = _none_list[:]

        rot_mem = num_layers * num_kv_heads * head_dim * head_dim * 4 / 1024**2
        logger.info(
            "TurboQuantKVCache: %d-bit, %d layers, %d heads, dim=%d "
            "(rotation matrices: %.1f MB)",
            bits,
            num_layers,
            num_kv_heads,
            head_dim,
            rot_mem,
        )

    def _quantize_heads(
        self, states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize all heads and bit-pack the indices.

        Dispatches to the Triton fused kernel on CUDA, falls back to
        the Python per-head loop on CPU.

        Args:
            states: ``(B, n_kv_heads, seq, head_dim)``
            layer_idx: Decoder layer index.

        Returns:
            ``(packed_indices, scales)`` where packed_indices has shape
            ``(B, H, S, packed_dim)`` uint8 and scales ``(B, H, S, 1)``.
        """
        if states.is_cuda:
            from qwen3_tts_triton.kernels.fused_dequant import (
                triton_fused_quant,
            )

            return triton_fused_quant(
                states,
                self._rotations_stacked[layer_idx],
                self.boundaries,
                self.bits,
                scale_dtype=states.dtype,
            )
        return self._quantize_heads_cpu(states, layer_idx)

    def _quantize_heads_cpu(
        self, states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU fallback: quantize via Python per-head loop.

        Args:
            states: ``(B, n_kv_heads, seq, head_dim)``
            layer_idx: Decoder layer index.

        Returns:
            ``(packed_indices, scales)`` where packed_indices has shape
            ``(B, H, S, packed_dim)`` uint8 and scales ``(B, H, S, 1)``.
        """
        b, h, s, d = states.shape
        all_indices = []
        all_scales = []
        for head_idx in range(h):
            head_data = states[:, head_idx, :, :]  # (B, seq, dim)
            rot = self.rotations[layer_idx][head_idx]
            idx, sc = quantize_vectors(head_data, rot, self.codebook, self.boundaries)
            all_indices.append(idx)
            all_scales.append(sc)
        # Stack back to (B, H, S, D) and (B, H, S, 1)
        indices = torch.stack(all_indices, dim=1)
        scales = torch.stack(all_scales, dim=1)

        # Bit-pack
        if self.bits == 4:
            indices = pack_4bit(indices)
        elif self.bits <= 3:
            indices = pack_3bit(indices)

        return indices, scales

    def _dequantize_heads(
        self,
        packed_indices: torch.Tensor,
        scales: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Unpack and dequantize all heads back to float.

        Dispatches to the Triton fused kernel on CUDA, falls back to
        the Python per-head loop on CPU.

        Args:
            packed_indices: ``(B, H, S, packed_dim)`` uint8 (bit-packed).
            scales: ``(B, n_kv_heads, seq, 1)`` float.
            layer_idx: Decoder layer index.

        Returns:
            Dequantized ``(B, n_kv_heads, seq, head_dim)`` in self.dtype.
        """
        if packed_indices.is_cuda:
            from qwen3_tts_triton.kernels.fused_dequant import (
                triton_fused_dequant,
            )

            return triton_fused_dequant(
                packed_indices,
                scales,
                self._rotations_stacked[layer_idx],
                self.codebook,
                self.head_dim,
                self.bits,
                out_dtype=self.dtype,
            )
        return self._dequantize_heads_cpu(packed_indices, scales, layer_idx)

    def _dequantize_heads_cpu(
        self,
        packed_indices: torch.Tensor,
        scales: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """CPU fallback: unpack and dequantize via Python per-head loop.

        Args:
            packed_indices: ``(B, H, S, packed_dim)`` uint8 (bit-packed).
            scales: ``(B, n_kv_heads, seq, 1)`` float.
            layer_idx: Decoder layer index.

        Returns:
            Dequantized ``(B, n_kv_heads, seq, head_dim)`` in self.dtype.
        """
        # Unpack
        if self.bits == 4:
            indices = unpack_4bit(packed_indices, self.head_dim)
        elif self.bits <= 3:
            indices = unpack_3bit(packed_indices, self.head_dim)
        else:
            indices = packed_indices

        b, h, s, d = indices.shape
        heads = []
        for head_idx in range(h):
            head_idx_data = indices[:, head_idx, :, :]  # (B, seq, dim)
            head_scales = scales[:, head_idx, :, :]  # (B, seq, 1)
            rot = self.rotations[layer_idx][head_idx]
            deq = dequantize_vectors(head_idx_data, head_scales, rot, self.codebook)
            heads.append(deq)
        return torch.stack(heads, dim=1).to(self.dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize new KV states and append to cache.

        Compatible with HuggingFace ``Cache.update()`` protocol.

        Args:
            key_states: New keys ``(B, n_kv_heads, new_seq, head_dim)``.
            value_states: New values ``(B, n_kv_heads, new_seq, head_dim)``.
            layer_idx: Current decoder layer index.
            cache_kwargs: Unused (HF API compatibility).

        Returns:
            ``(all_keys, all_values)`` dequantized for the full sequence.
        """
        # Quantize incoming tokens
        k_idx, k_sc = self._quantize_heads(key_states, layer_idx)
        v_idx, v_sc = self._quantize_heads(value_states, layer_idx)

        # Append to existing cache
        prev_ki = self._key_indices[layer_idx]
        if prev_ki is None:
            self._key_indices[layer_idx] = k_idx
            self._key_scales[layer_idx] = k_sc
            self._val_indices[layer_idx] = v_idx
            self._val_scales[layer_idx] = v_sc
        else:
            prev_ks = self._key_scales[layer_idx]
            prev_vi = self._val_indices[layer_idx]
            prev_vs = self._val_scales[layer_idx]
            assert prev_ks is not None and prev_vi is not None and prev_vs is not None
            self._key_indices[layer_idx] = torch.cat([prev_ki, k_idx], dim=2)
            self._key_scales[layer_idx] = torch.cat([prev_ks, k_sc], dim=2)
            self._val_indices[layer_idx] = torch.cat([prev_vi, v_idx], dim=2)
            self._val_scales[layer_idx] = torch.cat([prev_vs, v_sc], dim=2)

        cur_ki = self._key_indices[layer_idx]
        assert cur_ki is not None
        self._seq_lengths[layer_idx] = cur_ki.shape[2]

        # Incremental dequant: only decode NEW tokens, append to buffer
        new_keys = self._dequantize_heads(k_idx, k_sc, layer_idx)
        new_vals = self._dequantize_heads(v_idx, v_sc, layer_idx)

        prev_dk = self._deq_keys[layer_idx]
        if prev_dk is None:
            self._deq_keys[layer_idx] = new_keys
            self._deq_values[layer_idx] = new_vals
        else:
            prev_dv = self._deq_values[layer_idx]
            assert prev_dv is not None
            self._deq_keys[layer_idx] = torch.cat([prev_dk, new_keys], dim=2)
            self._deq_values[layer_idx] = torch.cat([prev_dv, new_vals], dim=2)

        ret_k = self._deq_keys[layer_idx]
        ret_v = self._deq_values[layer_idx]
        assert ret_k is not None and ret_v is not None
        return ret_k, ret_v

    def __len__(self) -> int:
        """Number of layers (HuggingFace cache protocol)."""
        return self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for a given layer."""
        return self._seq_lengths[layer_idx]

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return -1 (unlimited dynamic cache)."""
        return -1

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for attention mask generation.

        Matches HuggingFace DynamicCache protocol: for uninitialized
        layers, returns ``(cache_position.shape[0], 0)``.
        """
        seq_len = self.get_seq_length(layer_idx)
        if seq_len == 0:
            return cache_position.shape[0], 0
        return seq_len, 0

    def reset(self) -> None:
        """Clear all cached states."""
        for i in range(self.num_layers):
            self._key_indices[i] = None
            self._key_scales[i] = None
            self._val_indices[i] = None
            self._val_scales[i] = None
            self._deq_keys[i] = None
            self._deq_values[i] = None
            self._seq_lengths[i] = 0

    def get_memory_stats(self) -> dict[str, Any]:
        """Report actual vs uncompressed KV cache memory.

        Returns:
            Dict with compressed_mb, uncompressed_mb, compression_ratio,
            and per_layer breakdown.
        """
        compressed_bytes = 0
        uncompressed_bytes = 0
        per_layer_mb: list[float] = []

        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

        for i in range(self.num_layers):
            if self._key_indices[i] is None:
                per_layer_mb.append(0.0)
                continue

            seq_len = self._seq_lengths[i]
            # Compressed: bit-packed uint8 indices + float scales for K and V
            ki = self._key_indices[i]
            ks = self._key_scales[i]
            assert ki is not None and ks is not None
            layer_indices_bytes = ki.nelement() * 1  # uint8
            layer_scales_bytes = ks.nelement() * 2  # fp16
            layer_compressed = 2 * (layer_indices_bytes + layer_scales_bytes)  # K+V

            # Uncompressed: B * H * S * D * bytes_per_element for K and V
            b = ki.shape[0]
            layer_uncompressed = (
                2 * b * self.num_kv_heads * seq_len * self.head_dim * bytes_per_element
            )

            compressed_bytes += layer_compressed
            uncompressed_bytes += layer_uncompressed
            per_layer_mb.append(layer_compressed / 1024**2)

        compressed_mb = compressed_bytes / 1024**2
        uncompressed_mb = uncompressed_bytes / 1024**2
        ratio = uncompressed_mb / compressed_mb if compressed_mb > 0 else 0.0

        return {
            "compressed_mb": round(compressed_mb, 2),
            "uncompressed_mb": round(uncompressed_mb, 2),
            "compression_ratio": round(ratio, 2),
            "per_layer_mb": [round(x, 4) for x in per_layer_mb],
        }

    @property
    def is_initialized(self) -> bool:
        """Whether any layer has cached data."""
        return any(idx is not None for idx in self._key_indices)

    # Note: __len__ defined earlier returns self.num_layers (HF protocol)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Allow indexing for basic HF compatibility."""
        if self._key_indices[idx] is None:
            raise IndexError(f"Layer {idx} has no cached data")
        keys = self._dequantize_heads(
            self._key_indices[idx], self._key_scales[idx], idx
        )
        values = self._dequantize_heads(
            self._val_indices[idx], self._val_scales[idx], idx
        )
        return keys, values
