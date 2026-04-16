"""Triton fused dequantization kernel for TurboQuant KV cache.

Fuses unpack + codebook gather + rotation matmul + rescale into a single
kernel launch, eliminating the Python for-loop over heads and intermediate
tensor allocations.  Reduces ~1792 CUDA launches per token to ~56.

Forward-only, inference optimized for RTX 5090 (sm_120, CUDA 12.8).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dequant_4bit_kernel(
    # Output: (B, H, S, D)
    out_ptr,
    out_stride_b,
    out_stride_h,
    out_stride_s,
    # Packed indices: (B, H, S, PACKED_DIM) uint8
    packed_ptr,
    pack_stride_b,
    pack_stride_h,
    pack_stride_s,
    # Scales: (B, H, S, 1)
    scales_ptr,
    sc_stride_b,
    sc_stride_h,
    sc_stride_s,
    # Rotation matrices: (H, D, D) float32
    rot_ptr,
    rot_stride_h,
    rot_stride_row,
    # Codebook: (N_LEVELS,) float32
    cb_ptr,
    # Dimensions
    S,
    H,
    D: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    SQRT_DIM_INV: tl.constexpr,
    TILE: tl.constexpr,
):
    """Fused 4-bit dequantization: unpack + codebook + rotation + rescale.

    Each program processes one vector at (batch, head, seq_pos).
    Grid: (B * H * S,).
    """
    pid = tl.program_id(0)
    s_idx = pid % S
    h_idx = (pid // S) % H
    b_idx = pid // (S * H)

    # --- Step 1: Load packed indices and unpack 4-bit → D indices ---
    pack_base = b_idx * pack_stride_b + h_idx * pack_stride_h + s_idx * pack_stride_s
    d_offsets = tl.arange(0, D)
    pack_idx = d_offsets // 2  # which packed byte (0..63)
    is_odd = d_offsets % 2

    packed_vals = tl.load(packed_ptr + pack_base + pack_idx)
    hi = (packed_vals >> 4) & 0x0F
    lo = packed_vals & 0x0F
    indices = tl.where(is_odd == 0, hi, lo)  # (D,) uint8

    # --- Step 2: Codebook gather + scale by 1/sqrt(D) ---
    recon = tl.load(cb_ptr + indices.to(tl.int32)).to(tl.float32) * SQRT_DIM_INV

    # --- Step 3: Load scale ---
    sc_base = b_idx * sc_stride_b + h_idx * sc_stride_h + s_idx * sc_stride_s
    scale = tl.load(scales_ptr + sc_base).to(tl.float32)

    # --- Step 4: Rotation matmul (tiled) + rescale + store ---
    out_base = b_idx * out_stride_b + h_idx * out_stride_h + s_idx * out_stride_s
    rot_head_base = h_idx * rot_stride_h

    k_offsets = tl.arange(0, D)  # reduction dim

    for j_start in range(0, D, TILE):
        j_offsets = j_start + tl.arange(0, TILE)
        j_mask = j_offsets < D

        # Load rotation tile: (D, TILE) from rot[h, :, j_start:j_start+TILE]
        rot_tile = tl.load(
            rot_ptr
            + rot_head_base
            + k_offsets[:, None] * rot_stride_row
            + j_offsets[None, :],
            mask=j_mask[None, :],
            other=0.0,
        )

        # Vector-matrix multiply: recon (D,) @ rot_tile (D, TILE) → (TILE,)
        out_tile = tl.sum(recon[:, None] * rot_tile, axis=0)

        # Rescale and store
        tl.store(
            out_ptr + out_base + j_offsets,
            (out_tile * scale).to(out_ptr.dtype.element_ty),
            mask=j_mask,
        )


@triton.jit
def _fused_dequant_3bit_kernel(
    # Output: (B, H, S, D)
    out_ptr,
    out_stride_b,
    out_stride_h,
    out_stride_s,
    # Packed indices: (B, H, S, PACKED_DIM) uint8
    packed_ptr,
    pack_stride_b,
    pack_stride_h,
    pack_stride_s,
    # Scales: (B, H, S, 1)
    scales_ptr,
    sc_stride_b,
    sc_stride_h,
    sc_stride_s,
    # Rotation matrices: (H, D, D) float32
    rot_ptr,
    rot_stride_h,
    rot_stride_row,
    # Codebook: (N_LEVELS,) float32
    cb_ptr,
    # Dimensions
    S,
    H,
    D: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    SQRT_DIM_INV: tl.constexpr,
    TILE: tl.constexpr,
):
    """Fused 3-bit dequantization: unpack + codebook + rotation + rescale.

    3-bit packing: 8 values (24 bits) packed into 3 bytes.
    Each program processes one vector at (batch, head, seq_pos).
    Grid: (B * H * S,).
    """
    pid = tl.program_id(0)
    s_idx = pid % S
    h_idx = (pid // S) % H
    b_idx = pid // (S * H)

    # --- Step 1: Load packed bytes and unpack 3-bit → D indices ---
    pack_base = b_idx * pack_stride_b + h_idx * pack_stride_h + s_idx * pack_stride_s

    d_offsets = tl.arange(0, D)
    group_idx = d_offsets // 8  # which group of 8 (0..15 for D=128)
    within_group = d_offsets % 8  # position 0..7
    byte_base = group_idx * 3  # 3 bytes per group

    b0 = tl.load(packed_ptr + pack_base + byte_base).to(tl.int32)
    b1 = tl.load(packed_ptr + pack_base + byte_base + 1).to(tl.int32)
    b2 = tl.load(packed_ptr + pack_base + byte_base + 2).to(tl.int32)

    # Decode all 8 positions
    v0 = (b0 >> 5) & 0x07
    v1 = (b0 >> 2) & 0x07
    v2 = ((b0 & 0x03) << 1) | ((b1 >> 7) & 0x01)
    v3 = (b1 >> 4) & 0x07
    v4 = (b1 >> 1) & 0x07
    v5 = ((b1 & 0x01) << 2) | ((b2 >> 6) & 0x03)
    v6 = (b2 >> 3) & 0x07
    v7 = b2 & 0x07

    # Select based on within_group position
    indices = tl.where(
        within_group == 0,
        v0,
        tl.where(
            within_group == 1,
            v1,
            tl.where(
                within_group == 2,
                v2,
                tl.where(
                    within_group == 3,
                    v3,
                    tl.where(
                        within_group == 4,
                        v4,
                        tl.where(
                            within_group == 5,
                            v5,
                            tl.where(within_group == 6, v6, v7),
                        ),
                    ),
                ),
            ),
        ),
    )

    # --- Step 2: Codebook gather + scale by 1/sqrt(D) ---
    recon = tl.load(cb_ptr + indices.to(tl.int32)).to(tl.float32) * SQRT_DIM_INV

    # --- Step 3: Load scale ---
    sc_base = b_idx * sc_stride_b + h_idx * sc_stride_h + s_idx * sc_stride_s
    scale = tl.load(scales_ptr + sc_base).to(tl.float32)

    # --- Step 4: Rotation matmul (tiled) + rescale + store ---
    out_base = b_idx * out_stride_b + h_idx * out_stride_h + s_idx * out_stride_s
    rot_head_base = h_idx * rot_stride_h

    k_offsets = tl.arange(0, D)

    for j_start in range(0, D, TILE):
        j_offsets = j_start + tl.arange(0, TILE)
        j_mask = j_offsets < D

        rot_tile = tl.load(
            rot_ptr
            + rot_head_base
            + k_offsets[:, None] * rot_stride_row
            + j_offsets[None, :],
            mask=j_mask[None, :],
            other=0.0,
        )

        out_tile = tl.sum(recon[:, None] * rot_tile, axis=0)

        tl.store(
            out_ptr + out_base + j_offsets,
            (out_tile * scale).to(out_ptr.dtype.element_ty),
            mask=j_mask,
        )


def triton_fused_dequant(
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    codebook: torch.Tensor,
    head_dim: int,
    bits: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused dequantization: unpack + codebook + rotation + rescale.

    Replaces the Python for-loop in ``_dequantize_heads()`` with a single
    Triton kernel launch that processes all heads in parallel.

    Args:
        packed_indices: Bit-packed indices ``(B, H, S, packed_dim)`` uint8.
        scales: Per-vector L2 norm scales ``(B, H, S, 1)``.
        rotations: Pre-stacked rotation matrices ``(H, D, D)`` float32.
        codebook: Lloyd-Max centroids ``(n_levels,)`` float32.
        head_dim: Head dimension (typically 128).
        bits: Quantization bit-width (3 or 4).
        out_dtype: Output dtype (default bfloat16).

    Returns:
        Dequantized tensor ``(B, H, S, D)`` in ``out_dtype``.
    """
    B, H, S, _ = packed_indices.shape
    D = head_dim

    packed_indices = packed_indices.contiguous()
    scales = scales.contiguous()
    rotations = rotations.contiguous()
    codebook = codebook.contiguous()

    out = torch.empty(B, H, S, D, dtype=out_dtype, device=packed_indices.device)

    grid = (B * H * S,)
    TILE = min(32, D)
    sqrt_dim_inv = 1.0 / (D**0.5)

    kernel = _fused_dequant_4bit_kernel if bits == 4 else _fused_dequant_3bit_kernel

    kernel[grid](
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        packed_indices,
        packed_indices.stride(0),
        packed_indices.stride(1),
        packed_indices.stride(2),
        scales,
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        rotations,
        rotations.stride(0),
        rotations.stride(1),
        codebook,
        S=S,
        H=H,
        D=D,
        PACKED_DIM=packed_indices.shape[-1],
        SQRT_DIM_INV=sqrt_dim_inv,
        TILE=TILE,
        num_warps=4,
    )

    return out


# =====================================================================
# Fused Quantize Kernel
# =====================================================================


@triton.jit
def _fused_quant_kernel(
    # Output: raw indices (B, H, S, D) uint8 (unpacked)
    idx_ptr,
    idx_stride_b,
    idx_stride_h,
    idx_stride_s,
    # Output: scales (B, H, S, 1)
    sc_ptr,
    sc_stride_b,
    sc_stride_h,
    sc_stride_s,
    # Input: (B, H, S, D)
    x_ptr,
    x_stride_b,
    x_stride_h,
    x_stride_s,
    # Rotation matrices: (H, D, D) float32
    rot_ptr,
    rot_stride_h,
    rot_stride_row,
    # Boundaries: (N_BOUNDS,) float32
    bd_ptr,
    # Dimensions
    S,
    H,
    N_BOUNDS: tl.constexpr,
    D: tl.constexpr,
    SQRT_DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    """Fused quantize: norm + rotate + searchsorted.

    Outputs raw uint8 indices (not bit-packed). Packing is done in the
    Python wrapper via pack_4bit/pack_3bit (cheap element-wise ops).

    Each program processes one vector at (batch, head, seq_pos).
    Grid: (B * H * S,).
    """
    pid = tl.program_id(0)
    s_idx = pid % S
    h_idx = (pid // S) % H
    b_idx = pid // (S * H)

    # --- Step 1: Load input vector ---
    x_base = b_idx * x_stride_b + h_idx * x_stride_h + s_idx * x_stride_s
    d_offsets = tl.arange(0, D)
    x = tl.load(x_ptr + x_base + d_offsets).to(tl.float32)

    # --- Step 2: L2 norm and normalize ---
    scale = tl.sqrt(tl.sum(x * x, axis=0) + 1e-16)
    x_norm = x / scale

    # --- Step 3: Rotation matmul: x_rot = x_norm @ R^T * sqrt(D) ---
    # x_rot[j] = sum_k(x_norm[k] * R[j, k]) * sqrt(D)  (ROW access, not column)
    # Tile over output dimension j
    rot_head_base = h_idx * rot_stride_h
    k_offsets = tl.arange(0, D)
    idx_base = b_idx * idx_stride_b + h_idx * idx_stride_h + s_idx * idx_stride_s

    for j_start in range(0, D, TILE):
        j_offsets = j_start + tl.arange(0, TILE)
        j_mask = j_offsets < D

        # Load R[j_start:j_start+TILE, :] = (TILE, D) — rows of R
        rot_tile = tl.load(
            rot_ptr
            + rot_head_base
            + j_offsets[:, None] * rot_stride_row
            + k_offsets[None, :],
            mask=j_mask[:, None],
            other=0.0,
        )

        # dot(R[j,:], x_norm) for each j in tile → (TILE,)
        x_rot_tile = tl.sum(rot_tile * x_norm[None, :], axis=1) * SQRT_DIM

        # --- Step 4: Searchsorted (linear scan) ---
        indices_tile = tl.zeros([TILE], dtype=tl.int32)
        for bi in range(N_BOUNDS):
            boundary = tl.load(bd_ptr + bi)
            indices_tile = tl.where(
                x_rot_tile > boundary, indices_tile + 1, indices_tile
            )

        # Store raw indices for this tile
        tl.store(
            idx_ptr + idx_base + j_offsets,
            indices_tile.to(tl.uint8),
            mask=j_mask,
        )

    # Store scale
    sc_base = b_idx * sc_stride_b + h_idx * sc_stride_h + s_idx * sc_stride_s
    tl.store(sc_ptr + sc_base, scale.to(sc_ptr.dtype.element_ty))


def triton_fused_quant(
    x: torch.Tensor,
    rotations: torch.Tensor,
    boundaries: torch.Tensor,
    bits: int,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused quantization: norm + rotate + searchsorted + bit-pack.

    Replaces the Python for-loop in ``_quantize_heads()`` with a single
    Triton kernel launch plus a cheap PyTorch bit-pack step.

    Args:
        x: Input tensor ``(B, H, S, D)``.
        rotations: Pre-stacked rotation matrices ``(H, D, D)`` float32.
        boundaries: Lloyd-Max decision boundaries ``(n_levels - 1,)`` float32.
        bits: Quantization bit-width (3 or 4).
        scale_dtype: Dtype for output scales (default bfloat16).

    Returns:
        ``(packed_indices, scales)`` where packed_indices is bit-packed
        uint8 and scales is ``(B, H, S, 1)`` in ``scale_dtype``.
    """
    from qwen3_tts_triton.kernels.turboquant import pack_3bit, pack_4bit

    B, H, S, D = x.shape

    x = x.contiguous()
    rotations = rotations.contiguous()
    boundaries = boundaries.contiguous()

    # Kernel outputs raw indices (unpacked)
    indices = torch.empty(B, H, S, D, dtype=torch.uint8, device=x.device)
    scales = torch.empty(B, H, S, 1, dtype=scale_dtype, device=x.device)

    n_bounds = 2**bits - 1
    grid = (B * H * S,)
    TILE = min(32, D)
    sqrt_dim = D**0.5

    _fused_quant_kernel[grid](
        indices,
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        scales,
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rotations,
        rotations.stride(0),
        rotations.stride(1),
        boundaries,
        S=S,
        H=H,
        N_BOUNDS=n_bounds,
        D=D,
        SQRT_DIM=sqrt_dim,
        TILE=TILE,
        num_warps=4,
    )

    # Bit-pack (cheap element-wise PyTorch ops)
    if bits == 4:
        indices = pack_4bit(indices)
    elif bits <= 3:
        indices = pack_3bit(indices)

    return indices, scales
