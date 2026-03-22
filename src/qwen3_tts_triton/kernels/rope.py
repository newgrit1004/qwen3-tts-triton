"""Fused M-RoPE (Multi-dimensional Rotary Position Embedding) Triton kernel.

Implements interleaved M-RoPE for Qwen3-TTS with 3 dimensions:
- temporal (t), height (h), width (w)
- sections=[24, 20, 20] (total 64 = head_dim/2)
- Each section uses cos/sin from its corresponding dimension.

Forward-only, optimized for inference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_mrope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """M-RoPE forward kernel with interleaved rotation.

    Each program instance processes one (batch, seq_pos) pair across all heads.
    The head_dim is split into 3 sections, each using cos/sin from a
    different positional dimension (temporal, height, width).

    Interleaved rotation pairs: (x_{2i}, x_{2i+1}) for i in [0, hd/2).
    """
    pid = tl.program_id(0)

    # Data layout after transpose: (bsz, seq_len, n_head, head_dim)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # Section boundaries in half-head space
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    # cos/sin base pointers for 3 dimensions
    # cos shape: (3, bsz, seq_len, head_dim)
    # stride between dims = bs * sl * hd
    t_cos = cos_ptr + pid * hd
    h_cos = t_cos + bs * sl * hd
    w_cos = h_cos + bs * sl * hd
    t_sin = sin_ptr + pid * hd
    h_sin = t_sin + bs * sl * hd
    w_sin = h_sin + bs * sl * hd

    # Load cos/sin per section (only first hd/2 elements are meaningful)
    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < hd // 2)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    # Disjoint masks -> addition merges correctly
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # --- Q rotation (interleaved pairs) ---
    even_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :] * 2
    )
    odd_q_offsets = even_q_offsets + 1
    q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )

    q_even = tl.load(q_ptr + even_q_offsets, mask=q_mask, other=0).to(sin_row.dtype)
    q_odd = tl.load(q_ptr + odd_q_offsets, mask=q_mask, other=0).to(sin_row.dtype)

    new_q_even = q_even * cos_row - q_odd * sin_row
    new_q_odd = q_odd * cos_row + q_even * sin_row

    tl.store(q_ptr + even_q_offsets, new_q_even, mask=q_mask)
    tl.store(q_ptr + odd_q_offsets, new_q_odd, mask=q_mask)

    # --- K rotation (interleaved pairs) ---
    even_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :] * 2
    )
    odd_k_offsets = even_k_offsets + 1
    k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )

    k_even = tl.load(k_ptr + even_k_offsets, mask=k_mask, other=0).to(sin_row.dtype)
    k_odd = tl.load(k_ptr + odd_k_offsets, mask=k_mask, other=0).to(sin_row.dtype)

    new_k_even = k_even * cos_row - k_odd * sin_row
    new_k_odd = k_odd * cos_row + k_even * sin_row

    tl.store(k_ptr + even_k_offsets, new_k_even, mask=k_mask)
    tl.store(k_ptr + odd_k_offsets, new_k_odd, mask=k_mask)


def triton_mrope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved M-RoPE using a fused Triton kernel.

    Args:
        q: Query tensor, shape (bsz, n_q_head, seq_len, head_dim).
        k: Key tensor, shape (bsz, n_kv_head, seq_len, head_dim).
        cos: Cosine embeddings, shape (3, bsz, seq_len, head_dim).
        sin: Sine embeddings, shape (3, bsz, seq_len, head_dim).
        mrope_section: Section sizes [t, h, w], sum must equal head_dim/2.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.
    """
    # Transpose to (bsz, seq_len, n_head, head_dim) for contiguous access
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]

    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    _triton_mrope_kernel[(n_row,)](
        q,
        k,
        cos,
        sin,
        seq_len,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q.transpose(1, 2), k.transpose(1, 2)
