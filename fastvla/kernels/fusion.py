"""
Vision-Language Fusion — Optimized Cross-Attention.
Implements a memory-efficient Cross-Attention mechanism where text tokens (Q)
attend to visual features (K, V) from multiple cameras.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _cross_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_ob, stride_ot, stride_od,
    B, T_q, T_kv, D,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for memory-efficient Cross-Attention.
    Each program handles a block of Query tokens attending to all KV tokens.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # Index of Query blocks

    # Offsets for Query block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Mask for Query tokens
    mask_q = rm < T_q

    # Initialize pointers for Q
    q_ptrs = Q_ptr + pid_b * stride_qb + rm[:, None] * stride_qt + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

    # Online Softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Loop over KV tokens
    for start_n in range(0, T_kv, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        mask_kv = rn < T_kv

        # Load K and V
        k_ptrs = K_ptr + pid_b * stride_kb + rn[None, :] * stride_kt + tl.arange(0, BLOCK_D)[:, None] * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + rn[:, None] * stride_vt + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_kv[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv[:, None], other=0.0)

        # QK^T scaled
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(mask_q[:, None] & mask_kv[None, :], qk, float("-inf"))

        # Online Softmax update
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16)) * beta[:, None]
        l_i = l_i * alpha + l_ij * beta
        m_i = m_next

    # Finalize normalization
    acc = acc / l_i[:, None]

    # Store result
    out_ptrs = Out_ptr + pid_b * stride_ob + rm[:, None] * stride_ot + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_q[:, None])


def vision_language_cross_attention(
    text: torch.Tensor, visual: torch.Tensor
) -> torch.Tensor:
    """
    Standardized Multi-Modal Cross-Attention.
    
    Args:
        text: Queries [B, T_q, D]
        visual: Keys/Values [B, T_kv, D]
    Returns:
        fused: [B, T_q, D]
    """
    # 1. Shape and Device Validation
    B, T_q, D = text.shape
    _, T_kv, D_v = visual.shape
    assert D == D_v, f"Feature dimension mismatch: text={D}, visual={D_v}"
    
    if text.dtype != visual.dtype:
        visual = visual.to(text.dtype)

    # 2. CPU / Low-Compute Fallback
    if not text.is_cuda:
        # High-performance PyTorch native scaled dot-product attention
        return torch.nn.functional.scaled_dot_product_attention(
            text, visual, visual
        )

    # 3. Triton CUDA Path
    out = torch.empty_like(text)
    sm_scale = 1.0 / math.sqrt(D)

    # Auto-tuning block sizes based on D
    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_D > 1024:
        # Fallback to PyTorch for very large dimensions if Triton block size limit reached
        return torch.nn.functional.scaled_dot_product_attention(
            text, visual, visual
        )

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (B, triton.cdiv(T_q, BLOCK_M))

    _cross_attention_fwd_kernel[grid](
        text, visual, visual, out,
        text.stride(0), text.stride(1), text.stride(2),
        visual.stride(0), visual.stride(1), visual.stride(2),
        visual.stride(0), visual.stride(1), visual.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, T_q, T_kv, D,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2
    )

    return out


def vision_language_fusion_forward(
    visual: torch.Tensor, text: torch.Tensor
) -> torch.Tensor:
    """
    Legacy wrapper for backward compatibility. 
    Swaps arguments to (text, visual) to match Q, KV order.
    """
    return vision_language_cross_attention(text, visual)
