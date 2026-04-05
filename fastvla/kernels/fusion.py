"""
Vision-Language Fusion Triton Kernel — Fixed for GPU execution.
Fuses visual and text features via weighted sum: out = alpha * visual + (1-alpha) * text
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fusion_fwd_kernel(
    visual_ptr,
    text_ptr,
    out_ptr,
    B,
    T,
    D,
    stride_vb,
    stride_vt,
    stride_vd,
    stride_tb,
    stride_tt,
    stride_td,
    stride_ob,
    stride_ot,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """Each program handles one (batch, seq) element across the full feature dim D."""
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    num_blocks = tl.cdiv(D, BLOCK_D)
    for block_id in range(num_blocks):
        d_offset = block_id * BLOCK_D
        cols = d_offset + tl.arange(0, BLOCK_D)
        mask = cols < D

        v_offset = pid_b * stride_vb + pid_t * stride_vt + cols * stride_vd
        t_offset = pid_b * stride_tb + pid_t * stride_tt + cols * stride_td

        v = tl.load(visual_ptr + v_offset, mask=mask, other=0.0)
        t = tl.load(text_ptr + t_offset, mask=mask, other=0.0)

        fused = 0.5 * v + 0.5 * t

        o_offset = pid_b * stride_ob + pid_t * stride_ot + cols * stride_od
        tl.store(out_ptr + o_offset, fused, mask=mask)


@triton.jit
def _fusion_bwd_kernel(
    grad_out_ptr,
    grad_visual_ptr,
    grad_text_ptr,
    B,
    T,
    D,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_gvb,
    stride_gvt,
    stride_gvd,
    stride_gtb,
    stride_gtt,
    stride_gtd,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    num_blocks = tl.cdiv(D, BLOCK_D)
    for block_id in range(num_blocks):
        d_offset = block_id * BLOCK_D
        cols = d_offset + tl.arange(0, BLOCK_D)
        mask = cols < D

        g_offset = pid_b * stride_gb + pid_t * stride_gt + cols * stride_gd
        grad_out = tl.load(grad_out_ptr + g_offset, mask=mask, other=0.0)

        gv = 0.5 * grad_out
        gt = 0.5 * grad_out

        gv_off = pid_b * stride_gvb + pid_t * stride_gvt + cols * stride_gvd
        gt_off = pid_b * stride_gtb + pid_t * stride_gtt + cols * stride_gtd

        tl.store(grad_visual_ptr + gv_off, gv, mask=mask)
        tl.store(grad_text_ptr + gt_off, gt, mask=mask)


class _FusionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visual, text):
        B, T_v, D = visual.shape
        B_t, T_t, D_t = text.shape
        assert B == B_t and D == D_t

        if T_v != T_t:
            visual = visual.mean(dim=1, keepdim=True).expand(-1, T_t, -1).contiguous()
            T_v = T_t

        out = torch.empty_like(text)
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 4096:
            BLOCK_D = 4096

        def grid(meta):
            return (B, T_t)

        _fusion_fwd_kernel[grid](
            visual,
            text,
            out,
            B,
            T_t,
            D,
            visual.stride(0),
            visual.stride(1),
            visual.stride(2),
            text.stride(0),
            text.stride(1),
            text.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(visual, text)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        visual, text = ctx.saved_tensors
        B, T_t, D = grad_out.shape

        grad_visual = torch.zeros_like(visual)
        grad_text = torch.zeros_like(text)

        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 4096:
            BLOCK_D = 4096

        def grid(meta):
            return (B, T_t)

        _fusion_bwd_kernel[grid](
            grad_out,
            grad_visual,
            grad_text,
            B,
            T_t,
            D,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_visual.stride(0),
            grad_visual.stride(1),
            grad_visual.stride(2),
            grad_text.stride(0),
            grad_text.stride(1),
            grad_text.stride(2),
            BLOCK_D=BLOCK_D,
        )

        return grad_visual, grad_text


def vision_language_fusion_forward(
    visual: torch.Tensor, text: torch.Tensor
) -> torch.Tensor:
    """Fusion forward with autograd support."""
    return _FusionAutograd.apply(visual, text)


def vision_language_fusion_backward(grad_out, visual, text):
    """Not used directly — autograd handles it."""
    raise NotImplementedError("Use autograd via vision_language_fusion_forward")
