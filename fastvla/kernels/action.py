"""
Action Decoding Triton Kernels — Optimized Forward and Backward.
Implements a 2-layer fused MLP: tanh(ReLU(x @ W1 + b1) @ W2 + b2)
"""
import torch
import triton
import triton.language as tl


# ── FORWARD KERNEL ─────────────────────────────────────────────────────────

@triton.jit
def _action_fwd_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    B, D, H, A,
    stride_xb, stride_xd,
    stride_w1d, stride_w1h,
    stride_w2h, stride_w2a,
    stride_ob, stride_oa,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    """
    Fused forward pass: out = tanh(ReLU(x @ W1 + b1) @ W2 + b2)
    Tiled over Batch, Hidden (internally), and Action.
    """
    pid_b = tl.program_id(0)
    pid_a = tl.program_id(1)

    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    off_a = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    mask_b = off_b < B
    mask_a = off_a < A

    # Intermediate accumulator for the second layer input
    out_acc = tl.zeros([BLOCK_B, BLOCK_A], dtype=tl.float32)

    # Loop over tiles of H to stay within Shared Memory (SRAM) limits
    for h_start in range(0, H, BLOCK_H):
        off_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = off_h < H
        
        # 1. Compute layer 1 tile: h1_tile = ReLU(x @ W1_tile + b1_tile)
        h1_tile_acc = tl.zeros([BLOCK_B, BLOCK_H], dtype=tl.float32)
        for d_start in range(0, D, 32):
            off_d = d_start + tl.arange(0, 32)
            mask_d = off_d < D
            x = tl.load(x_ptr + off_b[:, None] * stride_xb + off_d[None, :] * stride_xd, 
                        mask=mask_b[:, None] & mask_d[None, :], other=0.0)
            w1 = tl.load(w1_ptr + off_d[:, None] * stride_w1d + off_h[None, :] * stride_w1h,
                         mask=mask_d[:, None] & mask_h[None, :], other=0.0)
            h1_tile_acc += tl.dot(x, w1)
        
        b1 = tl.load(b1_ptr + off_h, mask=mask_h, other=0.0)
        h1_tile = tl.maximum(h1_tile_acc + b1[None, :], 0.0)

        # 2. Accumulate into layer 2: out_acc += h1_tile @ W2_tile
        w2 = tl.load(w2_ptr + off_h[:, None] * stride_w2h + off_a[None, :] * stride_w2a,
                     mask=mask_h[:, None] & mask_a[None, :], other=0.0)
        out_acc += tl.dot(h1_tile.to(w2.dtype), w2)

    # 3. Add bias2 and apply Tanh (stable sigmoid-based implementation for Triton 3.x)
    b2 = tl.load(b2_ptr + off_a, mask=mask_a, other=0.0)
    out = tl.sigmoid(2 * (out_acc + b2[None, :])) * 2 - 1

    tl.store(out_ptr + off_b[:, None] * stride_ob + off_a[None, :] * stride_oa, 
             out, mask=mask_b[:, None] & mask_a[None, :])


# ── BACKWARD FUNCTIONS ─────────────────────────────────────────────────────

def action_decode_forward(
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """Forward pass using optimized Triton kernel."""
    B, D = hidden.shape
    D_chk, H = weight1.shape
    H_chk, A = weight2.shape
    assert D == D_chk and H == H_chk

    out = torch.empty(B, A, device=hidden.device, dtype=hidden.dtype)

    BLOCK_B = 16
    BLOCK_H = 256 # Fixed block size to avoid shared memory limits on T4
    BLOCK_A = 32 if A > 16 else 16

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(A, BLOCK_A))

    _action_fwd_kernel[grid](
        hidden, weight1, bias1, weight2, bias2, out,
        B, D, H, A,
        hidden.stride(0), hidden.stride(1),
        weight1.stride(0), weight1.stride(1),
        weight2.stride(0), weight2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_H=BLOCK_H, BLOCK_A=BLOCK_A,
    )
    return out


def action_decode_backward(grad_output, hidden, weight1, bias1, weight2, bias2):
    """
    Custom autograd-compatible backward.
    Optimized for robotics action dimensions with gradient checkpointing.
    """
    # Recompute intermediate hidden state (Gradient Checkpointing)
    # This avoids storing h1 during the forward pass of the model.
    h1 = torch.nn.functional.relu(hidden @ weight1 + bias1)
    output = torch.tanh(h1 @ weight2 + bias2)
    
    # d_tanh = grad_output * (1 - tanh^2)
    d_out = grad_output * (1.0 - output * output)
    
    # Gradients for Layer 2
    grad_weight2 = h1.t() @ d_out
    grad_bias2 = d_out.sum(dim=0)
    d_h1 = d_out @ weight2.t()
    
    # Gradient of ReLU
    d_h1_relu = d_h1 * (h1 > 0).to(d_h1.dtype)
    
    # Gradients for Layer 1
    grad_weight1 = hidden.t() @ d_h1_relu
    grad_bias1 = d_h1_relu.sum(dim=0)
    grad_hidden = d_h1_relu @ weight1.t()

    return grad_hidden, grad_weight1, grad_bias1, grad_weight2, grad_bias2
