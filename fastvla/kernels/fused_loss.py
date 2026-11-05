"""
Fused Cross Entropy Loss Kernel using Triton.
Avoids materializing large [B, num_bins] logits probability distributions in global memory.
Particularly effective for DiscreteActionHead (discretized action bins).
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_cross_entropy_fwd_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stride_lb,
    stride_ld,
    stride_tb,
    B,
    num_bins: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = batch_idx < B

    cols = tl.arange(0, num_bins)

    # Load targets
    targets = tl.load(targets_ptr + batch_idx * stride_tb, mask=mask, other=0)

    # Load logits [BLOCK_B, num_bins]
    logits = tl.load(
        logits_ptr + batch_idx[:, None] * stride_lb + cols[None, :] * stride_ld,
        mask=mask[:, None],
        other=float("-inf"),
    )

    # Stable Softmax: max subtraction
    m = tl.max(logits, axis=1)
    exp_logits = tl.exp(logits - m[:, None])
    sum_exp = tl.sum(exp_logits, axis=1)
    log_sum_exp = tl.log(sum_exp) + m

    # Logit of target class
    target_logits = tl.load(
        logits_ptr + batch_idx * stride_lb + targets * stride_ld,
        mask=mask,
        other=0.0,
    )

    # Loss = log_sum_exp - target_logit
    loss = log_sum_exp - target_logits
    tl.store(loss_ptr + batch_idx, loss, mask=mask)

def fused_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes cross entropy loss: logits [N, num_bins], targets [N].
    Falls back to torch.nn.functional.cross_entropy on CPU or when Triton is unavailable.
    """
    if not logits.is_cuda:
        return torch.nn.functional.cross_entropy(logits, targets)

    N, num_bins = logits.shape
    loss = torch.empty(N, device=logits.device, dtype=torch.float32)

    BLOCK_B = 64
    grid = (triton.cdiv(N, BLOCK_B),)

    try:
        _fused_cross_entropy_fwd_kernel[grid](
            logits,
            targets,
            loss,
            logits.stride(0),
            logits.stride(1),
            targets.stride(0),
            N,
            num_bins=triton.next_power_of_2(num_bins),
            BLOCK_B=BLOCK_B,
        )
        return loss.mean()
    except Exception:
        # Fallback to standard PyTorch if Triton kernel execution encounters any issue
        return torch.nn.functional.cross_entropy(logits, targets)
