"""Custom Triton kernels for FastVLA with automatic CPU/PyTorch fallbacks."""

import torch

from .cpu_fallbacks import (
    action_decode_cpu,
    multi_cam_pack_cpu,
    vision_language_fusion_cpu,
)


# ── Triton availability check ────────────────────────────────────────────
# Must check BOTH cuda AND triton importability.
# Kaggle/Colab always have CUDA but triton may not be installed yet.
TRITON_AVAILABLE = False

def _check_triton_available() -> bool:
    """Return True only if CUDA is present AND triton can be imported."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False

TRITON_AVAILABLE = _check_triton_available()

# ── Conditional Triton imports ───────────────────────────────────────────
_triton_action_forward = None
_triton_action_backward = None
_triton_fusion_forward = None
_triton_fusion_backward = None

if TRITON_AVAILABLE:
    try:
        from .action import (
            action_decode_backward as _triton_action_backward,
            action_decode_forward as _triton_action_forward,
        )
        from .fusion import (
            vision_language_cross_attention as _triton_cross_attention,
            vision_language_fusion_forward as _triton_fusion_forward,
        )
    except Exception:
        # If Triton kernels fail to compile (version mismatch, etc.)
        TRITON_AVAILABLE = False

# TritonActionHead always works — it has its own internal CPU fallback
from .action_head import TritonActionHead


def _use_triton(tensor: torch.Tensor) -> bool:
    """Use Triton only if tensor is on CUDA AND Triton is available."""
    return TRITON_AVAILABLE and tensor.is_cuda


def vision_language_cross_attention(
    text: torch.Tensor, visual: torch.Tensor
) -> torch.Tensor:
    """Standardized Multi-Modal Cross-Attention. Uses Triton on GPU, PyTorch on CPU."""
    if _use_triton(text) and _triton_cross_attention is not None:
        return _triton_cross_attention(text, visual)
    from .fusion import vision_language_cross_attention as _cpu_impl
    return _cpu_impl(text, visual)


def vision_language_fusion_forward(
    visual_feat: torch.Tensor, text_feat: torch.Tensor
) -> torch.Tensor:
    """Legacy wrapper. Swaps to Q, KV order for Cross-Attention."""
    return vision_language_cross_attention(text_feat, visual_feat)


def vision_language_fusion_backward(
    grad_output: torch.Tensor,
    visual_feat: torch.Tensor,
    text_feat: torch.Tensor,
) -> tuple:
    """Compute gradients for fusion via autograd."""
    visual_feat = visual_feat.clone().detach().requires_grad_(True)
    text_feat = text_feat.clone().detach().requires_grad_(True)
    fused = vision_language_cross_attention(text_feat, visual_feat)
    fused.backward(grad_output, retain_graph=True)
    return visual_feat.grad, text_feat.grad


def multi_cam_pack_forward(cams: torch.Tensor) -> torch.Tensor:
    """Pack multiple camera views. Pure PyTorch (Triton buys nothing for reshape)."""
    return multi_cam_pack_cpu(cams)


def multi_cam_pack_backward(grad_output: torch.Tensor, num_cams: int) -> torch.Tensor:
    """Backward for multi-camera packing."""
    B, C_total, H, W = grad_output.shape
    C = C_total // num_cams
    return grad_output.reshape(B, num_cams, C, H, W)


def action_decode_forward(
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """Action decoding MLP."""
    if _use_triton(hidden) and _triton_action_forward is not None:
        return _triton_action_forward(hidden, weight1, bias1, weight2, bias2)
    return action_decode_cpu(hidden, weight1, bias1, weight2, bias2)


def action_decode_backward(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
):
    """Backward for action decoding."""
    if _use_triton(hidden) and _triton_action_backward is not None:
        return _triton_action_backward(
            grad_output, hidden, weight1, bias1, weight2, bias2
        )
    return action_decode_cpu(hidden.clone(), weight1, bias1, weight2, bias2)


__all__ = [
    "action_decode_backward",
    "action_decode_forward",
    "multi_cam_pack_backward",
    "multi_cam_pack_forward",
    "TritonActionHead",
    "TRITON_AVAILABLE",
    "vision_language_cross_attention",
    "vision_language_fusion_backward",
    "vision_language_fusion_forward",
]
