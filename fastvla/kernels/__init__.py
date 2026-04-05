"""Custom Triton kernels for FastVLA with CPU fallbacks."""

import torch

from .cpu_fallbacks import (
    action_decode_cpu,
    multi_cam_pack_cpu,
    vision_language_fusion_cpu,
)


def _is_gpu_available() -> bool:
    """Check if GPU with Triton is available."""
    return torch.cuda.is_available()


if _is_gpu_available():
    from .action import (
        action_decode_backward as _triton_action_backward,
    )
    from .action import (
        action_decode_forward as _triton_action_forward,
    )
    from .action_head import TritonActionHead
    from .fusion import (
        vision_language_fusion_backward as _triton_fusion_backward,
    )
    from .fusion import (
        vision_language_fusion_forward as _triton_fusion_forward,
    )
else:
    from .action_head import TritonActionHead


def _use_triton(tensor: torch.Tensor) -> bool:
    """Use Triton only if tensor is on a CUDA device."""
    return tensor.is_cuda


def vision_language_fusion_forward(
    visual_feat: torch.Tensor, text_feat: torch.Tensor
) -> torch.Tensor:
    """Fuse visual and language features. Uses Triton on GPU, PyTorch on CPU."""
    if _use_triton(visual_feat):
        return _triton_fusion_forward(visual_feat, text_feat)
    return vision_language_fusion_cpu(visual_feat, text_feat)


def vision_language_fusion_backward(
    grad_output: torch.Tensor,
    visual_feat: torch.Tensor,
    text_feat: torch.Tensor,
) -> tuple:
    """Compute gradients for fusion."""
    if _use_triton(grad_output):
        return _triton_fusion_backward(grad_output, visual_feat, text_feat)
    visual_feat = visual_feat.clone().requires_grad_(True)
    text_feat = text_feat.clone().requires_grad_(True)
    fused = vision_language_fusion_cpu(visual_feat, text_feat)
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
    if _use_triton(hidden):
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
    if _use_triton(hidden):
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
    "vision_language_fusion_backward",
    "vision_language_fusion_forward",
]
