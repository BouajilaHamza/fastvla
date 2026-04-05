"""
Pure PyTorch CPU fallbacks for custom Triton kernels.
Used when GPU is not available.
"""
import torch
import torch.nn.functional as F


def vision_language_fusion_cpu(visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
    """
    Fuse visual and language features using weighted sum.
    Matches the Triton kernel's alpha=0.5 behavior.

    Args:
        visual_feat: [B, T_v, D] Visual features
        text_feat: [B, T_t, D] Text features

    Returns:
        [B, T_t, D] Fused features
    """
    alpha = 0.5

    # If sequence lengths don't match, pool visual features
    if visual_feat.size(1) != text_feat.size(1):
        # Pool visual features to a single token
        visual_pooled = visual_feat.mean(dim=1, keepdim=True)  # [B, 1, D]
        visual_pooled = visual_pooled.expand(-1, text_feat.size(1), -1)  # [B, T_t, D]
        visual_feat = visual_pooled

    fused = alpha * visual_feat + (1 - alpha) * text_feat
    return fused


def multi_cam_pack_cpu(cams: torch.Tensor) -> torch.Tensor:
    """
    Pack multiple camera views into a single tensor.
    Equivalent to the Triton multi_cam_pack_forward.

    Args:
        cams: [B, num_cams, C, H, W] Input camera views

    Returns:
        [B, num_cams*C, H, W] Packed tensor
    """
    B, num_cams, C, H, W = cams.shape
    # Rearrange: [B, num_cams, C, H, W] -> [B, num_cams*C, H, W]
    cams = cams.permute(0, 1, 2, 3, 4)  # already in correct order
    return cams.reshape(B, num_cams * C, H, W)


def action_decode_cpu(
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """
    Two-layer MLP for action decoding.
    Matches the Triton kernel: ReLU(h @ W1 + b1) -> Tanh(h @ W2 + b2).

    Args:
        hidden: [B, D] Input hidden states
        weight1: [D, H] First layer weights
        bias1: [H] First layer bias
        weight2: [H, A] Second layer weights
        bias2: [A] Second layer bias

    Returns:
        [B, A] Output actions in [-1, 1]
    """
    h = F.relu(hidden @ weight1 + bias1)
    output = torch.tanh(h @ weight2 + bias2)
    return output
