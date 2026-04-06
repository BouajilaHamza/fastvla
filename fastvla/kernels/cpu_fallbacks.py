import torch
import torch.nn.functional as F

def vision_language_fusion_cpu(visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Robust PyTorch fallback for vision-language fusion."""
    # Ensure dtypes match for the weighted sum
    if visual.dtype != text.dtype:
        visual = visual.to(text.dtype)
        
    # Handle sequence length mismatch (same logic as Triton kernel)
    if visual.size(1) != text.size(1):
        visual = visual.mean(dim=1, keepdim=True).expand(-1, text.size(1), -1)
        
    return 0.5 * visual + 0.5 * text

def action_decode_cpu(
    hidden: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """Robust PyTorch fallback for action decoding."""
    # Ensure dtypes match for linear layers
    dtype = hidden.dtype
    if weight1.dtype != dtype: weight1 = weight1.to(dtype)
    if bias1.dtype != dtype: bias1 = bias1.to(dtype)
    if weight2.dtype != dtype: weight2 = weight2.to(dtype)
    if bias2.dtype != dtype: bias2 = bias2.to(dtype)
    
    # MLP: tanh(ReLU(x @ W1 + b1) @ W2 + b2)
    h1 = F.linear(hidden, weight1.t(), bias1)
    h1_relu = F.relu(h1)
    out = F.linear(h1_relu, weight2.t(), bias2)
    return torch.tanh(out)

def multi_cam_pack_cpu(cams: torch.Tensor) -> torch.Tensor:
    """Reorder from [B, Cam, C, H, W] to [B, C*Cam, H, W]."""
    B, Cam, C, H, W = cams.shape
    return cams.permute(0, 2, 1, 3, 4).reshape(B, C * Cam, H, W)
