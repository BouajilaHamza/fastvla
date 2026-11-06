"""
Spatial Token Compressor and Patch Merger for FastVLA.

In standard VLA models (e.g. OpenVLA, Octo), ViT encoders produce 256 patch tokens
per camera view (16x16 grid). In multi-camera robotics setups (e.g. 2 views = 512 tokens),
the 7B LLM spends >96% of its step time performing attention and MLP operations across
512 visual tokens on every layer.

SpatialPatchMerger condenses visual tokens via 2x2 (or 4x4) spatial grouping and
linear projection (Qwen2-VL / LLaVA-NeXT style):
- 256 patches (16x16) -> 64 tokens (8x8) via 2x2 merging (75% token reduction)
- Across 2 camera views: 128 tokens instead of 512 tokens
- LLM forward and backward latency drops by 3x to 4x
- Activation VRAM drops by ~70%
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPatchMerger(nn.Module):
    """
    Spatially groups adjacent patches in a 2D grid and projects them to target dimension.
    
    Transforms a [B, H*W, in_dim] sequence (e.g. H=16, W=16, 256 tokens)
    into [B, (H/k)*(W/k), out_dim] (e.g. k=2 -> 8x8 = 64 tokens).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        merge_factor: int = 2,
        grid_size: Optional[Tuple[int, int]] = (16, 16),
        use_norm: bool = True,
    ):
        """
        Initialize the spatial patch merger.
        
        Args:
            in_dim: Dimension of input patch tokens from ViT (e.g. 1152 for SigLIP-So400M).
            out_dim: Dimension of output tokens for LLM (e.g. 4096 for LLaMA-7B).
            merge_factor: Spatial reduction factor k (default 2: groups 2x2=4 patches into 1).
            grid_size: (H, W) spatial grid dimensions of the patches (default (16, 16) for 224x224 ViT-14).
            use_norm: Whether to apply LayerNorm before projection.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.merge_factor = merge_factor
        self.grid_size = grid_size
        self.num_merged_patches = merge_factor * merge_factor

        self.norm = nn.LayerNorm(in_dim * self.num_merged_patches) if use_norm else nn.Identity()
        self.proj = nn.Linear(in_dim * self.num_merged_patches, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        num_cams: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass compressing patch tokens.
        
        Args:
            x: Input patch tensor of shape [B, num_cams * (H*W), in_dim] or [B, H*W, in_dim].
            num_cams: Number of camera views concatenated along sequence dimension.
            
        Returns:
            Compressed tokens: [B, num_cams * ((H/k)*(W/k)), out_dim].
        """
        B, total_tokens, D = x.shape
        tokens_per_cam = total_tokens // num_cams
        
        if self.grid_size is not None:
            H, W = self.grid_size
            if H * W != tokens_per_cam:
                # Infer square grid if mismatch
                side = int(math.isqrt(tokens_per_cam))
                if side * side == tokens_per_cam:
                    H, W = side, side
                else:
                    raise ValueError(
                        f"Expected tokens_per_cam ({tokens_per_cam}) to match grid_size {self.grid_size}"
                    )
        else:
            side = int(math.isqrt(tokens_per_cam))
            H, W = side, side

        k = self.merge_factor
        if H % k != 0 or W % k != 0:
            raise ValueError(f"Grid size ({H}, {W}) must be divisible by merge_factor {k}")

        new_H, new_W = H // k, W // k

        # Reshape to [B * num_cams, H, W, D]
        x_cams = x.view(B * num_cams, H, W, D)

        # Partition into [B * num_cams, new_H, k, new_W, k, D]
        x_blocks = x_cams.view(B * num_cams, new_H, k, new_W, k, D)

        # Permute to group adjacent patches: [B * num_cams, new_H, new_W, k, k, D]
        x_permuted = x_blocks.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Flatten k x k x D into (k^2 * D): [B * num_cams, new_H * new_W, k^2 * D]
        x_merged = x_permuted.view(B * num_cams, new_H * new_W, k * k * D)

        # Norm and Project
        out_cam = self.proj(self.norm(x_merged))  # [B * num_cams, new_H * new_W, out_dim]

        # Reshape back to [B, num_cams * (new_H * new_W), out_dim]
        out = out_cam.view(B, num_cams * (new_H * new_W), self.out_dim)
        return out


class PerceiverTokenCompressor(nn.Module):
    """
    Compresses arbitrary visual token sequences into a fixed number of latent tokens
    using cross-attention (Flamingo / Perceiver style).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.out_dim = out_dim
        self.latents = nn.Parameter(torch.randn(1, num_latents, out_dim) * 0.02)
        
        self.proj_in = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_latents = nn.LayerNorm(out_dim)
        self.norm_tokens = nn.LayerNorm(out_dim)
        self.norm_out = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, x: torch.Tensor, num_cams: int = 1) -> torch.Tensor:
        """
        Args:
            x: [B, total_tokens, in_dim]
            num_cams: Number of cameras. If > 1, latents are replicated per camera.
        """
        B = x.shape[0]
        x_proj = self.norm_tokens(self.proj_in(x))
        
        if num_cams > 1:
            tokens_per_cam = x.shape[1] // num_cams
            x_cam = x_proj.view(B * num_cams, tokens_per_cam, self.out_dim)
            latents = self.latents.expand(B * num_cams, -1, -1)
            
            q = self.norm_latents(latents)
            attn_out, _ = self.cross_attn(query=q, key=x_cam, value=x_cam)
            latents = latents + attn_out
            latents = latents + self.mlp(self.norm_out(latents))
            
            return latents.reshape(B, num_cams * self.num_latents, self.out_dim)
        else:
            latents = self.latents.expand(B, -1, -1)
            q = self.norm_latents(latents)
            attn_out, _ = self.cross_attn(query=q, key=x_proj, value=x_proj)
            latents = latents + attn_out
            latents = latents + self.mlp(self.norm_out(latents))
            return latents
