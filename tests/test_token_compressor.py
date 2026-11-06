"""
Unit tests for FastVLA Spatial Token Compressor and Patch Merger.
"""

import pytest
import torch
from fastvla.adapters.token_compressor import SpatialPatchMerger, PerceiverTokenCompressor


class TestSpatialPatchMerger:
    def test_single_camera_2x2_merging(self):
        """256 patches (16x16) merged 2x2 should yield exactly 64 tokens."""
        B, num_patches, in_dim, out_dim = 2, 256, 1152, 4096
        merger = SpatialPatchMerger(
            in_dim=in_dim,
            out_dim=out_dim,
            merge_factor=2,
            grid_size=(16, 16),
        )

        x = torch.randn(B, num_patches, in_dim, requires_grad=True)
        out = merger(x, num_cams=1)

        assert out.shape == (B, 64, out_dim), f"Expected [2, 64, 4096], got {out.shape}"
        assert out.requires_grad

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_multi_camera_2x2_merging(self):
        """2 cameras x 256 patches (512 total) should yield 2 x 64 = 128 tokens."""
        B, num_cams, patches_per_cam = 4, 2, 256
        in_dim, out_dim = 1152, 4096
        merger = SpatialPatchMerger(
            in_dim=in_dim,
            out_dim=out_dim,
            merge_factor=2,
            grid_size=(16, 16),
        )

        x = torch.randn(B, num_cams * patches_per_cam, in_dim, requires_grad=True)
        out = merger(x, num_cams=num_cams)

        # 512 tokens reduced to 128 tokens (75% token reduction)
        assert out.shape == (B, 128, out_dim), f"Expected [4, 128, 4096], got {out.shape}"

        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_factor_4_extreme_compression(self):
        """256 patches merged 4x4 should yield 16 tokens (93.75% token reduction)."""
        B, in_dim, out_dim = 2, 1152, 2048
        merger = SpatialPatchMerger(
            in_dim=in_dim,
            out_dim=out_dim,
            merge_factor=4,
            grid_size=(16, 16),
        )

        x = torch.randn(B, 256, in_dim)
        out = merger(x, num_cams=1)

        assert out.shape == (B, 16, out_dim)

    def test_perceiver_compressor(self):
        """Perceiver resampler compresses tokens to fixed latent count."""
        B, num_cams = 2, 2
        in_dim, out_dim = 1152, 4096
        compressor = PerceiverTokenCompressor(
            in_dim=in_dim,
            out_dim=out_dim,
            num_latents=32,
            num_heads=8,
        )

        x = torch.randn(B, num_cams * 256, in_dim, requires_grad=True)
        out = compressor(x, num_cams=num_cams)

        # 2 cameras x 32 latents = 64 tokens
        assert out.shape == (B, 64, out_dim)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
