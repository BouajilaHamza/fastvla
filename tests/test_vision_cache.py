"""
Tests for VisionTokenCache and CachedVisionDataset.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from fastvla.data.cache import VisionTokenCache, CachedVisionDataset
from fastvla.model import FastVLAModel, FastVLAConfig


class MockRoboticsDataset(Dataset):
    """Mock robotics dataset returning raw image tensors."""

    def __init__(self, size: int = 10, num_cams: int = 2):
        self.size = size
        self.num_cams = num_cams

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "pixel_values": torch.randn(self.num_cams, 3, 224, 224),
            "input_ids": torch.randint(0, 500, (16,)),
            "action": torch.randn(7),
        }


class MockVisionTower(nn.Module):
    """Mock frozen vision encoder (SigLIP / ViT-like)."""

    def __init__(self, num_patches: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=56, stride=56)

    def forward(self, x):
        # x: [B, 3, 224, 224] -> [B, hidden_dim, 4, 4] -> [B, 16, hidden_dim]
        B = x.shape[0]
        feats = self.conv(x).flatten(2).transpose(1, 2)
        return feats


def test_vision_token_cache_in_memory():
    cache = VisionTokenCache(in_memory=True)
    tokens = torch.randn(32, 128)

    cache.put(0, tokens)
    assert cache.exists(0)
    assert not cache.exists(1)

    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(tokens.to(retrieved.dtype), retrieved)
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0
    assert not cache.exists(0)


def test_vision_token_cache_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = VisionTokenCache(cache_dir=tmpdir, in_memory=False)
        tokens = torch.randn(32, 128)

        cache.put("ep0_t1", tokens)
        assert cache.exists("ep0_t1")

        retrieved = cache.get("ep0_t1")
        assert retrieved is not None
        assert retrieved.shape == tokens.shape

        # Verify disk persistence
        assert (Path(tmpdir) / "token_ep0_t1.pt").exists()


def test_cached_vision_dataset_precache():
    base_ds = MockRoboticsDataset(size=5, num_cams=2)
    vision_tower = MockVisionTower(num_patches=16, hidden_dim=128)
    cached_ds = CachedVisionDataset(base_ds, in_memory=True)

    # Precache all samples
    cached_count = cached_ds.precache_all(vision_tower, device="cpu")
    assert cached_count == 5

    # Check item delivery: raw pixel_values should be replaced by visual_features
    sample = cached_ds[0]
    assert "visual_features" in sample
    assert "pixel_values" not in sample
    # num_cams * num_patches = 2 * 16 = 32 tokens
    assert sample["visual_features"].shape == (32, 128)


def test_fastvla_model_forward_with_cached_visual_features():
    """Verify FastVLAModel can accept visual_features directly without pixel_values."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=128,
        llm_hidden_size=128,
        action_dim=7,
        chunk_size=1,
    )
    model = FastVLAModel(config)
    model.eval()

    batch_size = 2
    seq_len = 8
    num_tokens = 32  # 2 cams * 16 patches
    cached_features = torch.randn(batch_size, num_tokens, 128)
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    labels = torch.randn(batch_size, 7)

    # Forward without pixel_values!
    preds, loss = model(
        pixel_values=None,
        visual_features=cached_features,
        input_ids=input_ids,
        labels=labels,
    )

    assert preds is not None
    assert preds.shape == (batch_size, 7)
    assert loss is not None
    assert loss.item() > 0.0
