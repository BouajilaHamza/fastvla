"""Unit tests for custom Triton kernels (with CPU fallbacks)."""
import torch
import pytest
from fastvla.kernels import (
    multi_cam_pack_forward,
    vision_language_fusion_forward,
    action_decode_forward,
)

# Test parameters
BATCH_SIZE = 2
NUM_CAMS = 3
C, H, W = 3, 32, 32  # Smaller for CPU speed
SEQ_LENGTH = 16
HIDDEN_DIM = 64
ACTION_DIM = 7

class TestMultiCamPackKernel:
    """Tests for multi-camera packing kernel."""

    def test_forward(self):
        """Test forward pass of multi-camera packing."""
        x = torch.randn(BATCH_SIZE, NUM_CAMS, C, H, W)

        packed = multi_cam_pack_forward(x)

        assert packed.shape == (BATCH_SIZE, NUM_CAMS * C, H, W)

        for b in range(BATCH_SIZE):
            for c in range(NUM_CAMS):
                assert torch.allclose(
                    x[b, c],
                    packed[b, c*C:(c+1)*C],
                    atol=1e-6
                )

class TestVisionLanguageFusionKernel:
    """Tests for vision-language fusion kernel."""

    def test_forward(self):
        """Test forward pass of vision-language fusion."""
        visual = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
        text = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)

        fused = vision_language_fusion_forward(visual, text)

        assert fused.shape == (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
        assert not torch.allclose(fused, torch.zeros_like(fused), atol=1e-6)

class TestActionDecodeKernel:
    """Tests for action decoding kernel."""

    def test_forward(self):
        """Test forward pass of action decoding."""
        hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        weight1 = torch.randn(HIDDEN_DIM, 2*HIDDEN_DIM)
        bias1 = torch.randn(2*HIDDEN_DIM)
        weight2 = torch.randn(2*HIDDEN_DIM, ACTION_DIM)
        bias2 = torch.randn(ACTION_DIM)

        actions = action_decode_forward(hidden, weight1, bias1, weight2, bias2)

        assert actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert torch.all(actions >= -1.0 - 1e-6) and torch.all(actions <= 1.0 + 1e-6)

if __name__ == "__main__":
    pytest.main([__file__])
