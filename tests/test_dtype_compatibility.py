"""
Comprehensive test suite for FastVLA dtype handling and mixed precision compatibility.
Tests all critical paths: CPU/GPU, float16/float32, all action heads, full model.
"""
import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastvla.kernels.action_head import TritonActionHead
from fastvla.adapters.action_head import (
    DiscreteActionHead,
    ContinuousActionHead,
    FlowMatchingActionHead,
)
from fastvla.kernels.fusion import vision_language_fusion_forward
from fastvla.data.collator import UnslothVLACollator


# ── Helper Functions ────────────────────────────────────────────────────


def check_forward_pass(module, input_tensor, expected_output_shape, name="module"):
    """Test forward pass and validate output shape."""
    with torch.no_grad():
        output = module(input_tensor)
    assert output.shape == expected_output_shape, (
        f"{name} output shape mismatch: expected {expected_output_shape}, got {output.shape}"
    )
    assert output.dtype == input_tensor.dtype or output.dtype == torch.float32, (
        f"{name} unexpected output dtype: {output.dtype}"
    )
    assert not torch.isnan(output).any(), f"{name} output contains NaN"
    return output


def check_backward_pass(module, input_tensor, name="module"):
    """Test backward pass produces valid gradients."""
    input_tensor = input_tensor.clone().requires_grad_(True)
    output = module(input_tensor)
    loss = output.sum()
    loss.backward()
    
    assert input_tensor.grad is not None, f"{name} has no gradient"
    assert not torch.isnan(input_tensor.grad).any(), f"{name} gradient contains NaN"
    assert not torch.isinf(input_tensor.grad).any(), f"{name} gradient contains Inf"
    return input_tensor.grad


# ── TritonActionHead Tests ─────────────────────────────────────────────


class TestTritonActionHead:
    """Test TritonActionHead with various dtypes."""

    @pytest.fixture
    def action_head(self):
        return TritonActionHead(input_dim=768, hidden_dim=256, output_dim=7)

    def test_cpu_float32_forward(self, action_head):
        """CPU float32 forward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        output = check_forward_pass(
            action_head, x, (2, 7), "TritonActionHead CPU float32"
        )
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            "TritonActionHead output not in [-1, 1]"
        )

    def test_cpu_float16_forward(self, action_head):
        """CPU float16 forward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        output = check_forward_pass(
            action_head, x, (2, 7), "TritonActionHead CPU float16"
        )
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            "TritonActionHead output not in [-1, 1]"
        )

    def test_cpu_float32_backward(self, action_head):
        """CPU float32 backward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        grad = check_backward_pass(
            action_head, x, "TritonActionHead CPU float32"
        )
        assert grad.shape == (2, 768)

    def test_cpu_float16_backward(self, action_head):
        """CPU float16 backward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        grad = check_backward_pass(
            action_head, x, "TritonActionHead CPU float16"
        )
        assert grad.shape == (2, 768)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_float32_forward(self, action_head):
        """GPU float32 forward pass."""
        action_head = action_head.cuda()
        x = torch.randn(2, 768, dtype=torch.float32, device='cuda')
        output = check_forward_pass(
            action_head, x, (2, 7), "TritonActionHead GPU float32"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_float16_forward(self, action_head):
        """GPU float16 forward pass (mixed precision)."""
        action_head = action_head.cuda()
        x = torch.randn(2, 768, dtype=torch.float16, device='cuda')
        output = check_forward_pass(
            action_head, x, (2, 7), "TritonActionHead GPU float16"
        )


# ── DiscreteActionHead Tests ────────────────────────────────────────────


class TestDiscreteActionHead:
    """Test DiscreteActionHead with various dtypes."""

    @pytest.fixture
    def action_head(self):
        return DiscreteActionHead(input_dim=768, action_dim=7, hidden_dim=256, num_bins=256)

    def test_cpu_float32_forward(self, action_head):
        """CPU float32 forward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        output = check_forward_pass(
            action_head, x, (2, 7), "DiscreteActionHead CPU float32"
        )
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            "DiscreteActionHead output not in [0, 1]"
        )

    def test_cpu_float16_forward(self, action_head):
        """CPU float16 forward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        output = check_forward_pass(
            action_head, x, (2, 7), "DiscreteActionHead CPU float16"
        )
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            "DiscreteActionHead output not in [0, 1]"
        )

    def test_cpu_float32_backward(self, action_head):
        """CPU float32 backward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        grad = check_backward_pass(
            action_head, x, "DiscreteActionHead CPU float32"
        )
        assert grad.shape == (2, 768)

    def test_cpu_float16_backward(self, action_head):
        """CPU float16 backward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        grad = check_backward_pass(
            action_head, x, "DiscreteActionHead CPU float16"
        )
        assert grad.shape == (2, 768)


# ── ContinuousActionHead Tests ──────────────────────────────────────────


class TestContinuousActionHead:
    """Test ContinuousActionHead with various dtypes."""

    @pytest.fixture
    def action_head(self):
        return ContinuousActionHead(input_dim=768, action_dim=7, hidden_dim=256, use_triton=False)

    def test_cpu_float32_forward(self, action_head):
        """CPU float32 forward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        output = check_forward_pass(
            action_head, x, (2, 7), "ContinuousActionHead CPU float32"
        )
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            "ContinuousActionHead output not in [-1, 1]"
        )

    def test_cpu_float16_forward(self, action_head):
        """CPU float16 forward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        output = check_forward_pass(
            action_head, x, (2, 7), "ContinuousActionHead CPU float16"
        )
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            "ContinuousActionHead output not in [-1, 1]"
        )

    def test_cpu_float32_backward(self, action_head):
        """CPU float32 backward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        grad = check_backward_pass(
            action_head, x, "ContinuousActionHead CPU float32"
        )
        assert grad.shape == (2, 768)

    def test_cpu_float16_backward(self, action_head):
        """CPU float16 backward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        grad = check_backward_pass(
            action_head, x, "ContinuousActionHead CPU float16"
        )
        assert grad.shape == (2, 768)


# ── FlowMatchingActionHead Tests ────────────────────────────────────────


class TestFlowMatchingActionHead:
    """Test FlowMatchingActionHead with various dtypes."""

    @pytest.fixture
    def action_head(self):
        return FlowMatchingActionHead(input_dim=768, action_dim=7, hidden_dim=256)

    def test_cpu_float32_forward(self, action_head):
        """CPU float32 forward pass."""
        x = torch.randn(2, 768, dtype=torch.float32)
        output = check_forward_pass(
            action_head, x, (2, 7), "FlowMatchingActionHead CPU float32"
        )
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            "FlowMatchingActionHead output not in [-1, 1]"
        )

    def test_cpu_float16_forward(self, action_head):
        """CPU float16 forward pass (mixed precision simulation)."""
        x = torch.randn(2, 768, dtype=torch.float16)
        output = check_forward_pass(
            action_head, x, (2, 7), "FlowMatchingActionHead CPU float16"
        )
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            "FlowMatchingActionHead output not in [-1, 1]"
        )


# ── Vision-Language Fusion Tests ────────────────────────────────────────


class TestVisionLanguageFusion:
    """Test vision-language fusion with various dtypes."""

    def test_cpu_float32_fusion(self):
        """CPU float32 fusion."""
        visual = torch.randn(2, 196, 768, dtype=torch.float32)
        text = torch.randn(2, 196, 768, dtype=torch.float32)
        output = vision_language_fusion_forward(visual, text)
        assert output.shape == (2, 196, 768)
        assert not torch.isnan(output).any()

    def test_cpu_float16_fusion(self):
        """CPU float16 fusion (mixed precision simulation)."""
        visual = torch.randn(2, 196, 768, dtype=torch.float16)
        text = torch.randn(2, 196, 768, dtype=torch.float16)
        output = vision_language_fusion_forward(visual, text)
        assert output.shape == (2, 196, 768)
        assert not torch.isnan(output).any()

    def test_mixed_dtype_fusion(self):
        """Fusion with mismatched visual/text dtypes."""
        visual = torch.randn(2, 196, 768, dtype=torch.float16)
        text = torch.randn(2, 196, 768, dtype=torch.float32)
        output = vision_language_fusion_forward(visual, text)
        assert output.shape == (2, 196, 768)
        assert not torch.isnan(output).any()

    def test_sequence_length_mismatch(self):
        """Fusion with mismatched sequence lengths."""
        visual = torch.randn(2, 100, 768, dtype=torch.float32)
        text = torch.randn(2, 196, 768, dtype=torch.float32)
        output = vision_language_fusion_forward(visual, text)
        assert output.shape == (2, 196, 768)
        assert not torch.isnan(output).any()


# ── Data Collator Tests ─────────────────────────────────────────────────


class TestDataCollator:
    """Test data collator with various scenarios."""

    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
                }
        
        return MockTokenizer()

    def test_collator_with_all_fields(self, mock_tokenizer):
        """Collator with complete data."""
        collator = UnslothVLACollator(tokenizer=mock_tokenizer, action_dim=7)
        features = [
            {
                "images": {"cam1": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                "instructions": "pick up the block",
            }
            for _ in range(2)
        ]
        batch = collator(features)
        assert "pixel_values" in batch
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["pixel_values"].shape == (2, 1, 3, 224, 224)
        assert batch["labels"].shape == (2, 7)

    def test_collator_missing_instructions(self, mock_tokenizer):
        """Collator handles missing instructions gracefully."""
        collator = UnslothVLACollator(tokenizer=mock_tokenizer, action_dim=7)
        features = [
            {
                "images": {"cam1": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                # No "instructions" key
            }
            for _ in range(2)
        ]
        batch = collator(features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2

    def test_collator_missing_images_raises(self, mock_tokenizer):
        """Collator raises error when images are missing."""
        collator = UnslothVLACollator(tokenizer=mock_tokenizer, action_dim=7)
        features = [
            {
                "actions": torch.randn(7),
                "instructions": "pick up the block",
                # No "images" key
            }
            for _ in range(2)
        ]
        with pytest.raises(ValueError, match="missing required keys"):
            collator(features)


# ── Integration Tests ───────────────────────────────────────────────────


class TestMixedPrecisionIntegration:
    """Test mixed precision compatibility end-to-end."""

    def test_action_head_autocast_simulation_cpu(self):
        """Simulate autocast behavior on CPU (float16 input to float32 model)."""
        head = TritonActionHead(input_dim=256, hidden_dim=64, output_dim=5)
        
        # Simulate autocast providing float16 input
        x_fp16 = torch.randn(4, 256, dtype=torch.float16)
        
        # This should NOT raise dtype mismatch
        output = head(x_fp16)
        
        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()

    def test_action_head_dtype_consistency(self):
        """Verify float16 and float32 produce similar outputs."""
        head = TritonActionHead(input_dim=128, hidden_dim=32, output_dim=3)
        torch.manual_seed(42)
        
        x_fp32 = torch.randn(2, 128, dtype=torch.float32)
        x_fp16 = x_fp32.to(torch.float16)
        
        with torch.no_grad():
            out_fp32 = head(x_fp32)
            out_fp16 = head(x_fp16).to(torch.float32)
        
        # Outputs should be numerically similar (allowing for precision loss)
        assert torch.allclose(out_fp32, out_fp16, rtol=1e-2, atol=1e-2), (
            f"float16/float32 outputs differ significantly: max diff = {(out_fp32 - out_fp16).abs().max()}"
        )


# ── Run Tests ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
