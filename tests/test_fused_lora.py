"""
Tests for Fused 4-Bit LoRA Triton GEMM Kernel.

Verifies:
1. Exact numerical equivalence between FusedLoraBAddFunction and standard PyTorch LoRA.
2. Gradient correctness for both forward and backward passes (dH, dB, dBase).
3. In-place model patching with apply_fused_lora across multi-layer models.
4. Robustness across multiple LoRA ranks (r=8, 16, 32, 64) and hidden dimensions.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvla.kernels.fused_lora import (
    FusedLoraBAddFunction,
    fused_lora_b_add,
    FusedLoraLinear,
    apply_fused_lora,
    HAS_TRITON,
)


def _golden_lora_forward(h: torch.Tensor, b_weight: torch.Tensor, base_out: torch.Tensor, alpha: float) -> torch.Tensor:
    """Standard un-fused PyTorch forward pass."""
    lora_out = torch.matmul(h, b_weight.t()) * alpha
    return base_out + lora_out


@pytest.mark.parametrize("r", [8, 16, 32, 64])
@pytest.mark.parametrize("M, N", [(16, 128), (64, 512), (128, 1024)])
def test_fused_lora_forward_parity(r, M, N):
    """Test forward numerical parity against golden PyTorch reference."""
    device = "cuda" if (HAS_TRITON and torch.cuda.is_available()) else "cpu"
    dtype = torch.float32

    torch.manual_seed(42)
    h = torch.randn(M, r, device=device, dtype=dtype, requires_grad=True)
    b_weight = torch.randn(N, r, device=device, dtype=dtype, requires_grad=True)
    base_out = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
    alpha = 2.0

    # 1. Golden Reference
    golden_out = _golden_lora_forward(h, b_weight, base_out, alpha)

    # 2. Fused Kernel
    fused_out = fused_lora_b_add(h, b_weight, base_out, alpha)

    assert fused_out.shape == golden_out.shape
    max_diff = (fused_out - golden_out).abs().max().item()
    assert max_diff < 1e-4, f"Forward diff {max_diff} exceeded tolerance for r={r}, M={M}, N={N}"


@pytest.mark.parametrize("r", [8, 16, 32])
def test_fused_lora_backward_gradients(r):
    """Test analytical backward gradients for dH, dB, and dBase against PyTorch autograd."""
    device = "cuda" if (HAS_TRITON and torch.cuda.is_available()) else "cpu"
    dtype = torch.float32
    M, N = 32, 256
    alpha = 1.5

    torch.manual_seed(123)

    # Golden pass inputs
    h_ref = torch.randn(M, r, device=device, dtype=dtype, requires_grad=True)
    b_ref = torch.randn(N, r, device=device, dtype=dtype, requires_grad=True)
    base_ref = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)

    # Fused pass inputs (exact clones)
    h_fused = h_ref.clone().detach().requires_grad_(True)
    b_fused = b_ref.clone().detach().requires_grad_(True)
    base_fused = base_ref.clone().detach().requires_grad_(True)

    # Incoming gradient
    grad_out = torch.randn(M, N, device=device, dtype=dtype)

    # Forward
    out_ref = _golden_lora_forward(h_ref, b_ref, base_ref, alpha)
    out_fused = fused_lora_b_add(h_fused, b_fused, base_fused, alpha)

    # Backward
    out_ref.backward(grad_out)
    out_fused.backward(grad_out)

    # Verify gradients
    diff_h = (h_fused.grad - h_ref.grad).abs().max().item()
    diff_b = (b_fused.grad - b_ref.grad).abs().max().item()
    diff_base = (base_fused.grad - base_ref.grad).abs().max().item()

    assert diff_h < 1e-4, f"dH diff {diff_h} too high"
    assert diff_b < 1e-4, f"dB diff {diff_b} too high"
    assert diff_base < 1e-5, f"dBase diff {diff_base} too high"


class MockPeftLoraLayer(nn.Module):
    """Mock simulating peft.tuners.lora.Linear / Linear4bit."""

    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias=False)})
        self.scaling = {"default": alpha / r}
        self.active_adapters = ["default"]
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})

    def forward(self, x):
        base = self.base_layer(x)
        lora = self.lora_B["default"](self.lora_A["default"](x)) * self.scaling["default"]
        return base + lora


def test_apply_fused_lora_module_patching():
    """Verify that apply_fused_lora correctly patches multi-layer models."""
    class DummyTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # 7 standard target projections per layer
            self.q_proj = MockPeftLoraLayer(64, 64, r=16)
            self.k_proj = MockPeftLoraLayer(64, 64, r=16)
            self.v_proj = MockPeftLoraLayer(64, 64, r=16)
            self.o_proj = MockPeftLoraLayer(64, 64, r=16)
            self.gate_proj = MockPeftLoraLayer(64, 128, r=16)
            self.up_proj = MockPeftLoraLayer(64, 128, r=16)
            self.down_proj = MockPeftLoraLayer(128, 64, r=16)

    class DummyModel(nn.Module):
        def __init__(self, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([DummyTransformerBlock() for _ in range(num_layers)])

    model = DummyModel(num_layers=4)
    # Total LoRA layers = 4 * 7 = 28
    num_fused = apply_fused_lora(model)
    assert num_fused == 28, f"Expected 28 fused modules, got {num_fused}"

    # Verify that modules were converted to FusedLoraLinear
    for block in model.layers:
        assert isinstance(block.q_proj, FusedLoraLinear)
        assert isinstance(block.k_proj, FusedLoraLinear)
        assert isinstance(block.gate_proj, FusedLoraLinear)

    # Test forward pass through patched model
    x = torch.randn(2, 10, 64)
    y = model.layers[0].q_proj(x)
    assert y.shape == (2, 10, 64)


if __name__ == "__main__":
    print("Testing Fused LoRA Forward Parity...")
    for r in [8, 16, 32, 64]:
        for M, N in [(16, 128), (64, 512), (128, 1024)]:
            test_fused_lora_forward_parity(r, M, N)
    print("  ✅ All forward parity tests passed.")

    print("Testing Fused LoRA Backward Gradients...")
    for r in [8, 16, 32]:
        test_fused_lora_backward_gradients(r)
    print("  ✅ All backward gradient tests passed.")

    print("Testing apply_fused_lora Module Patching...")
    test_apply_fused_lora_module_patching()
    print("  ✅ All module patching tests passed.")

    print("\n🎉 ALL FUSED LORA TRITON TESTS PASSED PERFECTLY!")
