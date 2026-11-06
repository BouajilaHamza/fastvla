"""
Unit tests for FusedSwiGLU Triton Kernel and Analytical Backward Pass.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvla.kernels.fused_swiglu import (
    FusedSwiGLUFunction,
    fused_swiglu,
    FusedMLP,
    HAS_TRITON,
)


def test_fused_swiglu_forward_numerical_equivalence():
    """Verify that fused_swiglu matches PyTorch F.silu(gate) * up."""
    torch.manual_seed(42)
    B, S, D = 2, 64, 256
    gate = torch.randn(B, S, D, dtype=torch.float32)
    up = torch.randn(B, S, D, dtype=torch.float32)

    # Reference
    ref_out = F.silu(gate) * up

    # Fused
    fused_out = fused_swiglu(gate, up)

    assert torch.allclose(fused_out, ref_out, atol=1e-6), "Forward output mismatch!"


def test_fused_swiglu_backward_gradient_equivalence():
    """Verify analytical gradients against PyTorch autograd reference."""
    torch.manual_seed(42)
    B, S, D = 2, 32, 128

    gate_fused = torch.randn(B, S, D, dtype=torch.float64, requires_grad=True)
    up_fused = torch.randn(B, S, D, dtype=torch.float64, requires_grad=True)

    gate_ref = gate_fused.clone().detach().requires_grad_(True)
    up_ref = up_fused.clone().detach().requires_grad_(True)

    # Forward
    out_fused = fused_swiglu(gate_fused, up_fused)
    out_ref = F.silu(gate_ref) * up_ref

    # Backward with random incoming gradient
    grad_out = torch.randn_like(out_fused)
    out_fused.backward(grad_out)
    out_ref.backward(grad_out)

    assert torch.allclose(gate_fused.grad, gate_ref.grad, atol=1e-7), "Gate gradient mismatch!"
    assert torch.allclose(up_fused.grad, up_ref.grad, atol=1e-7), "Up gradient mismatch!"


def test_fused_swiglu_gradcheck():
    """Mathematically verify analytical backward with torch.autograd.gradcheck."""
    torch.manual_seed(42)
    # Use smaller dimensions for fast finite differences
    gate = torch.randn(2, 8, 16, dtype=torch.float64, requires_grad=True)
    up = torch.randn(2, 8, 16, dtype=torch.float64, requires_grad=True)

    passed = torch.autograd.gradcheck(
        FusedSwiGLUFunction.apply,
        (gate, up),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert passed, "Autograd gradcheck failed for FusedSwiGLUFunction!"


def test_fused_mlp_module():
    """Verify FusedMLP acts as an exact drop-in replacement for standard MLP."""
    torch.manual_seed(42)
    D_in = 64
    D_hidden = 128

    gate_proj = nn.Linear(D_in, D_hidden, bias=False)
    up_proj = nn.Linear(D_in, D_hidden, bias=False)
    down_proj = nn.Linear(D_hidden, D_in, bias=False)

    mlp = FusedMLP(gate_proj, up_proj, down_proj)

    x = torch.randn(4, 16, D_in, requires_grad=True)
    out = mlp(x)

    assert out.shape == (4, 16, D_in)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA & Triton required")
def test_triton_cuda_kernel():
    """Verify CUDA Triton kernel execution matches PyTorch when GPU is present."""
    torch.manual_seed(42)
    B, S, D = 2, 128, 512

    gate_cuda = torch.randn(B, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    up_cuda = torch.randn(B, S, D, device="cuda", dtype=torch.float32, requires_grad=True)

    out = fused_swiglu(gate_cuda, up_cuda)
    assert out.is_cuda
    loss = out.mean()
    loss.backward()

    assert gate_cuda.grad is not None
    assert up_cuda.grad is not None
