"""
Unit tests for Fused Flow-Matching Interpolation and Loss Kernels.
"""

import pytest
import torch
import torch.nn.functional as F

from fastvla.kernels.fused_flow_loss import (
    fused_flow_interpolate,
    fused_flow_mse_loss,
    FusedFlowMSELossFunction,
    HAS_TRITON,
)
from fastvla.adapters.action_head import FlowMatchingActionHead


def test_fused_flow_interpolate_numerical_equivalence():
    """Verify that fused_flow_interpolate exactly matches the linear path formula."""
    torch.manual_seed(42)
    B, D = 4, 112  # e.g., 16-chunk * 7-DoF
    target = torch.randn(B, D, dtype=torch.float32)
    x0 = torch.randn(B, D, dtype=torch.float32)
    t = torch.rand(B, dtype=torch.float32)

    # Reference
    t_exp = t.view(-1, 1)
    ref_xt = (1.0 - t_exp) * x0 + t_exp * target
    ref_vt = target - x0

    # Fused
    fused_xt, fused_vt = fused_flow_interpolate(target, x0, t)

    assert torch.allclose(fused_xt, ref_xt, atol=1e-6), "Interpolated sample xt mismatch!"
    assert torch.allclose(fused_vt, ref_vt, atol=1e-6), "Target velocity vt mismatch!"


def test_fused_flow_mse_loss_numerical_and_gradient():
    """Verify analytical loss and gradients of fused_flow_mse_loss."""
    torch.manual_seed(42)
    B, D = 4, 64

    pred_fused = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    target = torch.randn(B, D, dtype=torch.float64)

    pred_ref = pred_fused.clone().detach().requires_grad_(True)

    # Forward
    loss_fused = fused_flow_mse_loss(pred_fused, target)
    loss_ref = F.mse_loss(pred_ref, target)

    assert torch.allclose(loss_fused, loss_ref, atol=1e-7), "MSE Loss mismatch!"

    # Backward
    loss_fused.backward()
    loss_ref.backward()

    assert torch.allclose(pred_fused.grad, pred_ref.grad, atol=1e-7), "Gradient mismatch in pred_velocity!"


def test_fused_flow_mse_loss_gradcheck():
    """Mathematically verify analytical backward with torch.autograd.gradcheck."""
    torch.manual_seed(42)
    pred = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 16, dtype=torch.float64)

    passed = torch.autograd.gradcheck(
        FusedFlowMSELossFunction.apply,
        (pred, target),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert passed, "Autograd gradcheck failed for FusedFlowMSELossFunction!"


def test_flow_matching_action_head_compute_loss_integration():
    """Verify FlowMatchingActionHead.compute_loss works end-to-end with fused operations."""
    torch.manual_seed(42)
    head = FlowMatchingActionHead(input_dim=64, action_dim=7, hidden_dim=32)

    hidden = torch.randn(4, 64, requires_grad=True)
    targets = torch.randn(4, 7)

    loss = head.compute_loss(hidden, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

    loss.backward()
    assert hidden.grad is not None
    assert not torch.isnan(hidden.grad).any()
