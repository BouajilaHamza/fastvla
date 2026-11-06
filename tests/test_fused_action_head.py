"""
Tests for FusedActionChunkHead and In-SRAM Huber Loss.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvla.kernels.fused_action_head import FusedActionChunkHead, FusedActionHuberLossFunction
from fastvla.model import FastVLAModel, FastVLAConfig


def test_fused_action_head_forward():
    head = FusedActionChunkHead(input_dim=64, hidden_dim=32, action_dim=7, chunk_size=2)
    x = torch.randn(4, 64)
    out = head(x)
    assert out.shape == (4, 14)
    # Tanh outputs must be bounded in [-1, 1]
    assert (out >= -1.0).all() and (out <= 1.0).all()


def test_fused_huber_loss_numerical_equivalence():
    """
    Verify that FusedActionHuberLossFunction produces the exact same loss value
    as PyTorch's native nn.HuberLoss / nn.SmoothL1Loss.
    """
    torch.manual_seed(42)
    B, D_in, D_hid, D_out = 4, 32, 16, 7
    delta = 1.0

    x = torch.randn(B, D_in, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(D_in, D_hid, dtype=torch.float64, requires_grad=True)
    b1 = torch.randn(D_hid, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(D_hid, D_out, dtype=torch.float64, requires_grad=True)
    b2 = torch.randn(D_out, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(B, D_out, dtype=torch.float64)

    # 1. Custom Fused Loss
    fused_loss = FusedActionHuberLossFunction.apply(x, w1, b1, w2, b2, targets, delta)

    # 2. PyTorch Native Reference
    h1_ref = torch.relu(x @ w1 + b1)
    preds_ref = torch.tanh(h1_ref @ w2 + b2)
    ref_loss = F.huber_loss(preds_ref, targets, delta=delta, reduction="mean")

    assert torch.allclose(fused_loss, ref_loss, atol=1e-7), f"Fused: {fused_loss.item()}, Ref: {ref_loss.item()}"


def test_fused_huber_loss_backward_gradients():
    """
    Verify analytical backward pass against PyTorch autograd reference gradients.
    """
    torch.manual_seed(42)
    B, D_in, D_hid, D_out = 4, 32, 16, 7
    delta = 1.0

    # Tensors for fused computation
    x1 = torch.randn(B, D_in, dtype=torch.float64, requires_grad=True)
    w1_1 = torch.randn(D_in, D_hid, dtype=torch.float64, requires_grad=True)
    b1_1 = torch.randn(D_hid, dtype=torch.float64, requires_grad=True)
    w2_1 = torch.randn(D_hid, D_out, dtype=torch.float64, requires_grad=True)
    b2_1 = torch.randn(D_out, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(B, D_out, dtype=torch.float64)

    # Clone for reference computation
    x2 = x1.clone().detach().requires_grad_(True)
    w1_2 = w1_1.clone().detach().requires_grad_(True)
    b1_2 = b1_1.clone().detach().requires_grad_(True)
    w2_2 = w2_1.clone().detach().requires_grad_(True)
    b2_2 = b2_1.clone().detach().requires_grad_(True)

    # Forward
    loss_fused = FusedActionHuberLossFunction.apply(x1, w1_1, b1_1, w2_1, b2_1, targets, delta)
    
    h1_ref = torch.relu(x2 @ w1_2 + b1_2)
    preds_ref = torch.tanh(h1_ref @ w2_2 + b2_2)
    loss_ref = F.huber_loss(preds_ref, targets, delta=delta, reduction="mean")

    # Backward
    loss_fused.backward()
    loss_ref.backward()

    # Compare gradients
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6), "Gradient mismatch in x"
    assert torch.allclose(w1_1.grad, w1_2.grad, atol=1e-6), "Gradient mismatch in w1"
    assert torch.allclose(b1_1.grad, b1_2.grad, atol=1e-6), "Gradient mismatch in b1"
    assert torch.allclose(w2_1.grad, w2_2.grad, atol=1e-6), "Gradient mismatch in w2"
    assert torch.allclose(b2_1.grad, b2_2.grad, atol=1e-6), "Gradient mismatch in b2"


def test_model_with_fused_action_loss_head():
    """Verify FastVLAModel integrates seamlessly with head_type='fused_action_loss'."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        action_dim=7,
        chunk_size=1,
        head_type="fused_action_loss",
    )
    model = FastVLAModel(config)
    
    batch_size = 2
    cached_features = torch.randn(batch_size, 16, 64)
    input_ids = torch.randint(0, 100, (batch_size, 8))
    labels = torch.randn(batch_size, 7)

    preds, loss = model(
        visual_features=cached_features,
        input_ids=input_ids,
        labels=labels,
    )

    assert preds.shape == (batch_size, 7)
    assert loss is not None
    assert loss.item() > 0.0

    # Ensure backward works through the fused head
    loss.backward()
    for p in model.action_head.parameters():
        assert p.grad is not None
