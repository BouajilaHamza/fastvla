"""
Unit tests for 14-DoF / 16-DoF Bimanual and Humanoid Policy Support.
Verifies SE(3) x SE(3) product manifold flow matching, multi-camera processing,
and ALOHA / Unitree G1 action chunking.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from fastvla.geometry.se3 import se3_flow_interpolate
from fastvla.adapters.action_head import FlowMatchingActionHead
from fastvla.inference import FastVLAPolicyRunner, ActionChunkBuffer


def test_bimanual_se3_flow_interpolate_euler():
    """Verify 14-DoF bimanual SE(3) manifold interpolation for dual-arm manipulators."""
    torch.manual_seed(42)
    B = 4
    # 14-DoF: Left arm 7 [3 pos, 3 euler, 1 grip] + Right arm 7 [3 pos, 3 euler, 1 grip]
    x0 = torch.randn(B, 14)
    target = torch.randn(B, 14)
    t = torch.rand(B)

    x_t, v_t = se3_flow_interpolate(target, x0, t, format="14dof_bimanual_euler")

    assert x_t.shape == (B, 14)
    assert v_t.shape == (B, 14)
    assert not torch.isnan(x_t).any()
    assert not torch.isnan(v_t).any()


def test_bimanual_se3_flow_interpolate_chunked():
    """Verify chunked 16-step 14-DoF bimanual flow interpolation (16 * 14 = 224)."""
    torch.manual_seed(42)
    B = 4
    chunk_dim = 16 * 14
    x0 = torch.randn(B, chunk_dim)
    target = torch.randn(B, chunk_dim)
    t = torch.rand(B)

    x_t, v_t = se3_flow_interpolate(target, x0, t, format="14dof_bimanual_euler")

    assert x_t.shape == (B, chunk_dim)
    assert v_t.shape == (B, chunk_dim)
    assert not torch.isnan(x_t).any()
    assert not torch.isnan(v_t).any()


def test_bimanual_se3_flow_interpolate_quat():
    """Verify 16-DoF bimanual quaternion interpolation (8 left + 8 right)."""
    torch.manual_seed(42)
    B = 4
    # 16-DoF: Left [3 pos, 4 quat, 1 grip] + Right [3 pos, 4 quat, 1 grip]
    x0 = torch.randn(B, 16)
    target = torch.randn(B, 16)
    # Normalize quaternion components
    target[:, 3:7] = F.normalize(target[:, 3:7], dim=-1)
    target[:, 11:15] = F.normalize(target[:, 11:15], dim=-1)
    t = torch.rand(B)

    x_t, v_t = se3_flow_interpolate(target, x0, t, format="16dof_bimanual_quat")

    assert x_t.shape == (B, 16)
    assert v_t.shape == (B, 16)
    # Check that quaternions stay approximately on unit 3-sphere S^3
    q_l_norm = torch.norm(x_t[:, 3:7], dim=-1)
    q_r_norm = torch.norm(x_t[:, 11:15], dim=-1)
    assert torch.allclose(q_l_norm, torch.ones_like(q_l_norm), atol=1e-3)
    assert torch.allclose(q_r_norm, torch.ones_like(q_r_norm), atol=1e-3)


def test_flow_matching_action_head_bimanual():
    """Verify FlowMatchingActionHead trains and infers on 14-DoF bimanual actions."""
    torch.manual_seed(42)
    B, D_in, D_action = 4, 64, 14
    head = FlowMatchingActionHead(
        input_dim=D_in,
        action_dim=D_action,
        hidden_dim=32,
        manifold="se3",
        rotation_format="14dof_bimanual_euler",
    )

    hidden = torch.randn(B, D_in)
    targets = torch.randn(B, D_action)

    # Loss computation & backprop
    loss = head.compute_loss(hidden, targets)
    assert loss is not None and not torch.isnan(loss) and loss.item() > 0.0
    loss.backward()

    for name, param in head.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"

    # Forward rollout
    actions = head(hidden, solver="midpoint", steps=2)
    assert actions.shape == (B, D_action)
    assert (actions >= -1.0).all() and (actions <= 1.0).all()


def test_bimanual_multi_camera_inference_runner():
    """Verify FastVLAPolicyRunner processes 3-camera bimanual views (overhead + 2 wrists)."""
    runner = FastVLAPolicyRunner(
        model=None,
        action_dim=14,
        chunk_size=16,
        safety_config="aloha",
        mock_mode=True,
    )

    multi_cam = {
        "cam_high": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
        "cam_left_wrist": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
        "cam_right_wrist": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
    }

    # Preprocessed tensor should have shape [1, 3, 3, 224, 224] (batch 1, 3 views, 3 channels)
    tensor = runner.preprocess_image(multi_cam)
    assert tensor.shape == (1, 3, 3, 224, 224)

    # Test rolling step with 14-DoF
    action = runner.step(multi_cam, instruction="Lift box with both arms")
    assert action.shape == (14,)
    assert runner.telemetry.total_steps == 1
