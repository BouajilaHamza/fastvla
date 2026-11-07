"""
Unit and physical verification tests for SE(3) Riemannian Manifold Flow Matching.
Verifies SO(3) / S^3 geodesic paths, quaternion SLERP, and kinematic smoothness.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from fastvla.geometry.se3 import (
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_slerp,
    quaternion_angular_velocity,
    se3_flow_interpolate,
)
from fastvla.adapters.action_head import FlowMatchingActionHead


def test_euler_quaternion_roundtrip():
    """Verify conversion from Euler angles to quaternion and back within [-pi, pi]."""
    torch.manual_seed(42)
    # Random Euler angles avoiding gimbal lock pitch near +/- pi/2
    roll = torch.empty(50).uniform_(-math.pi + 0.1, math.pi - 0.1)
    pitch = torch.empty(50).uniform_(-math.pi / 2 + 0.1, math.pi / 2 - 0.1)
    yaw = torch.empty(50).uniform_(-math.pi + 0.1, math.pi - 0.1)
    rpy = torch.stack([roll, pitch, yaw], dim=-1)

    q = euler_to_quaternion(rpy)
    # Unit norm check
    norms = torch.norm(q, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Quaternions must have unit norm"

    rpy_rec = quaternion_to_euler(q)
    # Angle difference modulo 2pi
    diff = torch.remainder((rpy - rpy_rec) + math.pi, 2 * math.pi) - math.pi
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-4), (
        f"Euler roundtrip error too large: max diff = {diff.abs().max().item()}"
    )


def test_slerp_unit_norm():
    """Verify that all points along SLERP trajectory lie strictly on the 3-sphere S^3."""
    torch.manual_seed(42)
    B = 10
    q0 = F.normalize(torch.randn(B, 4), dim=-1)
    q1 = F.normalize(torch.randn(B, 4), dim=-1)

    for t_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        t = torch.full((B, 1), t_val)
        q_t, v_t = quaternion_slerp(q0, q1, t)
        norms = torch.norm(q_t, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"SLERP point at t={t_val} deviates from S^3: norm min={norms.min()}, max={norms.max()}"
        )


def test_slerp_kinematic_smoothness_vs_euclidean():
    """
    Physical verification: Along the Riemannian geodesic, angular speed is constant (zero angular acceleration).
    In contrast, Euclidean interpolation of Euler angles suffers from fluctuating angular velocities and high jerk.
    """
    torch.manual_seed(42)
    r0 = torch.tensor([[0.2, 0.3, 0.4]])
    r1 = torch.tensor([[1.2, 1.1, -1.5]])

    q0 = euler_to_quaternion(r0)
    q1 = euler_to_quaternion(r1)

    # 50 trajectory steps
    steps = 50
    ts = torch.linspace(0.05, 0.95, steps).view(-1, 1)
    q0_rep = q0.repeat(steps, 1)
    q1_rep = q1.repeat(steps, 1)

    # 1. Riemannian SLERP path
    q_t, v_t = quaternion_slerp(q0_rep, q1_rep, ts)
    omega_slerp = quaternion_angular_velocity(q_t, v_t)
    speed_slerp = torch.norm(omega_slerp, dim=-1)
    slerp_std = speed_slerp.std().item()

    # 2. Euclidean linear interpolation
    r0_rep = r0.repeat(steps, 1)
    r1_rep = r1.repeat(steps, 1)
    r_t = (1.0 - ts) * r0_rep + ts * r1_rep
    r_dot = r1_rep - r0_rep

    # Numerical quaternion velocity for Euclidean trajectory
    delta = 1e-4
    q_plus = euler_to_quaternion(r_t + delta * r_dot)
    q_minus = euler_to_quaternion(r_t - delta * r_dot)
    q_dot_euc = (q_plus - q_minus) / (2.0 * delta)
    omega_euc = quaternion_angular_velocity(euler_to_quaternion(r_t), q_dot_euc)
    speed_euc = torch.norm(omega_euc, dim=-1)
    euc_std = speed_euc.std().item()

    print(f"\nKinematic Smoothness (Angular Speed Std Dev):")
    print(f"  Riemannian SE(3) SLERP Std Dev: {slerp_std:.8f} rad/s")
    print(f"  Euclidean Euler Std Dev:        {euc_std:.8f} rad/s")
    print(f"  Smoothness Ratio:               {euc_std / max(slerp_std, 1e-9):.1f}x smoother")

    assert slerp_std < 1e-4, f"SLERP angular speed must be strictly constant, got std={slerp_std}"
    assert euc_std > 100 * slerp_std, "Euclidean interpolation must have significantly higher jerk/variance"


def test_se3_flow_interpolate_shapes_and_chunking():
    """Verify se3_flow_interpolate handles 7-DoF, 8-DoF, and chunked actions."""
    torch.manual_seed(42)
    B = 4

    # 7-DoF Euler
    x0_7 = torch.randn(B, 7)
    target_7 = torch.randn(B, 7)
    t = torch.rand(B)
    x_t, v_t = se3_flow_interpolate(target_7, x0_7, t, format="7dof_euler")
    assert x_t.shape == (B, 7)
    assert v_t.shape == (B, 7)

    # 8-DoF Quaternion
    x0_8 = torch.randn(B, 8)
    target_8 = torch.randn(B, 8)
    # Normalize quaternion portion
    target_8[:, 3:7] = F.normalize(target_8[:, 3:7], dim=-1)
    x_t_8, v_t_8 = se3_flow_interpolate(target_8, x0_8, t, format="8dof_quat")
    assert x_t_8.shape == (B, 8)
    assert v_t_8.shape == (B, 8)

    # Chunked 16-step 7-DoF (16 * 7 = 112)
    x0_chunk = torch.randn(B, 16 * 7)
    target_chunk = torch.randn(B, 16 * 7)
    x_t_chunk, v_t_chunk = se3_flow_interpolate(target_chunk, x0_chunk, t, format="7dof_euler")
    assert x_t_chunk.shape == (B, 112)
    assert v_t_chunk.shape == (B, 112)


def test_flow_matching_action_head_se3_end_to_end():
    """Verify FlowMatchingActionHead with SE(3) manifold trains and infers seamlessly."""
    torch.manual_seed(42)
    B, D_in, D_action = 4, 64, 7
    head = FlowMatchingActionHead(
        input_dim=D_in,
        action_dim=D_action,
        hidden_dim=32,
        manifold="se3",
        rotation_format="7dof_euler",
    )

    hidden = torch.randn(B, D_in)
    targets = torch.randn(B, D_action)

    # Training loss
    loss = head.compute_loss(hidden, targets)
    assert loss is not None and not torch.isnan(loss) and loss.item() > 0.0

    # Gradient backprop
    loss.backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients!"

    # Forward inference with 2-step Midpoint ODE solver
    actions = head(hidden, solver="midpoint", steps=2)
    assert actions.shape == (B, D_action)
    assert (actions >= -1.0).all() and (actions <= 1.0).all()
