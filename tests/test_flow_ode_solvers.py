"""
Unit tests for Flow Matching Higher-Order ODE Solvers (Euler, Midpoint, Heun).
"""

import pytest
import torch
import torch.nn.functional as F

from fastvla.adapters.action_head import FlowMatchingActionHead


def test_ode_solvers_forward_shapes_and_bounds():
    """Verify that all solvers output [B, action_dim] bounded strictly in [-1, 1]."""
    torch.manual_seed(42)
    B, D_in, D_action = 4, 64, 14
    head = FlowMatchingActionHead(input_dim=D_in, action_dim=D_action, hidden_dim=32)

    hidden = torch.randn(B, D_in)

    for solver in ["euler", "midpoint", "heun"]:
        for steps in [2, 5, 10]:
            out = head(hidden, solver=solver, steps=steps)
            assert out.shape == (B, D_action), f"Solver {solver} output shape mismatch: {out.shape}"
            assert (out >= -1.0).all() and (out <= 1.0).all(), f"Solver {solver} output out of [-1, 1] bounds!"


def test_heun_higher_order_accuracy_vs_euler():
    """
    Verify mathematically that 2nd-order Heun predictor-corrector achieves
    closer approximation to the fine ODE solution than 1st-order Euler at low step counts.
    """
    torch.manual_seed(42)
    B, D_in, D_action = 4, 64, 7
    head = FlowMatchingActionHead(input_dim=D_in, action_dim=D_action, hidden_dim=64)

    hidden = torch.randn(B, D_in)
    x0 = torch.randn(B, D_action)

    # High-resolution 50-step reference trajectory
    out_ref = head(hidden, solver="euler", steps=50, initial_noise=x0)

    # 2-step Euler vs 2-step Heun
    out_euler_2 = head(hidden, solver="euler", steps=2, initial_noise=x0)
    out_heun_2 = head(hidden, solver="heun", steps=2, initial_noise=x0)

    l2_euler_2 = F.mse_loss(out_euler_2, out_ref).item()
    l2_heun_2 = F.mse_loss(out_heun_2, out_ref).item()

    print(f"\n2-Step MSE vs 50-Step Ref: Euler={l2_euler_2:.6f} | Heun={l2_heun_2:.6f}")
    assert l2_heun_2 <= l2_euler_2, (
        f"Expected Heun 2-step ({l2_heun_2:.6f}) to be more accurate than Euler 2-step ({l2_euler_2:.6f})"
    )


def test_midpoint_higher_order_accuracy_vs_euler():
    """Verify 2nd-order Midpoint method maintains higher accuracy than 1st-order Euler."""
    torch.manual_seed(42)
    B, D_in, D_action = 4, 64, 7
    head = FlowMatchingActionHead(input_dim=D_in, action_dim=D_action, hidden_dim=64)

    hidden = torch.randn(B, D_in)
    x0 = torch.randn(B, D_action)

    out_ref = head(hidden, solver="euler", steps=50, initial_noise=x0)
    out_euler_3 = head(hidden, solver="euler", steps=3, initial_noise=x0)
    out_midpoint_3 = head(hidden, solver="midpoint", steps=3, initial_noise=x0)

    l2_euler_3 = F.mse_loss(out_euler_3, out_ref).item()
    l2_midpoint_3 = F.mse_loss(out_midpoint_3, out_ref).item()

    print(f"\n3-Step MSE vs 50-Step Ref: Euler={l2_euler_3:.6f} | Midpoint={l2_midpoint_3:.6f}")
    assert l2_midpoint_3 <= l2_euler_3, (
        f"Expected Midpoint 3-step ({l2_midpoint_3:.6f}) to be more accurate than Euler 3-step ({l2_euler_3:.6f})"
    )


def test_invalid_solver_raises():
    """Verify ValueError is raised for unrecognized solver name."""
    head = FlowMatchingActionHead(input_dim=32, action_dim=7)
    hidden = torch.randn(2, 32)
    with pytest.raises(ValueError, match="Unknown ODE solver"):
        head(hidden, solver="invalid_solver")
