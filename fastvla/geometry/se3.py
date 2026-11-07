"""
SE(3) Riemannian Manifold Operations for Robotic Manipulation.

Decomposes 7-DoF / 8-DoF robot actions into Riemannian product manifolds:
    SE(3) x [0, 1] ~ R^3 (translation) x SO(3) / S^3 (rotation) x R (gripper)

Under standard Euclidean flow matching, linear interpolation of Euler angles
creates gimbal singularities and non-constant angular velocities (high jerk).
Under Riemannian geodesic flow matching (Spherical Linear Interpolation / SLERP),
the angular velocity is strictly constant along the probability path,
yielding zero angular acceleration variance.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F


def euler_to_quaternion(rpy: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) in radians to unit quaternions (w, x, y, z).
    Convention: ZYX (yaw-pitch-roll intrinsic / roll-pitch-yaw extrinsic).

    Args:
        rpy: [..., 3] Euler angles in radians.
    Returns:
        q: [..., 4] Unit quaternions [w, x, y, z].
    """
    phi = rpy[..., 0] * 0.5
    theta = rpy[..., 1] * 0.5
    psi = rpy[..., 2] * 0.5

    c_phi = torch.cos(phi)
    s_phi = torch.sin(phi)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)

    qw = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    qx = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
    qy = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
    qz = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

    return torch.stack([qw, qx, qy, qz], dim=-1)


def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.

    Args:
        q: [..., 4] Quaternions [w, x, y, z].
    Returns:
        rpy: [..., 3] Euler angles in radians in [-pi, pi].
    """
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation) - clamped to prevent NaN outside domain
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.asin(torch.clamp(sinp, -1.0 + 1e-7, 1.0 - 1e-7))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quaternion_slerp(
    q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Spherical Linear Interpolation (SLERP) on S^3 with continuous time derivative (velocity).

    Args:
        q0: [..., 4] Initial unit quaternion at t=0.
        q1: [..., 4] Target unit quaternion at t=1.
        t:  [..., 1] Time parameter in [0, 1].

    Returns:
        q_t: [..., 4] Unit quaternion along the geodesic at time t.
        v_t: [..., 4] Time derivative dq/dt at time t (tangent vector on T_{q_t} S^3).
    """
    # Normalize inputs to ensure they lie strictly on S^3
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True).clamp_min(1e-8)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True).clamp_min(1e-8)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # Shortest path: q and -q represent the same SO(3) rotation.
    # If dot < 0, flip q1 so we traverse the shortest geodesic arc.
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.where(dot < 0.0, -dot, dot)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # For nearly collinear quaternions (omega -> 0), fallback to normalized LERP
    small_mask = sin_omega < 1e-4

    # LERP fallback
    lerp_q = (1.0 - t) * q0 + t * q1
    lerp_q = lerp_q / torch.norm(lerp_q, dim=-1, keepdim=True).clamp_min(1e-8)
    v_lerp = q1 - q0

    # Geodesic SLERP
    s0 = torch.sin((1.0 - t) * omega) / sin_omega.clamp_min(1e-8)
    s1 = torch.sin(t * omega) / sin_omega.clamp_min(1e-8)
    slerp_q = s0 * q0 + s1 * q1

    # Exact analytic time derivative: d(q_t)/dt
    ds0 = -omega * torch.cos((1.0 - t) * omega) / sin_omega.clamp_min(1e-8)
    ds1 = omega * torch.cos(t * omega) / sin_omega.clamp_min(1e-8)
    v_slerp = ds0 * q0 + ds1 * q1

    q_t = torch.where(small_mask, lerp_q, slerp_q)
    v_t = torch.where(small_mask, v_lerp, v_slerp)

    return q_t, v_t


def quaternion_angular_velocity(q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
    """
    Compute physical 3D angular velocity omega in rad/s from quaternion and its time derivative.
    Relation: omega = 2 * q_dot * q^{-1}.
    """
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]

    dw = q_dot[..., 0]
    dx = q_dot[..., 1]
    dy = q_dot[..., 2]
    dz = q_dot[..., 3]

    wx = 2.0 * (dx * qw - dw * qx - dy * qz + dz * qy)
    wy = 2.0 * (dy * qw - dw * qy - dz * qx + dx * qz)
    wz = 2.0 * (dz * qw - dw * qz - dx * qy + dy * qx)

    return torch.stack([wx, wy, wz], dim=-1)


def se3_flow_interpolate(
    target: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    format: str = "7dof_euler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes SE(3) Riemannian probability path and target velocity field:
        SE(3) ~ R^3 (translation) x SO(3) (rotation) x R (gripper).

    Args:
        target: [..., D] Target ground truth action (e.g. D=7 or D=8, or chunked D=T*7).
        x0:     [..., D] Source noise action sample from prior.
        t:      [...] or [..., 1] Time scalar in [0, 1].
        format: '7dof_euler' (3 transl, 3 euler, 1 grip) or '8dof_quat' (3 transl, 4 quat, 1 grip).

    Returns:
        x_t:             [..., D] Interpolated action on SE(3) manifold at time t.
        target_velocity: [..., D] Target velocity field dx/dt along the geodesic.
    """
    orig_shape = target.shape
    orig_dtype = target.dtype
    D = orig_shape[-1]

    # Perform geometry math in float32 for rigorous numerical stability, avoiding float16 subnormal underflows
    target = target.float()
    x0 = x0.float()
    t = t.float()

    # Handle chunked actions (e.g., T * 7 or T * 8)
    elem_dim = 8 if format == "8dof_quat" else 7
    is_chunked = (D != elem_dim) and (D % elem_dim == 0)

    if is_chunked:
        target = target.view(*orig_shape[:-1], -1, elem_dim)
        x0 = x0.view(*orig_shape[:-1], -1, elem_dim)

    # Ensure t has compatible shape for broadcasting: [..., 1] or [..., 1, 1]
    if t.ndim < target.ndim:
        t_expand = t.view(*t.shape, *([1] * (target.ndim - t.ndim)))
    else:
        t_expand = t

    # 1. Translation: Linear optimal transport path in R^3
    p0 = x0[..., :3]
    p1 = target[..., :3]
    p_t = (1.0 - t_expand) * p0 + t_expand * p1
    v_p = p1 - p0

    # 2. Gripper: Linear optimal transport path in R^1
    g0 = x0[..., -1:]
    g1 = target[..., -1:]
    g_t = (1.0 - t_expand) * g0 + t_expand * g1
    v_g = g1 - g0

    # 3. Rotation: Geodesic SLERP on SO(3) / S^3
    if format == "8dof_quat":
        q0 = x0[..., 3:7]
        q1 = target[..., 3:7]
        q_t, v_q = quaternion_slerp(q0, q1, t_expand)
        r_t = q_t
        v_r = v_q
    else:  # '7dof_euler'
        r0 = x0[..., 3:6]
        r1 = target[..., 3:6]
        q0 = euler_to_quaternion(r0)
        q1 = euler_to_quaternion(r1)

        # Geodesic point at t
        q_t, _ = quaternion_slerp(q0, q1, t_expand)
        r_t = quaternion_to_euler(q_t)

        # Compute smooth Euler velocities via two-point symmetric evaluation around t
        delta = 1e-4
        t_plus = torch.clamp(t_expand + delta, 0.0, 1.0)
        t_minus = torch.clamp(t_expand - delta, 0.0, 1.0)
        dt = (t_plus - t_minus).clamp_min(1e-8)

        q_plus, _ = quaternion_slerp(q0, q1, t_plus)
        q_minus, _ = quaternion_slerp(q0, q1, t_minus)

        r_plus = quaternion_to_euler(q_plus)
        r_minus = quaternion_to_euler(q_minus)

        # Wrap angle difference to [-pi, pi] to avoid discontinuous derivative
        r_diff = torch.remainder((r_plus - r_minus) + torch.pi, 2.0 * torch.pi) - torch.pi
        v_r = r_diff / dt

    x_t = torch.cat([p_t, r_t, g_t], dim=-1)
    target_velocity = torch.cat([v_p, v_r, v_g], dim=-1)

    if is_chunked:
        x_t = x_t.view(*orig_shape)
        target_velocity = target_velocity.view(*orig_shape)

    return x_t.to(orig_dtype), target_velocity.to(orig_dtype)
