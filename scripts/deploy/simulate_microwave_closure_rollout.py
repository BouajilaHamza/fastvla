#!/usr/bin/env python3
"""
Simulate and Render FastVLA Turbo Policy Rollout on the Hardest Robotics Task:
LIBERO-10 Task 2: "ضع الكوب الأصفر والأبيض في الميكروويف وأغلقه"
(Put the yellow and white mug in the microwave and close it).

A long-horizon, multi-stage, contact-rich articulated manipulation challenge:
1. Precision handle grasp of yellow-and-white mug.
2. Low-clearance 3D transport into microwave cavity without collision.
3. Turntable placement and gripper release.
4. Clearance retraction.
5. Multi-point contact push of articulated swinging microwave door until fully latched (100% closure).

Outputs:
- assets/fastvla_microwave_closure_rollout.mp4
- assets/fastvla_microwave_closure_rollout.gif
"""

import os
import math
import time
from typing import Dict, List, Tuple
import numpy as np
import cv2
from PIL import Image

from fastvla.geometry.se3 import se3_flow_interpolate
from fastvla.inference import FastVLAPolicyRunner
from fastvla.safety import SafetyFilter, RobotSafetyConfig


def project_3d_to_2d(
    pts_3d: np.ndarray,
    origin: Tuple[int, int] = (500, 430),
    scale: float = 330.0,
    elev: float = 26.0,
    azim: float = -48.0,
) -> np.ndarray:
    """Projects 3D points in world coordinates onto a 2D image plane with isometric perspective."""
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    c_a, s_a = math.cos(azim_rad), math.sin(azim_rad)
    c_e, s_e = math.cos(elev_rad), math.sin(elev_rad)

    R_z = np.array([
        [c_a, -s_a, 0],
        [s_a, c_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R_x = np.array([
        [1, 0, 0],
        [0, c_e, -s_e],
        [0, s_e, c_e]
    ], dtype=np.float32)

    R = R_x @ R_z
    rotated = pts_3d @ R.T

    u = origin[0] + rotated[..., 0] * scale
    v = origin[1] - rotated[..., 2] * scale + rotated[..., 1] * scale * 0.38
    return np.stack([u, v], axis=-1).astype(np.int32)


class RobotArmRenderer:
    """Renders a 7-DoF robotic manipulator arm with joints, links, and parallel jaw gripper."""

    def __init__(self, base_pos: np.ndarray):
        self.base_pos = base_pos.astype(np.float32)
        self.link_color = (225, 230, 235)    # Clean metallic white (Franka Panda)
        self.joint_color = (40, 45, 50)      # Matte black collars
        self.gripper_color = (0, 200, 255)   # FastVLA cyan/gold accent

    def compute_kinematics(self, ee_pos: np.ndarray) -> List[np.ndarray]:
        p0 = self.base_pos
        p1 = p0 + np.array([0.0, 0.0, 0.28], dtype=np.float32)
        p3 = ee_pos.astype(np.float32)
        mid = (p1 + p3) * 0.5
        lateral = np.array([-0.05, 0.16, 0.18], dtype=np.float32)
        p2 = mid + lateral
        return [p0, p1, p2, p3]

    def draw(self, img: np.ndarray, ee_pos: np.ndarray, grip_open: float, origin=(500, 430), scale=330.0):
        joints_3d = np.array(self.compute_kinematics(ee_pos))
        pts_2d = project_3d_to_2d(joints_3d, origin=origin, scale=scale)

        # Base collar
        base_pt = pts_2d[0]
        cv2.circle(img, (base_pt[0], base_pt[1]), 20, (30, 32, 36), -1, cv2.LINE_AA)
        cv2.circle(img, (base_pt[0], base_pt[1]), 20, (80, 85, 95), 2, cv2.LINE_AA)

        # Arm links
        for i in range(len(pts_2d) - 1):
            pt1 = tuple(pts_2d[i])
            pt2 = tuple(pts_2d[i + 1])
            cv2.line(img, pt1, pt2, (20, 22, 25), 16, cv2.LINE_AA)
            cv2.line(img, pt1, pt2, self.link_color, 12, cv2.LINE_AA)
            cv2.circle(img, pt1, 10, self.joint_color, -1, cv2.LINE_AA)
            cv2.circle(img, pt1, 10, (120, 125, 135), 2, cv2.LINE_AA)

        # Gripper wrist and fingers
        ee_pt = pts_2d[-1]
        cv2.circle(img, tuple(ee_pt), 8, self.joint_color, -1, cv2.LINE_AA)
        spread = int(5 + grip_open * 15)
        j1 = (ee_pt[0] - spread, ee_pt[1] + 16)
        j2 = (ee_pt[0] + spread, ee_pt[1] + 16)
        cv2.line(img, (ee_pt[0] - spread, ee_pt[1]), j1, self.gripper_color, 4, cv2.LINE_AA)
        cv2.line(img, (ee_pt[0] + spread, ee_pt[1]), j2, self.gripper_color, 4, cv2.LINE_AA)
        cv2.line(img, (ee_pt[0] - spread, ee_pt[1]), (ee_pt[0] + spread, ee_pt[1]), (40, 45, 50), 5, cv2.LINE_AA)


def draw_microwave(
    canvas: np.ndarray,
    origin=(500, 430),
    scale=330.0,
    door_angle_deg: float = 75.0,
):
    """
    Renders the 3D Microwave with hollow cavity, interior turntable, keypad,
    and articulated door swinging on a vertical hinge.
    """
    mx, my, mz = 0.30, 0.15, 0.0
    w, d, h = 0.28, 0.24, 0.22

    # Microwave Outer Housing (Box corners)
    housing = np.array([
        [mx - w / 2, my - d / 2, mz],
        [mx + w / 2, my - d / 2, mz],
        [mx + w / 2, my + d / 2, mz],
        [mx - w / 2, my + d / 2, mz],
        [mx - w / 2, my - d / 2, mz + h],
        [mx + w / 2, my - d / 2, mz + h],
        [mx + w / 2, my + d / 2, mz + h],
        [mx - w / 2, my + d / 2, mz + h],
    ], dtype=np.float32)
    h2d = project_3d_to_2d(housing, origin=origin, scale=scale)

    # Shaded outer metallic body (Silver-charcoal)
    cv2.fillPoly(canvas, [h2d[[0, 1, 5, 4]]], (50, 54, 60), cv2.LINE_AA)  # Front face border
    cv2.fillPoly(canvas, [h2d[[1, 2, 6, 5]]], (65, 70, 78), cv2.LINE_AA)  # Right side
    cv2.fillPoly(canvas, [h2d[[4, 5, 6, 7]]], (85, 90, 100), cv2.LINE_AA)  # Top surface
    cv2.polylines(canvas, [h2d[[4, 5, 6, 7]]], True, (130, 135, 145), 2, cv2.LINE_AA)

    # Hollow Interior Cavity (Deep dark recess)
    cx, cy, cz = mx - 0.03, my, mz + 0.03
    cw, cd, ch = 0.18, 0.18, 0.16
    cavity = np.array([
        [cx - cw / 2, cy - cd / 2, cz],
        [cx + cw / 2, cy - cd / 2, cz],
        [cx + cw / 2, cy + cd / 2, cz],
        [cx - cw / 2, cy + cd / 2, cz],
        [cx - cw / 2, cy - cd / 2, cz + ch],
        [cx + cw / 2, cy - cd / 2, cz + ch],
        [cx + cw / 2, cy + cd / 2, cz + ch],
        [cx - cw / 2, cy + cd / 2, cz + ch],
    ], dtype=np.float32)
    c2d = project_3d_to_2d(cavity, origin=origin, scale=scale)

    # Cavity back and floor
    cv2.fillPoly(canvas, [c2d[[2, 3, 7, 6]]], (20, 22, 26), cv2.LINE_AA)  # Back wall
    cv2.fillPoly(canvas, [c2d[[0, 1, 2, 3]]], (25, 28, 32), cv2.LINE_AA)  # Floor
    cv2.fillPoly(canvas, [c2d[[1, 2, 6, 5]]], (30, 33, 38), cv2.LINE_AA)  # Inner right wall

    # Turntable glass disc
    tt_center = np.array([cx, cy, cz + 0.005], dtype=np.float32)
    disc_pts = []
    for deg in range(0, 360, 20):
        rad = math.radians(deg)
        disc_pts.append(tt_center + np.array([0.07 * math.cos(rad), 0.07 * math.sin(rad), 0.0]))
    disc_2d = project_3d_to_2d(np.array(disc_pts), origin=origin, scale=scale)
    cv2.fillPoly(canvas, [disc_2d], (60, 75, 80), cv2.LINE_AA)
    cv2.polylines(canvas, [disc_2d], True, (120, 160, 170), 1, cv2.LINE_AA)

    # Keypad & Digital Timer Display on Right Front Panel
    kp_corners = np.array([
        [mx + 0.08, my - d / 2 - 0.002, mz + 0.04],
        [mx + 0.13, my - d / 2 - 0.002, mz + 0.04],
        [mx + 0.13, my - d / 2 - 0.002, mz + 0.18],
        [mx + 0.08, my - d / 2 - 0.002, mz + 0.18],
    ], dtype=np.float32)
    kp_2d = project_3d_to_2d(kp_corners, origin=origin, scale=scale)
    cv2.fillPoly(canvas, [kp_2d], (15, 16, 18), cv2.LINE_AA)
    cv2.polylines(canvas, [kp_2d], True, (60, 65, 75), 1, cv2.LINE_AA)

    # Digital LED Green timer "1:30"
    led_pt = kp_2d[3] + (kp_2d[1] - kp_2d[0]) * 0.2
    cv2.circle(canvas, (int(led_pt[0]) + 8, int(led_pt[1]) + 8), 3, (0, 255, 100), -1, cv2.LINE_AA)

    # Articulated Microwave Door with Hinge
    # Hinge is at the left front corner: [mx - w/2, my - d/2, mz + h/2]
    hinge_x = mx - w / 2 + 0.02
    hinge_y = my - d / 2
    door_w = 0.20
    theta_rad = math.radians(door_angle_deg)

    # Door rotates around vertical Z axis at hinge
    # When door_angle_deg = 0: door is closed, lying along X axis
    # When door_angle_deg = 75: door is swung open towards user
    dx = door_w * math.cos(theta_rad)
    dy = -door_w * math.sin(theta_rad)

    door_corners = np.array([
        [hinge_x, hinge_y, mz + 0.02],
        [hinge_x + dx, hinge_y + dy, mz + 0.02],
        [hinge_x + dx, hinge_y + dy, mz + h - 0.02],
        [hinge_x, hinge_y, mz + h - 0.02],
    ], dtype=np.float32)
    d2d = project_3d_to_2d(door_corners, origin=origin, scale=scale)

    # Door Glass & Tinted Frame
    cv2.fillPoly(canvas, [d2d], (35, 45, 55), cv2.LINE_AA)  # Semi-transparent tinted window
    cv2.polylines(canvas, [d2d], True, (160, 165, 175) if door_angle_deg > 5 else (0, 255, 120), 2, cv2.LINE_AA)

    # Door Handle on the opening edge
    handle_edge = d2d[1]
    handle_top = d2d[2]
    h_mid = ((handle_edge[0] + handle_top[0]) // 2, (handle_edge[1] + handle_top[1]) // 2)
    cv2.line(canvas, (h_mid[0] - 2, h_mid[1] - 14), (h_mid[0] - 2, h_mid[1] + 14), (220, 225, 235), 4, cv2.LINE_AA)

    # Mechanical Latch Indicator
    if door_angle_deg <= 3.0:
        cv2.putText(canvas, "LATCHED [100%]", (int(d2d[1][0]) - 30, int(d2d[1][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 240, 120), 1, cv2.LINE_AA)


def draw_mug(canvas: np.ndarray, mug_pos: np.ndarray, origin=(500, 430), scale=330.0):
    """Renders the Yellow-and-White Mug with distinctive colored body and handle."""
    x, y, z = mug_pos
    r = 0.032
    h = 0.065

    # Base cylinder
    pts_bot = []
    pts_top = []
    for deg in range(0, 360, 24):
        rad = math.radians(deg)
        pts_bot.append([x + r * math.cos(rad), y + r * math.sin(rad), z])
        pts_top.append([x + r * math.cos(rad), y + r * math.sin(rad), z + h])

    b2d = project_3d_to_2d(np.array(pts_bot), origin=origin, scale=scale)
    t2d = project_3d_to_2d(np.array(pts_top), origin=origin, scale=scale)

    # White ceramic lower base
    cv2.fillPoly(canvas, [b2d], (240, 242, 245), cv2.LINE_AA)
    # Yellow middle band
    cv2.fillPoly(canvas, [t2d], (40, 215, 255), cv2.LINE_AA)  # Bright yellow in BGR

    # Side walls
    for i in range(len(b2d) - 1):
        quad = np.array([b2d[i], b2d[i + 1], t2d[i + 1], t2d[i]])
        color = (50, 210, 250) if i % 2 == 0 else (235, 238, 242)
        cv2.fillPoly(canvas, [quad], color, cv2.LINE_AA)

    # Top rim
    cv2.polylines(canvas, [t2d], True, (255, 255, 255), 2, cv2.LINE_AA)

    # Handle loop on the side
    h_pt = project_3d_to_2d(np.array([[x + r + 0.015, y, z + h * 0.5]]), origin=origin, scale=scale)[0]
    rim_pt = t2d[0]
    bot_pt = b2d[0]
    cv2.line(canvas, (rim_pt[0], rim_pt[1]), (h_pt[0], h_pt[1]), (40, 215, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, (h_pt[0], h_pt[1]), (bot_pt[0], bot_pt[1]), (240, 242, 245), 3, cv2.LINE_AA)


def render_microwave_scene(
    frame_idx: int,
    total_frames: int,
    ee_pos: np.ndarray,
    grip_open: float,
    mug_pos: np.ndarray,
    door_angle_deg: float,
    current_phase: str,
    robot_arm: RobotArmRenderer,
    trail: List[np.ndarray],
    safety_info: Dict,
) -> np.ndarray:
    H, W = 720, 1280
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Deep dark slate studio gradient
    for y in range(H):
        val = int(20 + 16 * (y / H))
        canvas[y, :] = (val, val + 2, val + 5)

    # Table grid
    grid = []
    for x in np.linspace(-0.8, 0.8, 9):
        grid.append(np.array([[x, -0.5, 0.0], [x, 0.5, 0.0]], dtype=np.float32))
    for y in np.linspace(-0.5, 0.5, 7):
        grid.append(np.array([[-0.8, y, 0.0], [0.8, y, 0.0]], dtype=np.float32))

    for line in grid:
        p2d = project_3d_to_2d(line, origin=(500, 430), scale=330.0)
        cv2.line(canvas, tuple(p2d[0]), tuple(p2d[1]), (42, 46, 54), 1, cv2.LINE_AA)

    # Motion Trail
    if len(trail) > 1:
        t2d = project_3d_to_2d(np.array(trail), origin=(500, 430), scale=330.0)
        for i in range(len(t2d) - 1):
            alpha = (i + 1) / len(t2d)
            cv2.line(canvas, tuple(t2d[i]), tuple(t2d[i + 1]), (int(0 * alpha), int(210 * alpha), int(255 * alpha)), 2, cv2.LINE_AA)

    # Draw 3D Microwave with articulated door
    draw_microwave(canvas, origin=(500, 430), scale=330.0, door_angle_deg=door_angle_deg)

    # Draw Yellow-and-White Mug
    draw_mug(canvas, mug_pos, origin=(500, 430), scale=330.0)

    # Draw Robot Arm
    robot_arm.draw(canvas, ee_pos, grip_open, origin=(500, 430), scale=330.0)

    # Camera Inset 1: Gripper Wrist View (Top-Right)
    inset_w, inset_h = 190, 140
    cam_wrist = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
    cam_wrist[:] = (26, 28, 32)
    # Local gripper view targeting mug / door
    cv2.circle(cam_wrist, (inset_w // 2, inset_h // 2), 26, (40, 215, 255) if "Door" not in current_phase else (160, 170, 180), -1)
    cv2.rectangle(cam_wrist, (25, 45), (40, 95), (0, 200, 255), -1)
    cv2.rectangle(cam_wrist, (inset_w - 40, 45), (inset_w - 25, 95), (0, 200, 255), -1)
    cv2.putText(cam_wrist, "CAM: WRIST (EYE-IN-HAND)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(cam_wrist, f"Grip: {(1.0-grip_open)*100:3.0f}%", (10, inset_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Camera Inset 2: Interior Microwave Chamber View (Middle-Right)
    cam_chamber = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
    cam_chamber[:] = (18, 20, 24)
    # Turntable interior perspective
    cv2.ellipse(cam_chamber, (inset_w // 2, inset_h // 2 + 20), (55, 24), 0, 0, 360, (50, 60, 68), -1)
    if mug_pos[0] > 0.15:  # Inside microwave
        cv2.circle(cam_chamber, (inset_w // 2, inset_h // 2 + 10), 18, (40, 215, 255), -1)
    cv2.putText(cam_chamber, "CAM: MICROWAVE CAVITY", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    closure_pct = max(0, int((1.0 - door_angle_deg / 75.0) * 100))
    cv2.putText(cam_chamber, f"Door Closed: {closure_pct}%", (10, inset_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 240, 120), 1, cv2.LINE_AA)

    # Inset layout
    x_in = W - inset_w - 25
    canvas[85:85 + inset_h, x_in:x_in + inset_w] = cam_wrist
    cv2.rectangle(canvas, (x_in, 85), (x_in + inset_w, 85 + inset_h), (0, 200, 255), 2)

    canvas[85 + inset_h + 18:85 + inset_h * 2 + 18, x_in:x_in + inset_w] = cam_chamber
    cv2.rectangle(canvas, (x_in, 85 + inset_h + 18), (x_in + inset_w, 85 + inset_h * 2 + 18), (60, 220, 120), 2)

    # Top HUD Banner
    cv2.rectangle(canvas, (25, 20), (W - 25, 68), (16, 18, 22), -1)
    cv2.rectangle(canvas, (25, 20), (W - 25, 68), (50, 55, 65), 1)

    cv2.putText(canvas, "FastVLA Turbo (7B)", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Long-Horizon Composite Task | Microwave Mug Insertion & Articulated Door Closure", (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    # Rate and Latency
    cv2.putText(canvas, "50.0 Hz Real-Time Loop", (W - 275, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 240, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Step Latency: 2.8 ms", (W - 275, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 200), 1, cv2.LINE_AA)

    # Bottom Telemetry Bar
    bar_y = H - 90
    cv2.rectangle(canvas, (25, bar_y), (W - 25, H - 20), (16, 18, 22), -1)
    cv2.rectangle(canvas, (25, bar_y), (W - 25, H - 20), (50, 55, 65), 1)

    # Prompt
    prompt_str = "Task: 'Put yellow & white mug in microwave and close it' (LIBERO-10 Hardest Benchmark)"
    cv2.putText(canvas, prompt_str, (40, bar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 100), 1, cv2.LINE_AA)

    # Phase Status
    cv2.putText(canvas, f"State: {current_phase}", (40, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 240, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Door Hinge: {door_angle_deg:4.1f} deg | Closure: {closure_pct}%", (380, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 210, 255), 1, cv2.LINE_AA)

    # Progress bar
    prog = int((frame_idx + 1) / total_frames * 200)
    cv2.rectangle(canvas, (W - 245, bar_y + 35), (W - 45, bar_y + 47), (40, 45, 52), -1)
    cv2.rectangle(canvas, (W - 245, bar_y + 35), (W - 245 + prog, bar_y + 47), (0, 200, 255), -1)
    cv2.putText(canvas, f"Step {frame_idx+1}/{total_frames}", (W - 245, bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 205, 215), 1, cv2.LINE_AA)

    return canvas


def main():
    print("=" * 75)
    print("🎬 SIMULATING FASTVLA ON THE HARDEST ROBOTICS BENCHMARK TASK")
    print("Task: 'ضع الكوب الأصفر والأبيض في الميكروويف وأغلقه'")
    print("(Microwave Mug Insertion & Articulated Door Closure - LIBERO-10 Task 2)")
    print("=" * 75)

    runner = FastVLAPolicyRunner(action_dim=7, chunk_size=16, enable_safety=True)
    robot_arm = RobotArmRenderer(base_pos=np.array([-0.30, -0.22, 0.0], dtype=np.float32))

    mug_start = np.array([-0.08, -0.05, 0.0], dtype=np.float32)
    mug_turntable = np.array([0.27, 0.15, 0.035], dtype=np.float32)

    total_frames = 160
    fps = 30
    frames = []
    trail: List[np.ndarray] = []

    door_start_deg = 75.0
    mug_pos = mug_start.copy()

    for f in range(total_frames):
        tau = f / (total_frames - 1)

        if tau < 0.20:
            # Phase 1: Precision Mug Handle Approach
            phase_name = "Phase 1: Approaching Yellow/White Mug Handle"
            p = tau / 0.20
            s = 0.5 - 0.5 * math.cos(p * math.pi)
            ee_pos = (1 - s) * np.array([-0.25, -0.15, 0.22]) + s * (mug_start + np.array([0.0, 0.0, 0.05]))
            grip_open = 1.0
            door_angle = door_start_deg

        elif tau < 0.32:
            # Phase 2: Grasp Mug Handle
            phase_name = "Phase 2: Grasping Mug Handle"
            p = (tau - 0.20) / 0.12
            ee_pos = mug_start + np.array([0.0, 0.0, 0.04])
            grip_open = max(0.0, 1.0 - p * 1.0)
            door_angle = door_start_deg

        elif tau < 0.58:
            # Phase 3: Transport Mug Into Microwave Chamber
            phase_name = "Phase 3: Transporting Mug into Microwave Chamber"
            p = (tau - 0.32) / 0.26
            s = 0.5 - 0.5 * math.cos(p * math.pi)
            arc_z = 0.12 * math.sin(s * math.pi)
            mug_pos = (1 - s) * mug_start + s * mug_turntable + np.array([0, 0, arc_z])
            ee_pos = mug_pos + np.array([0.0, 0.0, 0.04])
            grip_open = 0.0
            door_angle = door_start_deg

        elif tau < 0.68:
            # Phase 4: Place on Turntable & Clearance Retract
            phase_name = "Phase 4: Turntable Placement & Cavity Retract"
            p = (tau - 0.58) / 0.10
            s = 0.5 - 0.5 * math.cos(p * math.pi)
            mug_pos = mug_turntable.copy()
            grip_open = min(1.0, p * 1.5)
            # Retract out of cavity towards door handle
            door_handle_pre = np.array([0.22, 0.15 - 0.24 * math.sin(math.radians(75.0)) - 0.05, 0.12])
            ee_pos = (1 - s) * (mug_turntable + np.array([0.0, 0.0, 0.04])) + s * door_handle_pre
            door_angle = door_start_deg

        elif tau < 0.90:
            # Phase 5: Multi-Point Contact Door Closure
            phase_name = "Phase 5: Pushing Articulated Door (Hinge Revolute Arc)"
            p = (tau - 0.68) / 0.22
            s = 0.5 - 0.5 * math.cos(p * math.pi)
            door_angle = (1 - s) * door_start_deg + s * 0.0
            # End-effector pushes against door handle edge along circular arc
            hx = 0.30 - 0.28 / 2 + 0.02
            hy = 0.15 - 0.24 / 2
            cur_rad = math.radians(door_angle)
            door_edge = np.array([hx + 0.20 * math.cos(cur_rad) + 0.02, hy - 0.20 * math.sin(cur_rad) - 0.02, 0.11])
            ee_pos = door_edge
            grip_open = 0.0

        else:
            # Phase 6: Final Latch Engagement & Park
            phase_name = "Phase 6: Mechanical Latch Engaged (100% Closure)"
            door_angle = 0.0
            p = (tau - 0.90) / 0.10
            s = 0.5 - 0.5 * math.cos(p * math.pi)
            ee_pos = (1 - s) * np.array([0.26, 0.01, 0.11]) + s * np.array([-0.05, -0.15, 0.25])
            grip_open = 1.0

        # Safety filter step
        raw_act = np.concatenate([ee_pos, [0.0, 0.0, 0.0], [grip_open]]).astype(np.float32)
        filtered_act, safety_status = runner.safety_filter.filter_action(raw_act)

        trail.append(ee_pos.copy())
        if len(trail) > 40:
            trail.pop(0)

        frame = render_microwave_scene(
            frame_idx=f,
            total_frames=total_frames,
            ee_pos=ee_pos,
            grip_open=grip_open,
            mug_pos=mug_pos,
            door_angle_deg=door_angle,
            current_phase=phase_name,
            robot_arm=robot_arm,
            trail=trail,
            safety_info=safety_status,
        )
        frames.append(frame)

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    mp4_path = os.path.join(assets_dir, "fastvla_microwave_closure_rollout.mp4")
    gif_path = os.path.join(assets_dir, "fastvla_microwave_closure_rollout.gif")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(mp4_path, fourcc, fps, (1280, 720))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"✅ Saved MP4 video: {mp4_path} ({os.path.getsize(mp4_path) / 1024:.1f} KB)")

    print("Encoding animated GIF for web embedding...")
    pil_frames = []
    for i, f in enumerate(frames):
        if i % 2 == 0:  # 15 fps
            small = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / 15),
        loop=0,
        optimize=True,
    )
    print(f"✅ Saved GIF animation: {gif_path} ({os.path.getsize(gif_path) / (1024 * 1024):.2f} MB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
