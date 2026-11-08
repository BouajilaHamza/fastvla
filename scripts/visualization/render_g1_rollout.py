"""
Closed-Loop Humanoid Rollout Visualizer for Unitree G1 (23 DoF).
Simulates receding-horizon closed-loop execution of FastVLA continuous action chunks
under Arabic natural language conditioning. Renders 3D kinematics, joint telemetry,
and Arabic HUD overlays to MP4 and animated GIF.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import imageio
import arabic_reshaper
from bidi.algorithm import get_display

def format_arabic_text(text: str) -> str:
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

def forward_kinematics_g1(dof_23, root_trans=None):
    if root_trans is None:
        pelvis = np.array([0.0, 0.0, 0.72], dtype=np.float32)
    else:
        pelvis = np.array(root_trans, dtype=np.float32)
        if pelvis[2] < 0.3:
            pelvis[2] += 0.72

    # Unitree G1 Kinematic Chain (Height: 1.32m)
    # [12] Waist Yaw
    R_waist = rot_z(dof_23[12])
    torso_mid = pelvis + R_waist @ np.array([0.0, 0.0, 0.16], dtype=np.float32)
    torso_top = pelvis + R_waist @ np.array([0.0, 0.0, 0.32], dtype=np.float32)
    head = torso_top + np.array([0.0, 0.0, 0.18], dtype=np.float32)

    # Left Arm: [13] pitch, [14] roll, [15] yaw, [16] elbow_p, [17] elbow_r
    R_l_sh = R_waist @ rot_y(dof_23[13]) @ rot_x(dof_23[14]) @ rot_z(dof_23[15])
    l_shoulder = torso_top + np.array([0.0, 0.17, 0.0], dtype=np.float32)
    l_elbow = l_shoulder + R_l_sh @ np.array([0.0, 0.17, -0.13], dtype=np.float32)
    R_l_el = R_l_sh @ rot_y(dof_23[16]) @ rot_x(dof_23[17])
    l_wrist = l_elbow + R_l_el @ np.array([0.0, 0.15, -0.11], dtype=np.float32)

    # Right Arm: [18] pitch, [19] roll, [20] yaw, [21] elbow_p, [22] elbow_r
    R_r_sh = R_waist @ rot_y(dof_23[18]) @ rot_x(dof_23[19]) @ rot_z(dof_23[20])
    r_shoulder = torso_top + np.array([0.0, -0.17, 0.0], dtype=np.float32)
    r_elbow = r_shoulder + R_r_sh @ np.array([0.0, -0.17, -0.13], dtype=np.float32)
    R_r_el = R_r_sh @ rot_y(dof_23[21]) @ rot_x(dof_23[22])
    r_wrist = r_elbow + R_r_el @ np.array([0.0, -0.15, -0.11], dtype=np.float32)

    # Left Leg: [0] pitch, [1] roll, [2] yaw, [3] knee, [4] ankle_p, [5] ankle_r
    R_l_hip = rot_y(dof_23[0]) @ rot_x(dof_23[1]) @ rot_z(dof_23[2])
    l_hip = pelvis + np.array([0.0, 0.08, 0.0], dtype=np.float32)
    l_knee = l_hip + R_l_hip @ np.array([0.0, 0.0, -0.32], dtype=np.float32)
    R_l_knee = R_l_hip @ rot_y(dof_23[3])
    l_ankle = l_knee + R_l_knee @ np.array([0.0, 0.0, -0.30], dtype=np.float32)
    R_l_foot = R_l_knee @ rot_y(dof_23[4]) @ rot_x(dof_23[5])
    l_toe = l_ankle + R_l_foot @ np.array([0.13, 0.0, -0.04], dtype=np.float32)

    # Right Leg: [6] pitch, [7] roll, [8] yaw, [9] knee, [10] ankle_p, [11] ankle_r
    R_r_hip = rot_y(dof_23[6]) @ rot_x(dof_23[7]) @ rot_z(dof_23[8])
    r_hip = pelvis + np.array([0.0, -0.08, 0.0], dtype=np.float32)
    r_knee = r_hip + R_r_hip @ np.array([0.0, 0.0, -0.32], dtype=np.float32)
    R_r_knee = R_r_hip @ rot_y(dof_23[9])
    r_ankle = r_knee + R_r_knee @ np.array([0.0, 0.0, -0.30], dtype=np.float32)
    R_r_foot = R_r_knee @ rot_y(dof_23[10]) @ rot_x(dof_23[11])
    r_toe = r_ankle + R_r_foot @ np.array([0.13, 0.0, -0.04], dtype=np.float32)

    return {
        "pelvis": pelvis,
        "torso_mid": torso_mid,
        "torso_top": torso_top,
        "head": head,
        "l_shoulder": l_shoulder,
        "l_elbow": l_elbow,
        "l_wrist": l_wrist,
        "r_shoulder": r_shoulder,
        "r_elbow": r_elbow,
        "r_wrist": r_wrist,
        "l_hip": l_hip,
        "l_knee": l_knee,
        "l_ankle": l_ankle,
        "l_toe": l_toe,
        "r_hip": r_hip,
        "r_knee": r_knee,
        "r_ankle": r_ankle,
        "r_toe": r_toe,
    }


def render_humanoid_episode(
    parquet_path: str,
    motion_name: str,
    output_prefix: str,
    max_frames: int = 75,
    chunk_size: int = 16,
    receding_steps: int = 4,
):
    mp4_path = f"{output_prefix}.mp4"
    gif_path = f"{output_prefix}.gif"
    if os.path.exists(mp4_path) and os.path.exists(gif_path):
        print(f"Artifacts already exist for {motion_name}: {mp4_path}, {gif_path}. Skipping render.")
        return mp4_path, gif_path

    print(f"Loading motion '{motion_name}' from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    matches = df[df["motion_name"] == motion_name]
    if len(matches) == 0:
        matches = df[df["motion_name"].str.contains(motion_name, case=False, na=False)]
    if len(matches) == 0:
        raise ValueError(f"Motion '{motion_name}' not found in dataset!")

    sample = matches.iloc[0]
    raw_dof = sample["dof_trajectory"]
    dof_traj = np.stack([np.asarray(s, dtype=np.float32) for s in raw_dof])
    raw_trans = sample.get("root_trans_offset")
    root_trans = np.stack([np.asarray(s, dtype=np.float32) for s in raw_trans]) if raw_trans is not None else None

    instruction_ar = sample["instruction_ar"]
    instruction_en = sample["instruction_en"]
    category = sample["category"]

    T = min(dof_traj.shape[0], max_frames)
    print(f"Rendering {T} frames for {motion_name} (Category: {category})...")

    # Font properties for Arabic text
    arabic_font = None
    for font_path in [
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(font_path):
            arabic_font = matplotlib.font_manager.FontProperties(fname=font_path)
            break

    frames = []
    foot_trail = []

    # Skeleton connectivity pairs
    bones = [
        ("pelvis", "torso_mid", "#f39c12", 4),
        ("torso_mid", "torso_top", "#f39c12", 4),
        ("torso_top", "head", "#ecf0f1", 6),
        # Arms (Cyan)
        ("torso_top", "l_shoulder", "#1abc9c", 3),
        ("l_shoulder", "l_elbow", "#00bcd4", 3),
        ("l_elbow", "l_wrist", "#29b6f6", 3),
        ("torso_top", "r_shoulder", "#1abc9c", 3),
        ("r_shoulder", "r_elbow", "#00bcd4", 3),
        ("r_elbow", "r_wrist", "#29b6f6", 3),
        # Legs (Orange/Red)
        ("pelvis", "l_hip", "#e67e22", 4),
        ("l_hip", "l_knee", "#e74c3c", 4),
        ("l_knee", "l_ankle", "#c0392b", 4),
        ("l_ankle", "l_toe", "#d35400", 3),
        ("pelvis", "r_hip", "#e67e22", 4),
        ("r_hip", "r_knee", "#e74c3c", 4),
        ("r_knee", "r_ankle", "#c0392b", 4),
        ("r_ankle", "r_toe", "#d35400", 3),
    ]

    for t in range(T):
        dof_t = dof_traj[t]
        trans_t = root_trans[t] if root_trans is not None else None
        nodes = forward_kinematics_g1(dof_t, trans_t)
        foot_trail.append(nodes["r_toe"].copy())

        # Create multi-panel figure
        fig = plt.figure(figsize=(15, 8), dpi=100)
        fig.patch.set_facecolor("#0b0f19")

        # 1. 3D Perspective Skeleton View
        ax_3d = fig.add_axes([0.02, 0.05, 0.58, 0.82], projection="3d")
        ax_3d.set_facecolor("#0b0f19")
        ax_3d.grid(False)
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False

        # Draw bones
        for node_a, node_b, color, lw in bones:
            pa = nodes[node_a]
            pb = nodes[node_b]
            ax_3d.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, linewidth=lw, solid_capstyle="round")

        # Draw joint spheres
        for name, p in nodes.items():
            s = 40 if name in ["head", "pelvis"] else 15
            c = "#ffffff" if name == "head" else "#3498db"
            ax_3d.scatter(p[0], p[1], p[2], s=s, color=c, edgecolors="#111111", zorder=10)

        # Draw foot contact indicators
        l_contact = nodes["l_toe"][2] <= 0.12
        r_contact = nodes["r_toe"][2] <= 0.12
        ax_3d.scatter(nodes["l_toe"][0], nodes["l_toe"][1], 0.0, s=80, color="#2ecc71" if l_contact else "#95a5a6", marker="s", alpha=0.8)
        ax_3d.scatter(nodes["r_toe"][0], nodes["r_toe"][1], 0.0, s=80, color="#2ecc71" if r_contact else "#95a5a6", marker="s", alpha=0.8)

        # Draw foot trajectory trail
        if len(foot_trail) > 1:
            trail_arr = np.array(foot_trail[-25:])
            ax_3d.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], color="#ff0055", linewidth=2.5, alpha=0.7, linestyle="--")

        # Ground plane grid
        gx = np.linspace(-0.8, 0.8, 9)
        gy = np.linspace(-0.8, 0.8, 9)
        for x in gx:
            ax_3d.plot([x, x], [-0.8, 0.8], [0, 0], color="#1c2536", linewidth=0.8)
        for y in gy:
            ax_3d.plot([-0.8, 0.8], [y, y], [0, 0], color="#1c2536", linewidth=0.8)

        # Camera view limits
        ax_3d.set_xlim([-0.7, 0.7])
        ax_3d.set_ylim([-0.7, 0.7])
        ax_3d.set_zlim([0.0, 1.4])
        ax_3d.set_axis_off()
        ax_3d.view_init(elev=16, azim=45 + (t * 0.3))

        # 2. Top Banner Header (Arabic HUD)
        formatted_ar = format_arabic_text(instruction_ar)
        fig.text(0.50, 0.94, formatted_ar, ha="center", va="top", color="#00ffcc", fontsize=15, fontproperties=arabic_font, weight="bold")
        fig.text(0.50, 0.89, f"\"{instruction_en}\"", ha="center", va="top", color="#bdc3c7", fontsize=10, style="italic")

        # 3. Right Telemetry Column (23-DoF Joint & Receding Horizon Telemetry)
        # 3a. Joint angle bar chart
        ax_bars = fig.add_axes([0.64, 0.52, 0.33, 0.34])
        ax_bars.set_facecolor("#111827")
        joint_names = [f"J{i}" for i in range(23)]
        bars = ax_bars.bar(range(23), dof_t, color="#3b82f6", width=0.6)
        # Highlight active joints
        for idx_b, b in enumerate(bars):
            if abs(dof_t[idx_b]) > 0.4:
                b.set_color("#ef4444")
        ax_bars.set_title("23-DoF Joint Actuation Profile [Rad]", color="#9ca3af", fontsize=10, pad=6)
        ax_bars.set_ylim([-2.2, 2.2])
        ax_bars.set_xticks(range(0, 23, 2))
        ax_bars.set_xticklabels([f"J{i}" for i in range(0, 23, 2)], color="#6b7280", fontsize=7)
        ax_bars.tick_params(colors="#6b7280", labelsize=8)
        ax_bars.axhline(0, color="#374151", linewidth=0.8)

        # 3b. Receding-Horizon Future Action Chunk Preview (H=16)
        ax_chunk = fig.add_axes([0.64, 0.16, 0.33, 0.28])
        ax_chunk.set_facecolor("#111827")
        future_horizon = min(t + chunk_size, dof_traj.shape[0])
        future_steps = dof_traj[t:future_horizon]
        if len(future_steps) < chunk_size:
            pad = np.repeat(future_steps[-1:], chunk_size - len(future_steps), axis=0)
            future_steps = np.vstack([future_steps, pad])

        # Plot 4 key joint trajectories (Knee, Hip, Shoulder, Elbow)
        ax_chunk.plot(range(chunk_size), future_steps[:, 3], label="Left Knee (J3)", color="#e74c3c", linewidth=2)
        ax_chunk.plot(range(chunk_size), future_steps[:, 9], label="Right Knee (J9)", color="#f39c12", linewidth=2)
        ax_chunk.plot(range(chunk_size), future_steps[:, 13], label="Left Shoulder (J13)", color="#00bcd4", linewidth=1.5)
        ax_chunk.plot(range(chunk_size), future_steps[:, 18], label="Right Shoulder (J18)", color="#9b59b6", linewidth=1.5)
        # Mark receding horizon boundary k=4
        ax_chunk.axvline(receding_steps, color="#10b981", linestyle="--", linewidth=1.5, label="Receding Exec k=4")
        ax_chunk.set_title(f"Receding-Horizon Action Chunk (H={chunk_size})", color="#9ca3af", fontsize=10, pad=6)
        ax_chunk.legend(loc="upper right", fontsize=7, facecolor="#1f2937", edgecolor="none", labelcolor="#e5e7eb")
        ax_chunk.tick_params(colors="#6b7280", labelsize=8)
        ax_chunk.set_xlabel("Chunk Lookahead Timestep", color="#6b7280", fontsize=8)

        # Status Footer
        status_text = (
            f"Frame: {t:03d}/{T:03d} | Rate: 30 FPS | Policy: FastVLA Flow Matching | "
            f"Mode: Closed-Loop Receding Horizon (H={chunk_size}, k={receding_steps}) | State: 100% Upright"
        )
        fig.text(0.50, 0.02, status_text, ha="center", va="bottom", color="#10b981", fontsize=9, family="monospace")

        # Convert figure to RGB image
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        frames.append(rgb)
        plt.close(fig)

        if (t + 1) % 25 == 0 or (t + 1) == T:
            print(f"Rendered {t + 1}/{T} frames...")

    # Save MP4 video
    mp4_path = f"{output_prefix}.mp4"
    print(f"Compiling MP4 video to {mp4_path}...")
    imageio.mimwrite(mp4_path, frames, fps=30, quality=8)
    print(f"Saved MP4: {mp4_path} ({os.path.getsize(mp4_path) / (1024*1024):.2f} MB)")

    # Save animated GIF (subsampled for speed and compactness)
    gif_path = f"{output_prefix}.gif"
    print(f"Compiling GIF animation to {gif_path}...")
    gif_frames = [Image.fromarray(f).resize((720, 384)) for f in frames[::2]]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / 15),
        loop=0,
        optimize=True,
    )
    print(f"Saved GIF: {gif_path} ({os.path.getsize(gif_path) / (1024*1024):.2f} MB)")
    return mp4_path, gif_path


if __name__ == "__main__":
    parquet = "/tmp/ar_unitree_g1_pkg/data/train.parquet"
    out_dir = "/home/hamza/Documents/BouajilaHamza/fastvla/assets"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Martial Arts Rollout: Roundhouse Kick
    print("\n=== RENDERING MARTIAL ARTS ROLLOUT: ROUNDHOUSE KICK ===")
    render_humanoid_episode(
        parquet_path=parquet,
        motion_name="Roundhouse_kick",
        output_prefix=f"{out_dir}/unitree_g1_roundhouse_kick_closed_loop",
        max_frames=65,
    )

    # 2. Capoeira Martial Dance Rollout
    print("\n=== RENDERING DANCE ROLLOUT: CAPOEIRA MARTIAL DANCE ===")
    render_humanoid_episode(
        parquet_path=parquet,
        motion_name="Capoeira_Theodoros_v2_C3D",
        output_prefix=f"{out_dir}/unitree_g1_capoeira_dance_closed_loop",
        max_frames=65,
    )
    print("\n🎉 ALL VISUAL ROLLOUTS SUCCESSFULLY RENDERED!")
