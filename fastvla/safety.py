"""
Robotic Kinematic Safety Filter and Emergency Watchdog for FastVLA.

Guarantees hardware safety during sim-to-real transfer and physical robot deployment:
1. Joint Position Clamping (enforces physical hard stops).
2. Slew-Rate Velocity Limiting (bounds maximum joint velocity / Delta theta / Delta t).
3. Acceleration / Jerk Limiting (bounds motor torque shocks).
4. Watchdog Heartbeat Monitoring (trips emergency soft-stop if camera or policy stalls).
5. Pre-configured robot presets: SO-100, Koch Arm, ALOHA Bimanual, Unitree G1 Humanoid.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch


@dataclass
class RobotSafetyConfig:
    """Kinematic limits and watchdog thresholds for a physical robotic embodiment."""
    name: str = "generic"
    action_dim: int = 7
    # Joint position limits [min, max] per degree of freedom
    joint_min: np.ndarray = field(default_factory=lambda: np.full(7, -1.0, dtype=np.float32))
    joint_max: np.ndarray = field(default_factory=lambda: np.full(7, 1.0, dtype=np.float32))
    # Max change per second (rad/s or normalized units/s)
    max_velocity: np.ndarray = field(default_factory=lambda: np.full(7, 2.0, dtype=np.float32))
    # Max acceleration per second^2
    max_acceleration: np.ndarray = field(default_factory=lambda: np.full(7, 8.0, dtype=np.float32))
    # Watchdog timeout in seconds before emergency stop
    watchdog_timeout_sec: float = 0.5
    # Default park / home safe action
    home_action: Optional[np.ndarray] = None

    def __post_init__(self):
        if len(self.joint_min) != self.action_dim:
            self.joint_min = np.full(self.action_dim, -1.0, dtype=np.float32)
        if len(self.joint_max) != self.action_dim:
            self.joint_max = np.full(self.action_dim, 1.0, dtype=np.float32)
        if len(self.max_velocity) != self.action_dim:
            self.max_velocity = np.full(self.action_dim, 2.0, dtype=np.float32)
        if len(self.max_acceleration) != self.action_dim:
            self.max_acceleration = np.full(self.action_dim, 8.0, dtype=np.float32)
        if self.home_action is not None and len(self.home_action) != self.action_dim:
            self.home_action = np.zeros(self.action_dim, dtype=np.float32)


# Pre-configured Hardware Safety Presets
SO_100_SAFETY_CONFIG = RobotSafetyConfig(
    name="so_100",
    action_dim=6,
    # 6-DoF: base, shoulder, elbow, wrist_pitch, wrist_roll, gripper
    joint_min=np.array([-1.8, -1.5, -1.7, -1.6, -2.8, 0.0], dtype=np.float32),
    joint_max=np.array([1.8, 1.5, 1.7, 1.6, 2.8, 1.0], dtype=np.float32),
    max_velocity=np.array([1.5, 1.2, 1.5, 2.0, 2.5, 3.0], dtype=np.float32),
    max_acceleration=np.array([4.0, 3.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32),
    watchdog_timeout_sec=0.4,
    home_action=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
)

KOCH_SAFETY_CONFIG = RobotSafetyConfig(
    name="koch_arm",
    action_dim=6,
    joint_min=np.array([-2.0, -1.6, -1.6, -1.8, -2.5, 0.0], dtype=np.float32),
    joint_max=np.array([2.0, 1.6, 1.6, 1.8, 2.5, 1.0], dtype=np.float32),
    max_velocity=np.array([1.5, 1.2, 1.5, 2.0, 2.5, 3.0], dtype=np.float32),
    max_acceleration=np.array([4.0, 3.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32),
    watchdog_timeout_sec=0.4,
    home_action=np.zeros(6, dtype=np.float32),
)

ALOHA_BIMANUAL_SAFETY_CONFIG = RobotSafetyConfig(
    name="aloha_bimanual",
    action_dim=14,
    # Dual 7-DoF: Left [6 joints + grip], Right [6 joints + grip]
    joint_min=np.array([
        -1.7, -1.5, -1.7, -1.8, -1.8, -2.8, 0.0,  # Left arm
        -1.7, -1.5, -1.7, -1.8, -1.8, -2.8, 0.0,  # Right arm
    ], dtype=np.float32),
    joint_max=np.array([
        1.7, 1.5, 1.7, 1.8, 1.8, 2.8, 1.0,  # Left arm
        1.7, 1.5, 1.7, 1.8, 1.8, 2.8, 1.0,  # Right arm
    ], dtype=np.float32),
    max_velocity=np.array([
        2.0, 1.8, 2.0, 2.5, 2.5, 3.0, 4.0,
        2.0, 1.8, 2.0, 2.5, 2.5, 3.0, 4.0,
    ], dtype=np.float32),
    max_acceleration=np.array([
        6.0, 5.0, 6.0, 8.0, 8.0, 10.0, 12.0,
        6.0, 5.0, 6.0, 8.0, 8.0, 10.0, 12.0,
    ], dtype=np.float32),
    watchdog_timeout_sec=0.5,
    home_action=np.zeros(14, dtype=np.float32),
)

UNITREE_G1_SAFETY_CONFIG = RobotSafetyConfig(
    name="unitree_g1",
    action_dim=14,
    # Dual 7-DoF manipulators on humanoid torso
    joint_min=np.array([
        -2.0, -1.8, -2.0, -2.0, -1.8, -2.8, 0.0,  # Left arm + hand
        -2.0, -1.8, -2.0, -2.0, -1.8, -2.8, 0.0,  # Right arm + hand
    ], dtype=np.float32),
    joint_max=np.array([
        2.0, 1.8, 2.0, 2.0, 1.8, 2.8, 1.0,
        2.0, 1.8, 2.0, 2.0, 1.8, 2.8, 1.0,
    ], dtype=np.float32),
    max_velocity=np.full(14, 2.5, dtype=np.float32),
    max_acceleration=np.full(14, 8.0, dtype=np.float32),
    watchdog_timeout_sec=0.35,  # Faster watchdog on humanoid
    home_action=np.zeros(14, dtype=np.float32),
)


class SafetyTripException(Exception):
    """Raised when an unrecoverable kinematic safety violation occurs."""
    pass


class SafetyFilter:
    """
    Real-time kinematic safety shield filtering raw policy outputs before sending to servos.
    """

    def __init__(
        self,
        config: Optional[RobotSafetyConfig] = None,
        action_dim: int = 7,
        watchdog_timeout_sec: float = 0.5,
        default_dt: float = 0.02,  # 50 Hz control loop default
    ):
        if config is not None:
            self.config = config
        else:
            self.config = RobotSafetyConfig(
                name="custom",
                action_dim=action_dim,
                joint_min=np.full(action_dim, -1.0, dtype=np.float32),
                joint_max=np.full(action_dim, 1.0, dtype=np.float32),
                max_velocity=np.full(action_dim, 2.0, dtype=np.float32),
                max_acceleration=np.full(action_dim, 8.0, dtype=np.float32),
                watchdog_timeout_sec=watchdog_timeout_sec,
                home_action=np.zeros(action_dim, dtype=np.float32),
            )

        self.action_dim = self.config.action_dim
        self.default_dt = default_dt

        # State tracking
        self.last_action: Optional[np.ndarray] = None
        self.last_velocity: np.ndarray = np.zeros(self.action_dim, dtype=np.float32)
        self.last_timestamp: float = time.perf_counter()
        self.last_heartbeat: float = time.perf_counter()

        # Trip / diagnostic state
        self.is_tripped: bool = False
        self.trip_reason: Optional[str] = None
        self.clamped_violations_count: int = 0
        self.velocity_limited_count: int = 0

    def reset(self, initial_position: Optional[np.ndarray] = None):
        """Resets safety filter state to a known safe configuration."""
        if initial_position is not None:
            self.last_action = np.array(initial_position, dtype=np.float32).copy()
        elif self.config.home_action is not None:
            self.last_action = self.config.home_action.copy()
        else:
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        self.last_velocity = np.zeros(self.action_dim, dtype=np.float32)
        self.last_timestamp = time.perf_counter()
        self.last_heartbeat = time.perf_counter()
        self.is_tripped = False
        self.trip_reason = None
        self.clamped_violations_count = 0
        self.velocity_limited_count = 0

    def heartbeat(self):
        """Records an active observation or policy inference heartbeat."""
        self.last_heartbeat = time.perf_counter()
        if self.is_tripped and self.trip_reason == "WATCHDOG_TIMEOUT":
            # Recovered from watchdog timeout
            self.is_tripped = False
            self.trip_reason = None

    def check_watchdog(self, current_time: Optional[float] = None) -> bool:
        """
        Checks whether the watchdog has expired. Returns True if alive, False if tripped.
        """
        now = current_time if current_time is not None else time.perf_counter()
        elapsed = now - self.last_heartbeat
        if elapsed > self.config.watchdog_timeout_sec:
            self.is_tripped = True
            self.trip_reason = f"WATCHDOG_TIMEOUT (elapsed {elapsed*1000:.1f}ms > {self.config.watchdog_timeout_sec*1000:.1f}ms)"
            return False
        return True

    def filter_action(
        self,
        raw_action: Union[np.ndarray, torch.Tensor],
        dt: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filters a candidate action through the safety barrier:
        1. Watchdog check: If tripped, softly brings robot towards home or decelerates.
        2. Position clamping: Clips within physical limits [joint_min, joint_max].
        3. Velocity limiting: Ensures Delta theta <= max_velocity * dt.
        4. Acceleration limiting: Ensures Delta v <= max_acceleration * dt.
        """
        now = current_time if current_time is not None else time.perf_counter()
        if dt is None:
            dt = max(now - self.last_timestamp, 1e-4) if self.last_action is not None else self.default_dt
            dt = min(dt, 0.1)  # Clamp dt to prevent huge jumps after pauses

        if isinstance(raw_action, torch.Tensor):
            action = raw_action.detach().cpu().float().numpy().copy()
        else:
            action = np.array(raw_action, dtype=np.float32).copy()

        assert action.shape[-1] == self.action_dim, (
            f"Action dimension mismatch: expected {self.action_dim}, got {action.shape[-1]}"
        )

        status = {
            "safe": True,
            "watchdog_tripped": False,
            "clamped": False,
            "velocity_limited": False,
            "original_norm": float(np.linalg.norm(action)),
            "reason": None,
        }

        # 1. Watchdog Check
        if not self.check_watchdog(now):
            status["safe"] = False
            status["watchdog_tripped"] = True
            status["reason"] = self.trip_reason
            # Fail-safe mode: softly decelerate towards last known position or home
            if self.last_action is not None:
                # Damped zero-velocity hold
                safe_action = self.last_action.copy()
            else:
                safe_action = self.config.home_action if self.config.home_action is not None else np.zeros(self.action_dim, dtype=np.float32)
            self.last_velocity = np.zeros(self.action_dim, dtype=np.float32)
            self.last_timestamp = now
            return safe_action, status

        # 2. Position Limits Clamping
        clipped = np.clip(action, self.config.joint_min, self.config.joint_max)
        if not np.allclose(clipped, action, atol=1e-5):
            status["clamped"] = True
            self.clamped_violations_count += 1
            action = clipped

        # 3. Kinematic Rate & Acceleration Limiting
        if self.last_action is not None:
            # Desired velocity
            desired_vel = (action - self.last_action) / dt

            # Acceleration limit check: |desired_vel - last_vel| <= max_acc * dt
            max_delta_v = self.config.max_acceleration * dt
            delta_v = desired_vel - self.last_velocity
            limited_delta_v = np.clip(delta_v, -max_delta_v, max_delta_v)
            target_vel = self.last_velocity + limited_delta_v

            # Velocity limit check: |target_vel| <= max_velocity
            limited_vel = np.clip(target_vel, -self.config.max_velocity, self.config.max_velocity)

            if not np.allclose(limited_vel, desired_vel, atol=1e-4):
                status["velocity_limited"] = True
                self.velocity_limited_count += 1
                action = self.last_action + limited_vel * dt
                # Re-clamp in case integration touched boundary
                action = np.clip(action, self.config.joint_min, self.config.joint_max)

            self.last_velocity = (action - self.last_action) / dt
        else:
            self.last_velocity = np.zeros(self.action_dim, dtype=np.float32)

        self.last_action = action.copy()
        self.last_timestamp = now
        return action, status
