"""
Unit and integration tests for Robotic Kinematic Safety Filter and Watchdog.
Verifies physical hard-stop clamping, slew-rate velocity limiting, and watchdog failsafe.
"""

import time
import numpy as np
import pytest
import torch

from fastvla.safety import (
    SafetyFilter,
    RobotSafetyConfig,
    SO_100_SAFETY_CONFIG,
    ALOHA_BIMANUAL_SAFETY_CONFIG,
    UNITREE_G1_SAFETY_CONFIG,
)


def test_safety_filter_clamping():
    """Verify actions exceeding physical joint limits are safely clamped."""
    cfg = RobotSafetyConfig(
        name="test_clamp",
        action_dim=3,
        joint_min=np.array([-1.0, -0.5, 0.0], dtype=np.float32),
        joint_max=np.array([1.0, 0.5, 1.0], dtype=np.float32),
    )
    sf = SafetyFilter(config=cfg)

    # Out of bounds action
    raw_action = np.array([2.5, -1.0, -0.5], dtype=np.float32)
    safe_action, status = sf.filter_action(raw_action)

    assert status["clamped"] is True
    assert np.allclose(safe_action, [1.0, -0.5, 0.0], atol=1e-5)
    assert sf.clamped_violations_count == 1


def test_safety_filter_velocity_limiting():
    """Verify abrupt step discontinuities (high jerk) are rate-limited by max velocity."""
    cfg = RobotSafetyConfig(
        name="test_vel",
        action_dim=2,
        joint_min=np.array([-5.0, -5.0], dtype=np.float32),
        joint_max=np.array([5.0, 5.0], dtype=np.float32),
        max_velocity=np.array([1.0, 1.0], dtype=np.float32),  # 1.0 unit / sec max
        max_acceleration=np.array([10.0, 10.0], dtype=np.float32),
    )
    sf = SafetyFilter(config=cfg, default_dt=0.1)
    sf.reset(initial_position=np.array([0.0, 0.0], dtype=np.float32))

    # Request jump of +2.0 in 0.1s -> requested vel = 20.0 (exceeds max_vel 1.0)
    desired = np.array([2.0, 2.0], dtype=np.float32)
    filtered, status = sf.filter_action(desired, dt=0.1)

    assert status["velocity_limited"] is True
    # At dt=0.1 with max_acc=10.0 and max_vel=1.0, allowable delta is <= 1.0 * 0.1 = 0.1
    assert filtered[0] <= 0.15
    assert filtered[1] <= 0.15


def test_watchdog_timeout_tripping():
    """Verify watchdog trips emergency fail-safe if heartbeat is missed."""
    cfg = RobotSafetyConfig(
        name="test_watchdog",
        action_dim=4,
        watchdog_timeout_sec=0.1,  # 100ms timeout
        home_action=np.zeros(4, dtype=np.float32),
    )
    sf = SafetyFilter(config=cfg)
    sf.reset()

    # Step 1: Normal heartbeat
    sf.heartbeat()
    act, status = sf.filter_action(np.ones(4, dtype=np.float32))
    assert status["safe"] is True
    assert status["watchdog_tripped"] is False

    # Simulate sensor stall (> 120ms)
    time.sleep(0.12)

    # Step 2: Policy tries to command without heartbeat
    stall_act, stall_status = sf.filter_action(np.ones(4, dtype=np.float32))
    assert stall_status["safe"] is False
    assert stall_status["watchdog_tripped"] is True
    assert "WATCHDOG_TIMEOUT" in stall_status["reason"]
    # Fail-safe holds last position or home
    assert np.allclose(stall_act, act, atol=1e-5)


def test_hardware_presets():
    """Verify hardware preset configurations load valid kinematic limits."""
    so100 = SafetyFilter(config=SO_100_SAFETY_CONFIG)
    assert so100.action_dim == 6
    assert len(so100.config.joint_min) == 6

    aloha = SafetyFilter(config=ALOHA_BIMANUAL_SAFETY_CONFIG)
    assert aloha.action_dim == 14
    assert len(aloha.config.joint_min) == 14

    g1 = SafetyFilter(config=UNITREE_G1_SAFETY_CONFIG)
    assert g1.action_dim == 14
    assert g1.config.watchdog_timeout_sec == 0.35
