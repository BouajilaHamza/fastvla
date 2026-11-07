"""
Unit tests for FastVLA Closed-Loop Physical Simulation Evaluation Harness.
"""

import pytest
import numpy as np
from PIL import Image

from fastvla.inference import FastVLAPolicyRunner, ActionChunkBuffer
from scripts.evaluation.eval_closed_loop import (
    SyntheticKinematicEnv,
    evaluate_closed_loop_episode,
    run_closed_loop_benchmark,
    make_eval_env,
)


def test_synthetic_kinematic_env_lifecycle():
    """Verify synthetic kinematic env reset, step, and goal dynamics."""
    env = SyntheticKinematicEnv(task_name="test_task", max_steps=100)
    obs, info = env.reset()

    assert "agentview_image" in obs
    assert "robot0_eye_in_hand_image" in obs
    assert "ee_pos" in obs
    assert obs["ee_pos"].shape == (3,)
    assert info["is_success"] is False

    # Take step
    action = np.array([0.1, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert "agentview_image" in next_obs
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "dist_to_goal" in step_info


def test_closed_loop_episode_execution():
    """Test full single-episode closed-loop rollout with FastVLAPolicyRunner."""
    runner = FastVLAPolicyRunner(action_dim=7, chunk_size=16)
    env = SyntheticKinematicEnv(max_steps=50)

    result = evaluate_closed_loop_episode(
        runner=runner,
        env=env,
        instruction="ضع الكوب على الطبق",
        max_steps=30,
        record_frames=True,
    )

    assert "success" in result
    assert "steps" in result
    assert result["steps"] > 0
    assert result["control_hz"] > 0.0
    assert "trajectory_jerk" in result
    assert len(result["frames"]) > 0


def test_closed_loop_benchmark_aggregation():
    """Test benchmark suite aggregation over multiple episodes."""
    runner = FastVLAPolicyRunner(action_dim=7, chunk_size=16)

    benchmark_summary = run_closed_loop_benchmark(
        runner=runner,
        suite="libero_10",
        num_episodes=3,
        max_steps=40,
        instruction="شغّل الموقد",
        record=False,
    )

    assert benchmark_summary["suite"] == "libero_10"
    assert benchmark_summary["num_episodes"] == 3
    assert 0.0 <= benchmark_summary["success_rate_pct"] <= 100.0
    assert benchmark_summary["average_control_hz"] > 0.0
    assert len(benchmark_summary["episodes"]) == 3


def test_make_eval_env_fallback():
    """Verify that make_eval_env returns a valid gym/kinematic env without throwing."""
    env = make_eval_env(suite="libero_spatial", task_id=0, max_steps=50)
    obs, info = env.reset()
    assert obs is not None

    pusht_env = make_eval_env(suite="pusht", task_id=0, max_steps=50)
    assert pusht_env is not None
