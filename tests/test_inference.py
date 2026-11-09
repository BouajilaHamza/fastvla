"""
Unit tests for high-frequency action chunking and edge inference runner.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from fastvla.inference import ActionChunkBuffer, EdgeTelemetryHUD, FastVLAPolicyRunner


def test_action_chunk_buffer_basic():
    buffer = ActionChunkBuffer(chunk_size=4, action_dim=2, blend_mode="receding_horizon")
    assert buffer.is_empty()
    assert buffer.remaining_steps() == 0

    chunk = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ], dtype=np.float32)

    buffer.add_chunk(chunk)
    assert buffer.remaining_steps() == 4
    assert not buffer.is_empty()

    action0 = buffer.pop_action()
    np.testing.assert_allclose(action0, [0.1, 0.2])
    assert buffer.remaining_steps() == 3

    action1 = buffer.pop_action()
    np.testing.assert_allclose(action1, [0.3, 0.4])

    buffer.clear()
    assert buffer.is_empty()
    assert buffer.pop_action() is None


def test_action_chunk_buffer_linear_blend():
    buffer = ActionChunkBuffer(chunk_size=4, action_dim=2, blend_mode="linear_blend", blend_alpha=0.5)
    
    chunk1 = np.ones((4, 2), dtype=np.float32)
    buffer.add_chunk(chunk1)
    _ = buffer.pop_action()
    _ = buffer.pop_action()
    # 2 items left in buffer: both are [1.0, 1.0]

    chunk2 = np.zeros((4, 2), dtype=np.float32)
    buffer.add_chunk(chunk2)
    assert buffer.remaining_steps() == 4

    # The first 2 elements should be a blend between 1.0 and 0.0
    action = buffer.pop_action()
    assert 0.0 < action[0] < 1.0


def test_edge_telemetry_hud():
    telemetry = EdgeTelemetryHUD(window_size=10)
    for i in range(10):
        telemetry.record_step(step_latency_ms=5.0, action=np.array([0.1 * i, -0.1 * i]))

    metrics = telemetry.get_metrics()
    assert metrics["total_steps"] == 10
    assert metrics["mean_step_latency_ms"] == 5.0
    assert 0.0 <= metrics["saturation_rate"] <= 1.0

    hud_str = telemetry.render_hud()
    assert "[FastVLA HUD]" in hud_str
    assert "Rate:" in hud_str


def test_policy_runner_step_mock():
    runner = FastVLAPolicyRunner(
        device="cpu",
        chunk_size=8,
        action_dim=7,
        ode_solver="midpoint",
        mock_mode=True,
    )

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    # Step multiple times
    actions = []
    for _ in range(12):
        action = runner.step(img, instruction="test instruction", replan_threshold=3)
        actions.append(action)
        assert action.shape == (7,)

    assert len(actions) == 12
    metrics = runner.telemetry.get_metrics()
    assert metrics["total_steps"] == 12
    assert metrics["mean_chunk_inference_ms"] > 0
