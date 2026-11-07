"""
Unit tests for the Universal Hugging Face LeRobot dataset loader in FastVLA.
Tests metadata parsing, multi-camera discovery, continuous action normalization,
action chunking across episode boundaries, and UnslothVLACollator integration.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from fastvla.data.lerobot import (
    LeRobotDataset,
    LeRobotMetadata,
    load_lerobot_metadata,
)
from fastvla.data.collator import UnslothVLACollator


@pytest.fixture
def mock_lerobot_dir():
    """Create a temporary local LeRobot directory with info.json, stats.json, and tasks.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        info = {
            "codebase_version": "v2.0",
            "fps": 30.0,
            "robot_type": "aloha_bimanual",
            "total_episodes": 2,
            "total_frames": 6,
            "features": {
                "observation.images.cam_high": {
                    "dtype": "image",
                    "shape": [3, 224, 224],
                },
                "observation.images.cam_left_wrist": {
                    "dtype": "image",
                    "shape": [3, 224, 224],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [14],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [14],
                },
            },
        }
        with open(os.path.join(meta_dir, "info.json"), "w") as f:
            json.dump(info, f)

        stats = {
            "action": {
                "min": [-10.0] * 14,
                "max": [10.0] * 14,
                "mean": [0.0] * 14,
                "std": [5.0] * 14,
                "q01": [-8.0] * 14,
                "q99": [8.0] * 14,
            },
            "observation.state": {
                "min": [-5.0] * 14,
                "max": [5.0] * 14,
            },
        }
        with open(os.path.join(meta_dir, "stats.json"), "w") as f:
            json.dump(stats, f)

        tasks = [
            {"task_index": 0, "task": "Pick the white mug and place it on the left plate"},
            {"task_index": 1, "task": "Rotate the stove dial and align the kettle"},
        ]
        with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

        yield tmpdir


def test_load_lerobot_metadata(mock_lerobot_dir):
    """Test loading and parsing LeRobot metadata from local directory."""
    meta = load_lerobot_metadata(mock_lerobot_dir)

    assert meta.codebase_version == "v2.0"
    assert meta.fps == 30.0
    assert meta.robot_type == "aloha_bimanual"
    assert meta.action_dim == 14
    assert meta.state_dim == 14
    assert "observation.images.cam_high" in meta.camera_keys
    assert "observation.images.cam_left_wrist" in meta.camera_keys

    # Check stats parsing
    assert meta.action_min is not None and meta.action_min.shape == (14,)
    assert meta.action_max is not None and meta.action_max.shape == (14,)
    assert meta.action_min[0].item() == -10.0
    assert meta.action_max[0].item() == 10.0

    # Check tasks mapping
    assert 0 in meta.tasks
    assert meta.tasks[0] == "Pick the white mug and place it on the left plate"
    assert 1 in meta.tasks


def test_lerobot_dataset_multi_camera(mock_lerobot_dir):
    """Test multi-camera image extraction and normalization."""
    meta = load_lerobot_metadata(mock_lerobot_dir)

    # Mock HF dataset rows with multi-view cameras
    mock_items = [
        {
            "observation.images.cam_high": np.zeros((100, 100, 3), dtype=np.uint8),
            "observation.images.cam_left_wrist": np.ones((100, 100, 3), dtype=np.uint8) * 255,
            "observation.state": np.zeros(14, dtype=np.float32),
            "action": np.ones(14, dtype=np.float32) * 5.0,  # inside [-10, 10]
            "task_index": 0,
            "episode_index": 0,
        }
    ]

    ds = LeRobotDataset(
        repo_id_or_path=mock_lerobot_dir,
        hf_dataset=mock_items,
        metadata=meta,
        image_size=(224, 224),
        normalize_actions=True,
        norm_mode="min_max",
    )

    sample = ds[0]

    # Verify multi-cam images
    assert "images" in sample
    assert "cam_high" in sample["images"]
    assert "cam_left_wrist" in sample["images"]
    assert "rgb" in sample["images"]  # primary camera alias
    assert sample["images"]["cam_high"].shape == (3, 224, 224)
    assert sample["images"]["cam_left_wrist"].shape == (3, 224, 224)

    # cam_high was zeros -> 0.0
    assert sample["images"]["cam_high"].mean().item() == 0.0
    # cam_left_wrist was 255 -> 1.0
    assert pytest.approx(sample["images"]["cam_left_wrist"].mean().item(), rel=1e-3) == 1.0

    # Verify task instruction resolution from task_index
    assert sample["instructions"] == "Pick the white mug and place it on the left plate"


def test_lerobot_action_normalization():
    """Test min-max, mean-std, and quantile normalization / unnormalization."""
    meta = LeRobotMetadata(
        action_dim=2,
        action_min=torch.tensor([0.0, 100.0]),
        action_max=torch.tensor([10.0, 200.0]),
        action_mean=torch.tensor([5.0, 150.0]),
        action_std=torch.tensor([2.5, 25.0]),
        action_q01=torch.tensor([1.0, 110.0]),
        action_q99=torch.tensor([9.0, 190.0]),
    )

    raw_action = torch.tensor([5.0, 150.0])  # Exactly midpoints

    # 1. Test Min-Max
    ds_minmax = LeRobotDataset("mock", hf_dataset=[{"action": raw_action}], metadata=meta, norm_mode="min_max")
    norm_minmax = ds_minmax.normalize_action(raw_action)
    assert pytest.approx(norm_minmax[0].item(), abs=1e-5) == 0.0
    assert pytest.approx(norm_minmax[1].item(), abs=1e-5) == 0.0

    unnorm_minmax = ds_minmax.unnormalize_action(norm_minmax)
    assert pytest.approx(unnorm_minmax[0].item(), abs=1e-5) == 5.0
    assert pytest.approx(unnorm_minmax[1].item(), abs=1e-5) == 150.0

    # 2. Test Mean-Std
    ds_meanstd = LeRobotDataset("mock", hf_dataset=[{"action": raw_action}], metadata=meta, norm_mode="mean_std")
    norm_meanstd = ds_meanstd.normalize_action(raw_action)
    assert pytest.approx(norm_meanstd[0].item(), abs=1e-5) == 0.0
    assert pytest.approx(norm_meanstd[1].item(), abs=1e-5) == 0.0


def test_lerobot_action_chunking_episode_boundary():
    """
    Test action chunking when approaching episode boundary.
    Episode 0 has 3 steps (idx 0, 1, 2).
    Episode 1 has 3 steps (idx 3, 4, 5).
    For idx 1 with chunk_size 4, lookahead reaches steps 1, 2, and then must repeat step 2
    without bleeding into Episode 1!
    """
    mock_items = [
        {"action": [0.0], "episode_index": 0},
        {"action": [1.0], "episode_index": 0},
        {"action": [2.0], "episode_index": 0},  # Last step of episode 0
        {"action": [99.0], "episode_index": 1},  # Start of episode 1
        {"action": [100.0], "episode_index": 1},
        {"action": [101.0], "episode_index": 1},
    ]

    meta = LeRobotMetadata(action_dim=1)
    ds = LeRobotDataset(
        repo_id_or_path="mock",
        hf_dataset=mock_items,
        metadata=meta,
        chunk_size=4,
        normalize_actions=False,
    )

    # Sample at idx 1 (step 1 of ep 0)
    # Expected chunk of 4: [1.0, 2.0, 2.0, 2.0] (repeated terminal step 2.0)
    sample = ds[1]
    actions = sample["actions"]
    assert actions.shape == (4, 1)
    assert actions[0, 0].item() == 1.0
    assert actions[1, 0].item() == 2.0
    assert actions[2, 0].item() == 2.0  # boundary protection
    assert actions[3, 0].item() == 2.0  # boundary protection
    # Must NEVER contain 99.0 from episode 1
    assert 99.0 not in actions.squeeze(-1).tolist()


def test_lerobot_collator_auto_configuration():
    """Test get_collator auto-configures action_dim and bounds."""
    meta = LeRobotMetadata(
        action_dim=7,
        action_min=torch.tensor([-1.0] * 7),
        action_max=torch.tensor([1.0] * 7),
    )
    mock_items = [{"action": [0.0] * 7, "episode_index": 0}]

    ds = LeRobotDataset("mock", hf_dataset=mock_items, metadata=meta, chunk_size=16)

    dummy_tokenizer = MagicMock()
    collator = ds.get_collator(tokenizer=dummy_tokenizer)

    assert isinstance(collator, UnslothVLACollator)
    assert collator.action_dim == 7
    assert collator.chunk_size == 16
