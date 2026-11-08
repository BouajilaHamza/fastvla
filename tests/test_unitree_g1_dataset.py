"""
Unit tests for UnitreeG1Dataset loader in FastVLA.
Verifies continuous 23-DoF joint actions, multi-step chunking (H=16),
Arabic instruction variation sampling, and get_dataset factory resolution.
"""

import os
import pytest
import numpy as np
import torch
import pandas as pd
import tempfile

from fastvla.data.datasets import UnitreeG1Dataset, get_dataset


@pytest.fixture
def mock_g1_parquet_dir():
    """Create a temporary parquet file with mock 23-DoF Unitree G1 trajectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.parquet")

        # 2 episodes: 10 frames and 20 frames of 23-DoF actions
        ep1_dof = np.random.randn(10, 23).astype(np.float32).tolist()
        ep2_dof = np.random.randn(20, 23).astype(np.float32).tolist()

        df = pd.DataFrame([
            {
                "episode_id": 0,
                "motion_file": "kungfu_retargeted/Roundhouse_kick.pkl",
                "motion_name": "Roundhouse_kick",
                "category": "martial_arts",
                "instruction_ar": "نفذ ركلة دائرية سريعة وقوية بالساق",
                "instruction_en": "Execute a high-velocity roundhouse kick.",
                "variations_ar": ["اركل ركلة دائرية قوية", "سدد ركلة راوند هاوس"],
                "fps": 30,
                "num_frames": 10,
                "duration_seconds": 0.333,
                "dof_trajectory": ep1_dof,
                "root_trans_offset": np.zeros((10, 3)).tolist(),
                "root_rot": np.zeros((10, 4)).tolist(),
                "contact_mask": np.ones((10, 2)).tolist(),
            },
            {
                "episode_id": 1,
                "motion_file": "dance_db_retargeted/Hasaposerviko.pkl",
                "motion_name": "Hasaposerviko",
                "category": "dance",
                "instruction_ar": "أدِّ رقصة هاسابوسيرفيكو اليونانية السريعة",
                "instruction_en": "Perform the fast-paced Greek Hasaposerviko dance.",
                "variations_ar": ["ارقص هاسابوسيرفيكو بخطوات سريعة"],
                "fps": 30,
                "num_frames": 20,
                "duration_seconds": 0.667,
                "dof_trajectory": ep2_dof,
                "root_trans_offset": np.zeros((20, 3)).tolist(),
                "root_rot": np.zeros((20, 4)).tolist(),
                "contact_mask": np.ones((20, 2)).tolist(),
            }
        ])
        df.to_parquet(train_path, index=False)
        yield tmpdir


def test_unitree_g1_dataset_loading_and_shapes(mock_g1_parquet_dir):
    chunk_size = 16
    ds = UnitreeG1Dataset(
        data_path=mock_g1_parquet_dir,
        split="train",
        chunk_size=chunk_size,
        language="ar",
        use_variations=True,
    )

    # 10 + 20 = 30 total timesteps
    assert len(ds) == 30

    sample = ds[0]
    assert "states" in sample
    assert "actions" in sample
    assert "instructions" in sample

    # Check shapes
    assert sample["states"].shape == (23,)
    assert sample["actions"].shape == (chunk_size, 23)
    assert isinstance(sample["instructions"], str)
    assert len(sample["instructions"]) > 0


def test_unitree_g1_tail_boundary_repeat(mock_g1_parquet_dir):
    """Verify that at the end of an episode, actions are repeated rather than crossing boundaries."""
    chunk_size = 8
    ds = UnitreeG1Dataset(
        data_path=mock_g1_parquet_dir,
        split="train",
        chunk_size=chunk_size,
    )

    # Step 9 is the last frame of episode 0 (which has length 10)
    sample_tail = ds[9]
    actions = sample_tail["actions"]  # [8, 23]
    # For lookaheads beyond frame 9, action must match frame 9
    for i in range(1, chunk_size):
        assert torch.allclose(actions[i], actions[0])


def test_unitree_g1_factory_resolution(mock_g1_parquet_dir):
    ds = get_dataset(
        "unitree_g1",
        data_path=mock_g1_parquet_dir,
        split="train",
        chunk_size=4,
    )
    assert isinstance(ds, UnitreeG1Dataset)
    assert len(ds) == 30
