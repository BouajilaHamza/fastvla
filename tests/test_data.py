import numpy as np
from unittest.mock import patch
from fastvla import get_dataset, LeRobotDataset

def test_get_dataset_factory():
    """Test the get_dataset factory returns correct classes."""
    # Mocking the initialization to avoid load_dataset calls
    with patch("fastvla.data.datasets.LeRobotDataset._load_data", return_value=[]):
        ds = get_dataset("pusht")
        assert isinstance(ds, LeRobotDataset)
        assert ds.data_path == "lerobot/pusht_image"

def test_le_robot_dataset_mock():
    """Test LeRobotDataset mock formatting."""
    with patch("fastvla.data.datasets.LeRobotDataset._load_data", return_value=[]):
        ds = LeRobotDataset("mock_hf_repo")
    
        # Mock some data
        mock_data = [
            {
                "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                "state": np.zeros(7, dtype=np.float32),
                "action": np.zeros(7, dtype=np.float32),
                "instruction": "push"
            }
        ]
        ds.data = mock_data # Inject mock data
        
        sample = ds[0]
        
        # Check robotics standard keys
        assert "images" in sample
        assert "rgb" in sample["images"]
        assert sample["images"]["rgb"].shape == (3, 224, 224)
        assert "actions" in sample
        assert "instructions" in sample
        assert sample["instructions"] == "push"

def test_dataset_kwargs():
    """Test passing image_size to dataset."""
    with patch("fastvla.data.datasets.LeRobotDataset._load_data", return_value=[]):
        ds = LeRobotDataset("mock_hf_repo", image_size=(112, 112))
        
        mock_data = [
            {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)}
        ]
        ds.data = mock_data
        
        sample = ds[0]
        assert sample["images"]["rgb"].shape == (3, 112, 112)
