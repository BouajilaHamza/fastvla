import os
from typing import Dict, List, Tuple
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class RoboticsDataset(Dataset):
    """Base class for robotics datasets."""
    
    def __init__(
        self,
        data_path: str,
        image_keys: List[str] = ['rgb'],
        state_key: str = 'state',
        action_key: str = 'action',
        instruction_key: str = 'instruction',
        image_size: Tuple[int, int] = (224, 224),
        max_sequence_length: int = 512,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset directory or file
            image_keys: List of camera view keys
            state_key: Key for robot state in the dataset
            action_key: Key for action in the dataset
            instruction_key: Key for language instruction
            image_size: Size to resize images to (H, W)
            max_sequence_length: Maximum sequence length for text
        """
        self.data_path = data_path
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.instruction_key = instruction_key
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load dataset from disk."""
        raise NotImplementedError("Subclasses must implement _load_data")
    
    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Process a single image."""
        # Convert to PIL Image if not already
        if not isinstance(image, Image.Image):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Resize and convert to tensor
        image = image.resize((self.image_size[1], self.image_size[0]))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        return image
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        item = self.data[idx]
        
        # Process images
        images = {}
        for cam in self.image_keys:
            if cam in item:
                images[cam] = self._process_image(item[cam])
        
        # Process other data
        sample = {
            'images': images,
            'states': torch.FloatTensor(item.get(self.state_key, [])),
            'actions': torch.FloatTensor(item.get(self.action_key, [])),
            'instructions': item.get(self.instruction_key, ''),
        }
        
        return sample


class LIBERODataset(RoboticsDataset):
    """LIBERO-10 dataset for robotic manipulation tasks."""
    
    def _load_data(self):
        """Load LIBERO-10 dataset."""
        data = []
        
        # LIBERO-10 stores each episode in a separate HDF5 file
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.hdf5'):
                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, 'r') as f:
                        # Get episode length
                        ep_len = len(f['observations'])
                        
                        # Create an entry for each timestep
                        for t in range(ep_len):
                            obs = f['observations'][t]
                            data.append({
                                'rgb': obs['images']['agentview_image'],
                                'state': obs['joint_positions'],
                                'action': f['actions'][t],
                                'instruction': f.attrs.get('language_instruction', '')
                            })
        
        return data


class FrankaKitchenDataset(RoboticsDataset):
    """Franka Kitchen dataset for robotic manipulation tasks."""
    
    def _load_data(self):
        """Load Franka Kitchen dataset."""
        data = []
        
        with h5py.File(self.data_path, 'r') as f:
            # Get dataset statistics
            total_timesteps = f['observations/qpos'].shape[0]
            
            # Create an entry for each timestep
            for t in range(total_timesteps):
                data.append({
                    'rgb': f['observations/images/agentview_image'][t],
                    'state': np.concatenate([
                        f['observations/qpos'][t],
                        f['observations/qvel'][t]
                    ]),
                    'action': f['actions'][t],
                    'instruction': 'Complete the kitchen task'  # Placeholder
                })
        
        return data


def get_dataset(
    dataset_name: str,
    data_path: str,
    **kwargs
) -> RoboticsDataset:
    """Factory function to get the appropriate dataset."""
    dataset_map = {
        'libero': LIBERODataset,
        'franka_kitchen': FrankaKitchenDataset,
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_map[dataset_name.lower()](data_path, **kwargs)
