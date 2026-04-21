import os
import logging
from typing import Dict, List, Tuple, Optional
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

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


class LeRobotDataset(RoboticsDataset):
    """Dataset wrapper for LeRobot format datasets on HuggingFace."""
    
    def _load_data(self):
        """Load dataset from HuggingFace."""
        from datasets import load_dataset
        print(f"📥 Loading dataset {self.data_path} from HuggingFace...")
        
        # Load the dataset (usually 'train' split)
        hf_ds = load_dataset(self.data_path, split='train')
        
        def get_nested(d, path, default=None):
            keys = path.split('.')
            curr = d
            for k in keys:
                try:
                    if isinstance(curr, dict) and k in curr:
                        curr = curr[k]
                    elif hasattr(curr, k):
                        curr = getattr(curr, k)
                    else:
                        return default
                except (TypeError, KeyError, AttributeError):
                    return default
            return curr

        data = []
        # LeRobot format mapping with deep key search
        for item in hf_ds:
            # Ensure we have a dict-like object
            if not hasattr(item, "get") and hasattr(item, "keys"):
                item = dict(item)
            
            # 1. Extract Images (handle nested or flat)
            img = None
            for k in ['observation.image', 'observation.images.laptop', 'observation.images.agentview', 'image']:
                val = get_nested(item, k) if '.' in k else item.get(k)
                if val is not None:
                    img = val
                    break
            
            # 2. Extract State
            state = get_nested(item, 'observation.state') or item.get('state', [])
            
            # 3. Extract Action
            action = item.get('action', [])
            
            # 4. Extract Instruction
            instruction = item.get('instruction') or item.get('language_instruction') or 'push the block to the goal'

            if img is not None:
                data.append({
                    'rgb': img,
                    'state': state,
                    'action': action,
                    'instruction': instruction
                })
            
        if not data:
            logger.warning(f"⚠️ No valid data found in {self.data_path}. Fallback to raw keys.")
            sample = hf_ds[0]
            print(f"DEBUG: Dataset Sample Keys: {list(sample.keys())}")
            # Just take the first image and state we find
            img_key = next((k for k in sample.keys() if 'image' in k or 'rgb' in k), None)
            if img_key: print(f"DEBUG: Selected fallback img_key: {img_key}")
            for item in hf_ds:
                data.append({
                    'rgb': item[img_key] if img_key else None,
                    'state': item.get('state', []),
                    'action': item.get('action', []),
                    'instruction': item.get('instruction', 'push the block to the goal')
                })

        return data


def get_dataset(
    dataset_name: str,
    data_path: Optional[str] = None,
    **kwargs
) -> RoboticsDataset:
    """Factory function to get the appropriate dataset."""
    dataset_map = {
        'libero': LIBERODataset,
        'franka_kitchen': FrankaKitchenDataset,
        'pusht': LeRobotDataset,
        'lerobot/pusht_image': LeRobotDataset,
    }
    
    name_lower = dataset_name.lower()
    
    # Auto-resolve path for HF datasets if not provided
    if data_path is None:
        if name_lower in ['pusht', 'lerobot/pusht_image']:
            data_path = "lerobot/pusht_image"
        elif "/" in dataset_name: # Treat as direct HF repo ID
            data_path = dataset_name
        else:
            raise ValueError(f"data_path must be provided for {dataset_name}")
            
    if name_lower not in dataset_map:
        # Fallback to LeRobot for unknown names (assumed to be HF repos)
        return LeRobotDataset(dataset_name if data_path is None else data_path, **kwargs)
    
    return dataset_map[name_lower](data_path, **kwargs)
