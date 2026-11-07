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
        image_keys: List[str] = ["rgb"],
        state_key: str = "state",
        action_key: str = "action",
        instruction_key: str = "instruction",
        image_size: Tuple[int, int] = (224, 224),
        max_sequence_length: int = 512,
        chunk_size: int = 1,
        **kwargs,
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
            chunk_size: Number of future actions to predict
        """
        self.data_path = data_path
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.instruction_key = instruction_key
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.chunk_size = chunk_size
        self.hf_token = kwargs.get("hf_token")

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

        # Process actions (with chunking support)
        if self.chunk_size > 1:
            # FIX 4: Always repeat the current-step action when lookahead goes
            # out of bounds or crosses an episode boundary.  The old code used
            # min(idx+i, len-1) which could pull actions from a completely
            # different episode/task for the last episode's tail steps.
            curr_ep = item.get("episode_id", 0)
            actions = []
            for i in range(self.chunk_size):
                look_ahead_idx = idx + i
                # Out of dataset bounds → repeat last valid action
                if look_ahead_idx >= len(self.data):
                    actions.append(actions[-1] if actions else item.get(self.action_key, []))
                else:
                    look_ahead_item = self.data[look_ahead_idx]
                    # Episode boundary → repeat last valid action
                    if look_ahead_item.get("episode_id", curr_ep) != curr_ep:
                        actions.append(actions[-1] if actions else item.get(self.action_key, []))
                    else:
                        actions.append(look_ahead_item.get(self.action_key, []))
            actions = np.array(actions)
        else:
            actions = item.get(self.action_key, [])

        # Process other data
        sample = {
            "images": images,
            "states": torch.FloatTensor(item.get(self.state_key, [])),
            "actions": torch.FloatTensor(actions),
            "instructions": item.get(self.instruction_key, ""),
        }

        return sample


class LIBERODataset(RoboticsDataset):
    """LIBERO-10 dataset for robotic manipulation tasks."""

    def _load_data(self):
        """Build a lazy index — store (file_path, timestep, instruction) only.

        FIX 5: The old implementation read every timestep into RAM at startup,
        which could be gigabytes for a full LIBERO dataset.  We now store only
        lightweight index records and read the actual tensors on demand in
        __getitem__.
        """
        index = []
        for root, _, files in os.walk(self.data_path):
            for file in sorted(files):
                if file.endswith(".hdf5"):
                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, "r") as f:
                        ep_len = len(f["observations"])
                        instr = f.attrs.get("language_instruction", "")
                    for t in range(ep_len):
                        index.append({
                            "_file": file_path,
                            "_t": t,
                            "instruction": instr,
                            "episode_id": file_path,  # unique per episode
                        })
        return index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """Lazy read from HDF5 — only the requested timestep is loaded."""
        entry = self.data[idx]
        with h5py.File(entry["_file"], "r") as f:
            t = entry["_t"]
            obs = {k: np.array(f["observations"][k][t]) for k in f["observations"].keys() if k != "images"}
            if "observations/images/agentview_image" in f:
                rgb = np.array(f["observations/images/agentview_image"][t])
            else:
                rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            action = np.array(f["actions"][t])

        return {
            "images": {"rgb": self._process_image(rgb)},
            "states": torch.FloatTensor(obs.get("joint_positions", np.zeros(7))),
            "actions": torch.FloatTensor(action),
            "instructions": entry["instruction"],
            "episode_id": entry["episode_id"],
        }


class FrankaKitchenDataset(RoboticsDataset):
    """Franka Kitchen dataset for robotic manipulation tasks."""

    def _load_data(self):
        """Load Franka Kitchen dataset."""
        data = []

        with h5py.File(self.data_path, "r") as f:
            # Get dataset statistics
            total_timesteps = f["observations/qpos"].shape[0]

            # Create an entry for each timestep
            for t in range(total_timesteps):
                data.append(
                    {
                        "rgb": f["observations/images/agentview_image"][t],
                        "state": np.concatenate(
                            [f["observations/qpos"][t], f["observations/qvel"][t]]
                        ),
                        "action": f["actions"][t],
                        "instruction": "Complete the kitchen task",  # Placeholder
                    }
                )

        return data


from .lerobot import LeRobotDataset, LeRobotMetadata, load_lerobot_metadata


def get_dataset(
    dataset_name: str, data_path: Optional[str] = None, **kwargs
) -> Dataset:
    """Factory function to get the appropriate dataset."""
    dataset_map = {
        "libero": LIBERODataset,
        "franka_kitchen": FrankaKitchenDataset,
        "pusht": LeRobotDataset,
        "lerobot/pusht_image": LeRobotDataset,
    }

    name_lower = dataset_name.lower()

    # Auto-resolve path for HF datasets if not provided
    if data_path is None:
        if name_lower in ["pusht", "lerobot/pusht_image"]:
            data_path = "lerobot/pusht_image"
        elif "/" in dataset_name:  # Treat as direct HF repo ID
            data_path = dataset_name
        else:
            raise ValueError(f"data_path must be provided for {dataset_name}")

    if name_lower not in dataset_map:
        # Fallback to LeRobot for unknown names (assumed to be HF repos)
        return LeRobotDataset(
            dataset_name if data_path is None else data_path, **kwargs
        )

    return dataset_map[name_lower](data_path, **kwargs)
