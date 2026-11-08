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


class UnitreeG1Dataset(RoboticsDataset):
    """
    Dataset for Unitree G1 Humanoid 23-DoF robot motions with Arabic instructions.
    
    Supports:
        - Loading from Hugging Face Hub (hamzabouajila/ar-unitree-g1-motions) or local Parquet file/dir.
        - Continuous 23-DoF whole-body joint actions with multi-step chunking (H=16).
        - Dynamic Arabic instruction variation sampling during training.
        - Tail repeat padding for boundary steps.
    """

    def __init__(
        self,
        data_path: str = "hamzabouajila/ar-unitree-g1-motions",
        split: str = "train",
        chunk_size: int = 16,
        language: str = "ar",
        use_variations: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        **kwargs,
    ):
        self.split = split
        self.language = language
        self.use_variations = use_variations
        super().__init__(
            data_path=data_path,
            chunk_size=chunk_size,
            image_size=image_size,
            **kwargs,
        )

    def _load_data(self):
        import pandas as pd

        # Resolve local file or Hugging Face Hub dataset
        if os.path.exists(self.data_path):
            if os.path.isdir(self.data_path):
                target = os.path.join(self.data_path, f"{self.split}.parquet")
                if not os.path.exists(target):
                    target = os.path.join(self.data_path, "data", f"{self.split}.parquet")
            else:
                target = self.data_path
            df = pd.read_parquet(target)
        else:
            from huggingface_hub import hf_hub_download
            parquet_file = hf_hub_download(
                repo_id=self.data_path,
                filename=f"data/{self.split}.parquet",
                repo_type="dataset",
                token=self.hf_token,
            )
            df = pd.read_parquet(parquet_file)

        episodes = []
        for ep_id, row in df.iterrows():
            raw_dof = row["dof_trajectory"]
            if isinstance(raw_dof, np.ndarray) and raw_dof.ndim == 2:
                dof = raw_dof.astype(np.float32)
            else:
                dof = np.stack([np.asarray(step, dtype=np.float32) for step in raw_dof])

            raw_root = row.get("root_trans_offset")
            if raw_root is not None:
                root_trans = raw_root if isinstance(raw_root, np.ndarray) and raw_root.ndim == 2 else np.stack([np.asarray(s, dtype=np.float32) for s in raw_root])
            else:
                root_trans = None

            raw_rot = row.get("root_rot")
            if raw_rot is not None:
                root_rot = raw_rot if isinstance(raw_rot, np.ndarray) and raw_rot.ndim == 2 else np.stack([np.asarray(s, dtype=np.float32) for s in raw_rot])
            else:
                root_rot = None

            raw_contact = row.get("contact_mask")
            if raw_contact is not None:
                contact = raw_contact if isinstance(raw_contact, np.ndarray) and raw_contact.ndim == 2 else np.stack([np.asarray(s, dtype=np.float32) for s in raw_contact])
            else:
                contact = None

            variations = list(row["variations_ar"]) if "variations_ar" in row and hasattr(row["variations_ar"], "__iter__") else []
            instr_ar = str(row.get("instruction_ar", ""))
            instr_en = str(row.get("instruction_en", ""))

            all_variations = [instr_ar] + [v for v in variations if v != instr_ar] if instr_ar else variations

            episodes.append({
                "ep_id": ep_id,
                "dof": dof,
                "root_trans": root_trans,
                "root_rot": root_rot,
                "contact": contact,
                "category": row.get("category", "locomotion"),
                "instruction_ar": instr_ar,
                "instruction_en": instr_en,
                "variations_ar": all_variations,
            })

        # Precompute step-level index
        samples = []
        for ep in episodes:
            T = ep["dof"].shape[0]
            for t in range(T):
                samples.append({"ep": ep, "t": t})

        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.data[idx]
        ep = entry["ep"]
        t = entry["t"]
        dof = ep["dof"]
        T = dof.shape[0]

        # Current robot state: 23 joint angles
        state = dof[t]

        # Action chunk: next chunk_size actions starting from t
        if self.chunk_size > 1:
            actions = []
            for step in range(self.chunk_size):
                lookahead = min(t + step, T - 1)
                actions.append(dof[lookahead])
            action_chunk = np.array(actions, dtype=np.float32)
        else:
            action_chunk = dof[t]

        # Select language prompt
        if self.language == "ar":
            if self.use_variations and ep["variations_ar"]:
                import random
                instruction = random.choice(ep["variations_ar"])
            else:
                instruction = ep["instruction_ar"]
        else:
            instruction = ep["instruction_en"]

        return {
            "images": {"rgb": torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)},
            "states": torch.FloatTensor(state),
            "actions": torch.FloatTensor(action_chunk),
            "instructions": instruction,
            "episode_id": ep["ep_id"],
        }


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
        "unitree_g1": UnitreeG1Dataset,
        "ar_unitree_g1": UnitreeG1Dataset,
        "ar-unitree-g1-motions": UnitreeG1Dataset,
        "hamzabouajila/ar-unitree-g1-motions": UnitreeG1Dataset,
    }

    name_lower = dataset_name.lower()

    # Auto-resolve path for HF datasets if not provided
    if data_path is None:
        if name_lower in ["pusht", "lerobot/pusht_image"]:
            data_path = "lerobot/pusht_image"
        elif name_lower in ["unitree_g1", "ar_unitree_g1", "ar-unitree-g1-motions", "hamzabouajila/ar-unitree-g1-motions"]:
            data_path = "hamzabouajila/ar-unitree-g1-motions"
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
