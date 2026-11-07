"""
Universal Hugging Face LeRobot (v2.0 / v3.0) Dataset Loader & Hub Integration for FastVLA.

Provides out-of-the-box loading, streaming, multi-camera discovery,
action chunking across episode boundaries, and continuous action normalization
from meta/stats.json for physical robotics datasets (e.g., ALOHA, SO-100, Franka, Push-T, DROID).
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    from datasets import load_dataset, Dataset as HFDataset
except ImportError:
    load_dataset = None
    HFDataset = None

logger = logging.getLogger(__name__)


def _first_not_none(*values: Any, default: Any = None) -> Any:
    """Return the first value that is not None, avoiding truth-value checks on arrays."""
    for v in values:
        if v is not None:
            return v
    return default


def _to_float_tensor(val: Any) -> torch.Tensor:
    """Safely convert list, numpy array, or tensor into a float32 torch.Tensor."""
    if val is None:
        return torch.empty(0, dtype=torch.float32)
    if isinstance(val, torch.Tensor):
        return val.clone().detach().float()
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val).float()
    return torch.tensor(val, dtype=torch.float32)


@dataclass
class LeRobotMetadata:
    """Metadata and normalization statistics parsed from LeRobot repository."""
    codebase_version: str = "v2.0"
    fps: float = 30.0
    robot_type: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    camera_keys: List[str] = field(default_factory=list)
    state_key: str = "observation.state"
    action_key: str = "action"
    action_dim: int = 7
    state_dim: int = 7
    total_episodes: int = 0
    total_frames: int = 0
    tasks: Dict[int, str] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    action_min: Optional[torch.Tensor] = None
    action_max: Optional[torch.Tensor] = None
    action_mean: Optional[torch.Tensor] = None
    action_std: Optional[torch.Tensor] = None
    action_q01: Optional[torch.Tensor] = None
    action_q99: Optional[torch.Tensor] = None
    state_min: Optional[torch.Tensor] = None
    state_max: Optional[torch.Tensor] = None


def load_lerobot_metadata(
    repo_id_or_path: str,
    hf_token: Optional[str] = None,
) -> LeRobotMetadata:
    """
    Load and parse meta/info.json, meta/stats.json, and meta/tasks.jsonl from
    either a local directory or remote Hugging Face Hub repository.
    """
    meta = LeRobotMetadata()
    is_local = os.path.isdir(repo_id_or_path)

    def _read_json_file(filename: str) -> Optional[Any]:
        if is_local:
            local_path = os.path.join(repo_id_or_path, filename)
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading local {local_path}: {e}")
            return None
        
        if hf_hub_download is None:
            return None

        # 1. Try local cache first (instant, no network latency)
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id_or_path,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
                local_files_only=True,
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

        # 2. Try remote download
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id_or_path,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # File might not exist (e.g. meta/tasks.jsonl on older formats)
            return None

    def _read_jsonl_file(filename: str) -> Optional[List[Dict[str, Any]]]:
        content_lines = []
        if is_local:
            local_path = os.path.join(repo_id_or_path, filename)
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                content_lines.append(json.loads(line))
                    return content_lines
                except Exception as e:
                    logger.warning(f"Error reading local {local_path}: {e}")
            return None

        if hf_hub_download is None:
            return None

        # 1. Try local cache first
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id_or_path,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
                local_files_only=True,
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        content_lines.append(json.loads(line))
            return content_lines
        except Exception:
            pass

        # 2. Try remote download
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id_or_path,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        content_lines.append(json.loads(line))
            return content_lines
        except Exception:
            return None

    # 1. Parse meta/info.json
    info = _read_json_file("meta/info.json")
    if info and isinstance(info, dict):
        meta.codebase_version = info.get("codebase_version", "v2.0")
        meta.fps = float(info.get("fps", 30.0))
        meta.robot_type = info.get("robot_type")
        meta.total_episodes = int(info.get("total_episodes", 0))
        meta.total_frames = int(info.get("total_frames", 0))
        meta.features = info.get("features", {})

        # Discover camera keys from features
        discovered_cams = []
        for feat_name, feat_meta in meta.features.items():
            dtype = feat_meta.get("dtype") if isinstance(feat_meta, dict) else ""
            if dtype in ["image", "video"] or feat_name.startswith("observation.images.") or "image" in feat_name:
                discovered_cams.append(feat_name)
        if discovered_cams:
            meta.camera_keys = discovered_cams

        # Deduce action & state dimensions
        if "action" in meta.features and isinstance(meta.features["action"], dict):
            shape = meta.features["action"].get("shape", [])
            if shape and isinstance(shape, list):
                meta.action_dim = int(shape[0])
        if "observation.state" in meta.features and isinstance(meta.features["observation.state"], dict):
            shape = meta.features["observation.state"].get("shape", [])
            if shape and isinstance(shape, list):
                meta.state_dim = int(shape[0])

    # 2. Parse meta/stats.json
    stats = _read_json_file("meta/stats.json")
    if stats and isinstance(stats, dict):
        meta.stats = stats
        act_stat = stats.get("action") or stats.get("actions")
        if isinstance(act_stat, dict):
            if "min" in act_stat and act_stat["min"] is not None:
                meta.action_min = torch.tensor(act_stat["min"], dtype=torch.float32)
            if "max" in act_stat and act_stat["max"] is not None:
                meta.action_max = torch.tensor(act_stat["max"], dtype=torch.float32)
            if "mean" in act_stat and act_stat["mean"] is not None:
                meta.action_mean = torch.tensor(act_stat["mean"], dtype=torch.float32)
            if "std" in act_stat and act_stat["std"] is not None:
                meta.action_std = torch.tensor(act_stat["std"], dtype=torch.float32)
            if "q01" in act_stat and act_stat["q01"] is not None:
                meta.action_q01 = torch.tensor(act_stat["q01"], dtype=torch.float32)
            if "q99" in act_stat and act_stat["q99"] is not None:
                meta.action_q99 = torch.tensor(act_stat["q99"], dtype=torch.float32)

            if meta.action_min is not None:
                meta.action_dim = meta.action_min.numel()

        st_stat = stats.get("observation.state") or stats.get("state")
        if isinstance(st_stat, dict):
            if "min" in st_stat and st_stat["min"] is not None:
                meta.state_min = torch.tensor(st_stat["min"], dtype=torch.float32)
            if "max" in st_stat and st_stat["max"] is not None:
                meta.state_max = torch.tensor(st_stat["max"], dtype=torch.float32)
            if meta.state_min is not None:
                meta.state_dim = meta.state_min.numel()

    # 3. Parse tasks mapping (meta/tasks.jsonl)
    tasks_jsonl = _read_jsonl_file("meta/tasks.jsonl")
    if tasks_jsonl:
        for entry in tasks_jsonl:
            t_idx = entry.get("task_index")
            t_name = entry.get("task") or entry.get("description")
            if t_idx is not None and t_name:
                meta.tasks[int(t_idx)] = str(t_name)

    return meta


class LeRobotDataset(Dataset):
    """
    High-performance, memory-efficient Dataset wrapper for Hugging Face LeRobot format.
    Supports Hub streaming or cached splits, dynamic multi-camera discovery,
    proper episode boundary repeating for action chunks, and continuous action normalization.
    """

    def __init__(
        self,
        repo_id_or_path: str,
        split: str = "train",
        chunk_size: int = 1,
        image_size: Tuple[int, int] = (224, 224),
        image_keys: Optional[List[str]] = None,
        normalize_actions: bool = True,
        norm_mode: str = "min_max",  # "min_max", "mean_std", or "quantile"
        default_instruction: str = "",
        hf_token: Optional[str] = None,
        hf_dataset: Optional[Any] = None,
        metadata: Optional[LeRobotMetadata] = None,
        streaming: bool = False,
    ):
        """
        Initialize the LeRobotDataset.

        Args:
            repo_id_or_path: Hugging Face repo ID (e.g. 'lerobot/pusht_image') or local directory.
            split: Dataset split ('train', 'test', etc.).
            chunk_size: Number of future actions to chunk (H >= 1).
            image_size: Target image size (H, W) for camera observations.
            image_keys: Specific camera view keys to extract (auto-discovered if None).
            normalize_actions: Whether to normalize actions using meta/stats.json.
            norm_mode: Normalization strategy ('min_max', 'mean_std', 'quantile').
            default_instruction: Fallback instruction text if dataset has no task label.
            hf_token: Optional Hugging Face auth token.
            hf_dataset: Pre-loaded Hugging Face dataset (useful for unit tests or memory reuse).
            metadata: Pre-parsed LeRobotMetadata (auto-fetched if None).
            streaming: Whether to stream data lazily from Hugging Face Hub.
        """
        self.repo_id_or_path = repo_id_or_path
        self.data_path = repo_id_or_path
        self.split = split
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.normalize_actions = normalize_actions
        self.norm_mode = norm_mode
        self.default_instruction = default_instruction
        self.hf_token = hf_token
        self.streaming = streaming

        # 1. Load or bind metadata
        if metadata is not None:
            self.metadata = metadata
        elif repo_id_or_path.startswith("mock") or repo_id_or_path == "pusht":
            self.metadata = LeRobotMetadata()
        else:
            try:
                self.metadata = load_lerobot_metadata(repo_id_or_path, hf_token=hf_token)
            except Exception as e:
                logger.warning(f"Could not load metadata for {repo_id_or_path}: {e}")
                self.metadata = LeRobotMetadata()

        # 2. Camera view keys configuration
        if image_keys is not None:
            self.image_keys = image_keys
        elif self.metadata.camera_keys:
            self.image_keys = self.metadata.camera_keys
        else:
            # Fallback heuristic
            self.image_keys = [
                "observation.image",
                "observation.images.laptop",
                "observation.images.wrist",
                "observation.images.phone",
                "observation.images.agentview",
                "image",
                "rgb",
            ]

        # 3. Load Hugging Face dataset
        if hf_dataset is not None:
            self.hf_ds = hf_dataset
        else:
            self.hf_ds = self._load_data()

        # 4. Fast episode index cache
        self._episode_indices = None
        if not streaming and hasattr(self.hf_ds, "__getitem__") and hasattr(self.hf_ds, "column_names"):
            try:
                # Fast column read in pyarrow
                if "episode_index" in self.hf_ds.column_names:
                    self._episode_indices = np.array(self.hf_ds["episode_index"])
                elif "episode_id" in self.hf_ds.column_names:
                    self._episode_indices = np.array(self.hf_ds["episode_id"])
            except Exception:
                self._episode_indices = None

    def _load_data(self) -> Any:
        """Internal loader hook that supports mocking in tests."""
        if self.repo_id_or_path == "mock_hf_repo" or self.repo_id_or_path.startswith("mock"):
            return []
        if load_dataset is None:
            raise ImportError("The 'datasets' package is required to load LeRobot datasets.")
        logger.info(f"Loading LeRobot dataset '{self.repo_id_or_path}' (split={self.split})...")
        try:
            return load_dataset(
                self.repo_id_or_path,
                split=self.split,
                token=self.hf_token,
                streaming=self.streaming,
            )
        except Exception as e:
            logger.warning(f"Failed to load dataset {self.repo_id_or_path}: {e}")
            return []

    @property
    def data(self):
        return self.hf_ds

    @data.setter
    def data(self, val):
        self.hf_ds = val
        if isinstance(val, list):
            try:
                self._episode_indices = np.array([
                    item.get("episode_index", item.get("episode_id", 0)) if isinstance(item, dict) else 0
                    for item in val
                ])
            except Exception:
                self._episode_indices = None

    def __len__(self) -> int:
        return len(self.hf_ds)

    def _get_nested(self, item: Any, path: str, default: Any = None) -> Any:
        """Deep lookup supporting dot-notation or flat dict keys."""
        if isinstance(item, dict) and path in item:
            return item[path]
        keys = path.split(".")
        curr = item
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            elif hasattr(curr, k):
                curr = getattr(curr, k)
            else:
                return default
        return curr

    def _process_image(self, img_data: Any) -> torch.Tensor:
        """Process any image data (PIL, numpy array, torch tensor) to normalized (C, H, W) tensor."""
        if isinstance(img_data, torch.Tensor):
            if img_data.dim() == 4:
                img_data = img_data.squeeze(0)
            if img_data.shape[0] != 3 and img_data.shape[-1] == 3:
                img_data = img_data.permute(2, 0, 1)
            # Resize if necessary
            target_h, target_w = self.image_size
            if img_data.shape[-2:] != (target_h, target_w):
                img_data = torch.nn.functional.interpolate(
                    img_data.unsqueeze(0).float(),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
            if img_data.dtype != torch.float32:
                img_data = img_data.float()
            if img_data.max() > 1.0:
                img_data = img_data / 255.0
            return img_data

        if not isinstance(img_data, Image.Image):
            if isinstance(img_data, np.ndarray):
                if img_data.dtype != np.uint8 and img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                elif img_data.dtype != np.uint8:
                    img_data = img_data.astype(np.uint8)
                img_data = Image.fromarray(img_data)
            else:
                img_data = Image.fromarray(np.array(img_data, dtype=np.uint8))

        # PIL image resize
        img_data = img_data.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.array(img_data).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        return tensor

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize an action tensor into standard policy space using loaded meta/stats.json.
        """
        if not self.normalize_actions:
            return action

        eps = 1e-6
        if self.norm_mode == "min_max" and self.metadata.action_min is not None and self.metadata.action_max is not None:
            n_min = self.metadata.action_min.to(action.device, dtype=action.dtype)
            n_max = self.metadata.action_max.to(action.device, dtype=action.dtype)
            normalized = 2.0 * (action - n_min) / torch.clamp(n_max - n_min, min=eps) - 1.0
            return torch.clamp(normalized, -1.0, 1.0)

        elif self.norm_mode == "mean_std" and self.metadata.action_mean is not None and self.metadata.action_std is not None:
            mean = self.metadata.action_mean.to(action.device, dtype=action.dtype)
            std = self.metadata.action_std.to(action.device, dtype=action.dtype)
            return (action - mean) / torch.clamp(std, min=eps)

        elif self.norm_mode == "quantile" and self.metadata.action_q01 is not None and self.metadata.action_q99 is not None:
            q01 = self.metadata.action_q01.to(action.device, dtype=action.dtype)
            q99 = self.metadata.action_q99.to(action.device, dtype=action.dtype)
            normalized = 2.0 * (action - q01) / torch.clamp(q99 - q01, min=eps) - 1.0
            return torch.clamp(normalized, -1.0, 1.0)

        return action

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation: convert normalized policy predictions back into raw robot action coordinates.
        """
        eps = 1e-6
        if self.norm_mode == "min_max" and self.metadata.action_min is not None and self.metadata.action_max is not None:
            n_min = self.metadata.action_min.to(action.device, dtype=action.dtype)
            n_max = self.metadata.action_max.to(action.device, dtype=action.dtype)
            return (action + 1.0) * 0.5 * torch.clamp(n_max - n_min, min=eps) + n_min

        elif self.norm_mode == "mean_std" and self.metadata.action_mean is not None and self.metadata.action_std is not None:
            mean = self.metadata.action_mean.to(action.device, dtype=action.dtype)
            std = self.metadata.action_std.to(action.device, dtype=action.dtype)
            return action * torch.clamp(std, min=eps) + mean

        elif self.norm_mode == "quantile" and self.metadata.action_q01 is not None and self.metadata.action_q99 is not None:
            q01 = self.metadata.action_q01.to(action.device, dtype=action.dtype)
            q99 = self.metadata.action_q99.to(action.device, dtype=action.dtype)
            return (action + 1.0) * 0.5 * torch.clamp(q99 - q01, min=eps) + q01

        return action

    def _get_episode_index(self, idx: int, item: Dict[str, Any]) -> int:
        if self._episode_indices is not None and idx < len(self._episode_indices):
            return int(self._episode_indices[idx])
        return int(
            self._get_nested(item, "episode_index")
            or self._get_nested(item, "episode_id")
            or 0
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single observation-action chunk sample.
        """
        item = self.hf_ds[idx]
        if not isinstance(item, dict) and hasattr(item, "keys"):
            item = dict(item)

        # 1. Multi-Camera Image Extraction
        images = {}
        for cam_path in self.image_keys:
            img_val = self._get_nested(item, cam_path)
            if img_val is not None:
                # Clean camera name: "observation.images.laptop" -> "laptop"
                clean_name = cam_path.split(".")[-1]
                images[clean_name] = self._process_image(img_val)

        # Backward compatibility fallback: ensure "rgb" key exists
        if "rgb" not in images and images:
            first_cam = next(iter(images.values()))
            images["rgb"] = first_cam

        # 2. Robot State Extraction
        state_val = _first_not_none(
            self._get_nested(item, "observation.state"),
            self._get_nested(item, "state"),
            default=[],
        )
        state_tensor = _to_float_tensor(state_val)

        # 3. Action Extraction & Chunking with Episode Boundary Protection
        curr_ep = self._get_episode_index(idx, item)
        act_key = self.metadata.action_key or "action"
        base_act = _first_not_none(
            self._get_nested(item, act_key),
            self._get_nested(item, "action"),
            self._get_nested(item, "actions"),
            default=[],
        )
        base_act_tensor = _to_float_tensor(base_act)

        if self.chunk_size > 1:
            chunked_actions = [base_act_tensor]
            for i in range(1, self.chunk_size):
                next_idx = idx + i
                if next_idx >= len(self.hf_ds):
                    # Out of bounds: repeat the last valid action
                    chunked_actions.append(chunked_actions[-1])
                else:
                    next_item = self.hf_ds[next_idx]
                    next_ep = self._get_episode_index(next_idx, next_item)
                    if next_ep != curr_ep:
                        # Episode boundary crossed: repeat last action of current episode
                        chunked_actions.append(chunked_actions[-1])
                    else:
                        act_next = _first_not_none(
                            self._get_nested(next_item, act_key),
                            self._get_nested(next_item, "action"),
                            self._get_nested(next_item, "actions"),
                            default=[],
                        )
                        chunked_actions.append(_to_float_tensor(act_next))

            raw_actions = torch.stack(chunked_actions, dim=0)  # [H, action_dim]
        else:
            raw_actions = base_act_tensor

        # 4. Action Normalization
        norm_actions = self.normalize_action(raw_actions)

        # 5. Language Instruction Extraction
        instruction = (
            self._get_nested(item, "instruction")
            or self._get_nested(item, "language_instruction")
            or self._get_nested(item, "task")
        )
        if not instruction and "task_index" in item:
            t_idx = item["task_index"]
            if t_idx is not None and int(t_idx) in self.metadata.tasks:
                instruction = self.metadata.tasks[int(t_idx)]

        if not instruction:
            instruction = self.default_instruction

        return {
            "images": images,
            "states": state_tensor,
            "actions": norm_actions,
            "raw_actions": raw_actions,
            "instructions": str(instruction),
            "episode_id": curr_ep,
        }

    def get_collator(self, tokenizer: Any, **collator_kwargs) -> Any:
        """
        Instantiate an UnslothVLACollator auto-configured with this dataset's
        action dimensions, normalization bounds, and chunk size.
        """
        from fastvla.data.collator import UnslothVLACollator

        norm_min = self.metadata.action_min
        norm_max = self.metadata.action_max
        action_dim = self.metadata.action_dim

        # If normalized internally, collator doesn't need to re-normalize
        use_collator_norm = not self.normalize_actions

        return UnslothVLACollator(
            tokenizer=tokenizer,
            action_dim=action_dim,
            chunk_size=self.chunk_size,
            norm_min=norm_min if use_collator_norm else None,
            norm_max=norm_max if use_collator_norm else None,
            **collator_kwargs,
        )
