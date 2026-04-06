from dataclasses import dataclass
from typing import Dict, List, Union
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class UnslothVLACollator:
    """
    Data collator for FastVLA that handles:
    - Multi-camera image batching (stacks into [B, num_cams, C, H, W])
    - Variable-length sequences
    - Mixed data types (images, states, actions, text)
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    padding: Union[bool, str] = True
    return_tensors: str = "pt"
    action_dim: int = 7  # Expected action dimension for validation

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            features: List of feature dictionaries from the dataset

        Returns:
            Batch dictionary with:
                - pixel_values: [B, num_cams, C, H, W] tensor
                - input_ids: [B, seq_len]
                - attention_mask: [B, seq_len]
                - labels: [B, action_dim]
        """
        batch = {}

        # ── Handle images: stack into [B, num_cams, C, H, W] ─────────
        if "images" in features[0]:
            # Collect all unique camera keys preserving order
            camera_keys = []
            for feature in features:
                for cam in feature["images"]:
                    if cam not in camera_keys:
                        camera_keys.append(cam)

            # Stack: [B, num_cams, C, H, W]
            cam_images = []
            for feature in features:
                cam_list = [feature["images"].get(cam, torch.zeros(3, 224, 224)) for cam in camera_keys]
                cam_images.append(torch.stack(cam_list, dim=0))  # [num_cams, C, H, W]

            batch["pixel_values"] = torch.stack(cam_images, dim=0)  # [B, num_cams, C, H, W]

        # ── Handle states ────────────────────────────────────────────
        if "states" in features[0]:
            states = [torch.as_tensor(f["states"]) for f in features]
            batch["states"] = torch.stack(states)

        # ── Handle actions (used as labels) ──────────────────────────
        if "actions" in features[0]:
            actions = []
            for f in features:
                action_tensor = torch.as_tensor(f["actions"])
                # Validate action dimension
                if action_tensor.dim() == 0:
                    # Scalar action, reshape to [1]
                    action_tensor = action_tensor.unsqueeze(0)
                actions.append(action_tensor)
            
            batch["labels"] = torch.stack(actions)
            
            # Validate action dimensions are consistent
            action_shapes = [a.shape for a in actions]
            if len(set(action_shapes)) > 1:
                raise ValueError(
                    f"Inconsistent action dimensions in batch: {action_shapes}. "
                    f"All actions must have the same dimension. Expected {self.action_dim}."
                )
            
            # Check if action dimension matches expected
            if actions[0].shape[-1] != self.action_dim:
                print(
                    f"⚠️ Warning: Action dimension mismatch. "
                    f"Expected {self.action_dim}, got {actions[0].shape[-1]}. "
                    f"Updating action_dim to {actions[0].shape[-1]}."
                )
                self.action_dim = actions[0].shape[-1]

        # ── Handle text instructions ────────────────────────────────
        if "instructions" in features[0]:
            texts = [f["instructions"] for f in features]
            text_inputs = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            batch["input_ids"] = text_inputs["input_ids"]
            batch["attention_mask"] = text_inputs["attention_mask"]
        else:
            # Provide default empty text input if instructions are missing
            # This prevents model crashes when text data is missing
            if hasattr(self.tokenizer, 'pad_token_id'):
                batch["input_ids"] = torch.full(
                    (len(features), 1),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=torch.long
                )
            else:
                batch["input_ids"] = torch.zeros((len(features), 1), dtype=torch.long)
            batch["attention_mask"] = torch.ones((len(features), 1), dtype=torch.long)

        # Validate required keys exist
        required_keys = ["pixel_values", "input_ids", "labels"]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise ValueError(
                f"Batch is missing required keys: {missing_keys}. "
                f"Ensure your dataset provides images, instructions (text), and actions."
            )

        return batch
