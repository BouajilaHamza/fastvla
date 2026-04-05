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
            actions = [torch.as_tensor(f["actions"]) for f in features]
            batch["labels"] = torch.stack(actions)

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

        return batch
