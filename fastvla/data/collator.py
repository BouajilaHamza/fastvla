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
    image_size: int = 224  # Target image size (H=W)

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
            is_dict = isinstance(features[0]["images"], dict)
            target_size = (self.image_size, self.image_size)
            
            if is_dict:
                # Collect all unique camera keys preserving order
                camera_keys = []
                for feature in features:
                    for cam_key in feature["images"]:
                        if cam_key not in camera_keys:
                            camera_keys.append(cam_key)
                
                # Stack: [B, num_cams, C, H, W]
                cam_images = []
                for feature in features:
                    cam_list = []
                    for k in camera_keys:
                        img = feature["images"].get(k, torch.zeros(3, *target_size))
                        # Resize if necessary
                        if img.shape[-2:] != target_size:
                            img = torch.nn.functional.interpolate(
                                img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
                            ).squeeze(0)
                        cam_list.append(img)
                    cam_images.append(torch.stack(cam_list, dim=0))
            else:
                # Assume list of images
                max_cams = max(len(f["images"]) for f in features)
                cam_images = []
                for feature in features:
                    imgs = []
                    for img in feature["images"]:
                        img_tensor = torch.as_tensor(img)
                        # Resize if necessary
                        if img_tensor.shape[-2:] != target_size:
                            if img_tensor.dim() == 3:
                                img_tensor = torch.nn.functional.interpolate(
                                    img_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
                                ).squeeze(0)
                        imgs.append(img_tensor)
                        
                    # Pad with zeros if necessary
                    while len(imgs) < max_cams:
                        imgs.append(torch.zeros(3, *target_size))
                    cam_images.append(torch.stack(imgs[:max_cams], dim=0))

            batch["pixel_values"] = torch.stack(cam_images, dim=0)

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
            
            # 1. Validate action dimensions are consistent within the batch
            action_shapes = [a.shape for a in actions]
            if len(set(action_shapes)) > 1:
                raise ValueError(
                    f"Inconsistent action dimensions in batch: {action_shapes}. "
                    f"All actions in a batch must have the same dimension."
                )

            # 2. Check if batch dimension matches expected and update if needed
            if actions[0].shape[-1] != self.action_dim:
                print(
                    f"⚠️ Warning: Action dimension mismatch (Batch: {actions[0].shape[-1]}, Collator: {self.action_dim}). "
                    f"Updating action_dim to match batch."
                )
                self.action_dim = actions[0].shape[-1]

            batch["labels"] = torch.stack(actions)

        # ── Handle text instructions ────────────────────────────────
        if "instructions" in features[0]:
            texts = [f["instructions"] for f in features]
            batch["instructions"] = texts  # Keep raw text for trainer-side translation
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
        required_keys = ["pixel_values", "input_ids"]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise ValueError(
                f"Batch is missing required keys: {missing_keys}. "
                f"Ensure your dataset provides images and instructions (text)."
            )

        return batch
