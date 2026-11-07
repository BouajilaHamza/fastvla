"""
Robust Multi-Modal Data Augmentation for Robotics Vision-Language-Action Models.

Provides:
1. RobustRoboticsVisionAugmentor:
   - GPU & CPU compatible photometric and geometric perturbations.
   - Spatial shifts, crops, color jitter, sensor noise, and multi-camera view dropout.
2. ArabicLanguagePerturber:
   - Robust dialectal synonym substitution (Levantine, Egyptian, Gulf vs Modern Standard Arabic).
   - Real-world orthographic noise (omitted hamzas, taa marbuta / haa interchangeability, alef variations).
   - Paraphrasing and instruction jittering.
"""

import math
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class RobustRoboticsVisionAugmentor(nn.Module):
    """
    Domain-invariant visual augmentation pipeline for robotics manipulation.
    Applies spatial shifts, photometric distortions, sensor noise, and camera dropout.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        shift_ratio: float = 0.10,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
        camera_dropout_prob: float = 0.10,
        enable_blur: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.shift_ratio = shift_ratio
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.noise_std = noise_std
        self.camera_dropout_prob = camera_dropout_prob
        self.enable_blur = enable_blur

    def _random_crop_and_shift(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [..., C, H, W] normalized in [0, 1]
        Applies random spatial translation and resize back to original image_size.
        """
        orig_shape = images.shape
        c, h, w = orig_shape[-3], orig_shape[-2], orig_shape[-1]
        flat_imgs = images.view(-1, c, h, w)
        batch_size = flat_imgs.shape[0]

        max_dx = int(w * self.shift_ratio)
        max_dy = int(h * self.shift_ratio)

        # Pad and random crop
        pad_h = max_dy
        pad_w = max_dx
        padded = F.pad(flat_imgs, (pad_w, pad_w, pad_h, pad_h), mode="replicate")

        crop_y = torch.randint(0, 2 * pad_h + 1, (batch_size,), device=images.device)
        crop_x = torch.randint(0, 2 * pad_w + 1, (batch_size,), device=images.device)

        cropped_list = []
        for i in range(batch_size):
            y1 = crop_y[i].item()
            x1 = crop_x[i].item()
            cropped = padded[i : i + 1, :, y1 : y1 + h, x1 : x1 + w]
            cropped_list.append(cropped)

        out = torch.cat(cropped_list, dim=0)
        return out.view(orig_shape)

    def _color_jitter(self, images: torch.Tensor) -> torch.Tensor:
        """Applies brightness, contrast, and saturation jitter."""
        orig_shape = images.shape
        c, h, w = orig_shape[-3], orig_shape[-2], orig_shape[-1]
        flat_imgs = images.view(-1, c, h, w)

        # Brightness factor
        if self.brightness_range[0] != self.brightness_range[1]:
            b_factor = (
                torch.empty(flat_imgs.shape[0], 1, 1, 1, device=images.device).uniform_(
                    self.brightness_range[0], self.brightness_range[1]
                )
            )
            flat_imgs = flat_imgs * b_factor

        # Contrast factor
        if self.contrast_range[0] != self.contrast_range[1]:
            c_factor = (
                torch.empty(flat_imgs.shape[0], 1, 1, 1, device=images.device).uniform_(
                    self.contrast_range[0], self.contrast_range[1]
                )
            )
            mean = flat_imgs.mean(dim=(-3, -2, -1), keepdim=True)
            flat_imgs = (flat_imgs - mean) * c_factor + mean

        # Clamp into valid range
        flat_imgs = flat_imgs.clamp(0.0, 1.0)
        return flat_imgs.view(orig_shape)

    def _add_sensor_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Adds Gaussian noise modeling sensor noise in physical robot cameras."""
        if self.noise_std > 0:
            noise = torch.randn_like(images) * self.noise_std
            images = (images + noise).clamp(0.0, 1.0)
        return images

    def _apply_camera_dropout(
        self, images_dict_or_tensor: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Randomly masks out one camera view with probability camera_dropout_prob,
        forcing the policy to not rely on a single view.
        """
        if isinstance(images_dict_or_tensor, dict):
            keys = list(images_dict_or_tensor.keys())
            if len(keys) > 1 and random.random() < self.camera_dropout_prob:
                drop_key = random.choice(keys)
                images_dict_or_tensor[drop_key] = torch.zeros_like(
                    images_dict_or_tensor[drop_key]
                )
            return images_dict_or_tensor

        if torch.is_tensor(images_dict_or_tensor):
            # Shape [B, num_cams, C, H, W]
            if images_dict_or_tensor.dim() >= 5 and images_dict_or_tensor.shape[1] > 1:
                if random.random() < self.camera_dropout_prob:
                    drop_idx = random.randint(0, images_dict_or_tensor.shape[1] - 1)
                    images_dict_or_tensor[:, drop_idx] = 0.0
            return images_dict_or_tensor

        return images_dict_or_tensor

    def forward(
        self,
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        apply_dropout: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply full vision augmentation pipeline."""
        if isinstance(images, dict):
            aug_dict = {}
            for k, v in images.items():
                x = self._random_crop_and_shift(v)
                x = self._color_jitter(x)
                x = self._add_sensor_noise(x)
                aug_dict[k] = x
            if apply_dropout:
                aug_dict = self._apply_camera_dropout(aug_dict)
            return aug_dict

        x = self._random_crop_and_shift(images)
        x = self._color_jitter(x)
        x = self._add_sensor_noise(x)
        if apply_dropout:
            x = self._apply_camera_dropout(x)
        return x


class ArabicLanguagePerturber:
    """
    Robust linguistic perturber for Arabic robotics instructions.
    Simulates real-world colloquial phrasing, dialectal variance, and spelling noise.
    """

    # Dialectal and synonym variations across Arabic dialects
    SYNONYM_MAP = {
        # Verbs: Put / Place
        "ضع": ["حط", "ركّب", "اترك", "ضع"],
        "ضعه": ["حطه", "ركبه", "اتركه"],
        "وضعه": ["وحطه", "وركبه"],
        # Verbs: Pick up / Take
        "التقط": ["شيل", "امسك", "خذ", "ارفد", "التقط"],
        "التقطه": ["شيله", "امسكه", "خذه"],
        # Verbs: Push
        "ادفع": ["دف", "زق", "ادفع", "حرّك"],
        "ادفعها": ["دفها", "زقها", "ادفعها"],
        # Verbs: Turn on / Activate
        "شغّل": ["ولّع", "افتح", "دوّر", "شغّل"],
        # Verbs: Open / Close
        "افتح": ["فك", "حل", "افتح"],
        "أغلق": ["سكّر", "اقفل", "أغلق"],
        # Nouns: Stove / Burner
        "الموقد": ["البوتجاز", "الغاز", "الفرن", "الموقد", "الطباخ"],
        "موقد": ["بوتجاز", "غاز", "فرن", "موقد"],
        # Nouns: Cabinet / Cupboard / Drawer
        "الخزانة": ["الدولاب", "الصالابيه", "الخزانة"],
        "الدرج": ["الجرار", "المجر", "الدرج"],
        # Nouns: Cup / Mug / Plate
        "الكوب": ["الكاسة", "الفنجان", "المج", "الكوب"],
        "كوب": ["كاسة", "فنجان", "مج", "كوب"],
        "الطبق": ["الصحن", "الماعون", "الطبق"],
        # Spatial prepositions
        "على": ["فوق", "على"],
        "داخل": ["في", "جوّه", "داخل"],
        "إلى": ["نحو", "لـ", "إلى"],
        "الأيمن": ["اليمين", "الأيمن"],
        "الأيسر": ["الشمال", "اليسار", "الأيسر"],
    }

    def __init__(
        self,
        dialect_prob: float = 0.35,
        orthographic_noise_prob: float = 0.25,
        paraphrase_prob: float = 0.20,
    ):
        self.dialect_prob = dialect_prob
        self.orthographic_noise_prob = orthographic_noise_prob
        self.paraphrase_prob = paraphrase_prob

    def perturb_dialect(self, text: str) -> str:
        """Substitute words with dialectal synonyms (Egyptian, Levantine, Gulf)."""
        words = text.split()
        perturbed_words = []
        for w in words:
            # Strip common punctuation for lookup
            clean_w = w.strip(".,;:!?،")
            if clean_w in self.SYNONYM_MAP and random.random() < self.dialect_prob:
                replacement = random.choice(self.SYNONYM_MAP[clean_w])
                # Preserve punctuation if present
                if w.endswith("،"):
                    replacement += "،"
                perturbed_words.append(replacement)
            else:
                perturbed_words.append(w)
        return " ".join(perturbed_words)

    def perturb_orthography(self, text: str) -> str:
        """
        Simulate common Arabic internet / OCR typing errors:
        - Omission or mixup of hamzat (أ, إ, آ -> ا)
        - Interchangeable taa marbuta and haa (ة <-> ه)
        - Alif maqsura and yaa (ى <-> ي)
        """
        chars = list(text)
        for i, c in enumerate(chars):
            if random.random() < self.orthographic_noise_prob:
                if c in ["أ", "إ", "آ"]:
                    chars[i] = "ا"
                elif c == "ة":
                    chars[i] = "ه"
                elif c == "ه" and i == len(chars) - 1:
                    chars[i] = "ة"
                elif c == "ى":
                    chars[i] = "ي"
        return "".join(chars)

    def add_filler_particles(self, text: str) -> str:
        """Add conversational command prefixes (e.g. من فضلك, يلا, بالراحة)."""
        if random.random() < self.paraphrase_prob:
            fillers = ["من فضلك ", "رجاءً ", "لو سمحت ", "يلا ", "بسرعة "]
            return random.choice(fillers) + text
        return text

    def __call__(self, instruction: str) -> str:
        """Run full Arabic perturbation pipeline."""
        if not instruction or not isinstance(instruction, str):
            return instruction

        out = instruction
        # 1. Dialectal synonym injection
        out = self.perturb_dialect(out)
        # 2. Real-world orthographic noise
        out = self.perturb_orthography(out)
        # 3. Conversational command wrappers
        out = self.add_filler_particles(out)
        return out
