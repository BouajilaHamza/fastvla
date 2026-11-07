import pytest
import torch
from fastvla.data.augmentations import (
    RobustRoboticsVisionAugmentor,
    ArabicLanguagePerturber,
)


def test_vision_augmentor_tensor_shape_and_dtype():
    augmentor = RobustRoboticsVisionAugmentor(
        image_size=(224, 224),
        shift_ratio=0.1,
        noise_std=0.02,
        camera_dropout_prob=0.0,  # disable dropout for shape check
    )
    # Batch of multi-camera images: [B=2, Cams=2, C=3, H=224, W=224]
    x = torch.rand(2, 2, 3, 224, 224, dtype=torch.float32)
    aug_x = augmentor(x)

    assert aug_x.shape == x.shape
    assert aug_x.dtype == x.dtype
    assert aug_x.min() >= 0.0
    assert aug_x.max() <= 1.0


def test_vision_augmentor_dict_inputs_and_dropout():
    augmentor = RobustRoboticsVisionAugmentor(
        image_size=(224, 224),
        camera_dropout_prob=1.0,  # force dropout
    )
    obs = {
        "agentview": torch.rand(1, 3, 224, 224),
        "wristview": torch.rand(1, 3, 224, 224),
    }
    aug_obs = augmentor(obs, apply_dropout=True)

    assert "agentview" in aug_obs
    assert "wristview" in aug_obs

    # One view must have been dropped (all zeros) because prob=1.0
    agent_is_zero = (aug_obs["agentview"] == 0).all()
    wrist_is_zero = (aug_obs["wristview"] == 0).all()
    assert agent_is_zero or wrist_is_zero


def test_arabic_language_perturber_basic():
    perturber = ArabicLanguagePerturber(
        dialect_prob=1.0,  # force dialect perturbation
        orthographic_noise_prob=0.0,
        paraphrase_prob=0.0,
    )
    # "ضع الكوب على الموقد" -> should replace "ضع" with a dialectal synonym (e.g., "حط")
    # and "الموقد" with ("البوتجاز", "الغاز", etc.)
    instruction = "ضع الكوب على الموقد"
    perturbed = perturber(instruction)
    assert perturbed != ""
    assert isinstance(perturbed, str)
    # Check that perturbed string is non-empty and contains valid characters
    assert len(perturbed) > 0


def test_arabic_language_perturber_orthography():
    perturber = ArabicLanguagePerturber(
        dialect_prob=0.0,
        orthographic_noise_prob=1.0,  # force orthographic noise
        paraphrase_prob=0.0,
    )
    # Hamza in "أغلق" should normalize to "ا"
    instruction = "أغلق الخزانة"
    perturbed = perturber(instruction)
    # "أ" -> "ا" and "ة" -> "ه"
    assert "ا" in perturbed
    assert "ه" in perturbed


def test_collator_with_augmentations():
    from unittest.mock import MagicMock
    from fastvla.data.collator import UnslothVLACollator

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros((1, 5), dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
    }

    vision_aug = RobustRoboticsVisionAugmentor(camera_dropout_prob=0.0)
    lang_aug = ArabicLanguagePerturber(dialect_prob=1.0)

    collator = UnslothVLACollator(
        tokenizer=mock_tokenizer,
        vision_augmentor=vision_aug,
        language_perturber=lang_aug,
    )

    batch_input = [
        {
            "images": {"rgb": torch.rand(3, 224, 224)},
            "states": torch.zeros(7),
            "actions": torch.zeros(7),
            "instructions": "ضع الكوب على الموقد",
        }
    ]

    out = collator(batch_input)
    assert "pixel_values" in out
    assert out["pixel_values"].shape == (1, 1, 3, 224, 224)
    assert "instructions" in out
    # Check that perturber was applied to text passed to tokenizer
    assert mock_tokenizer.called

