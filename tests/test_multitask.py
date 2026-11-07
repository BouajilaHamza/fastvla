"""
Unit tests for FastVLA Multi-Task Robotics Scaling & Active Data Augmentation.
Verifies multi-suite dataset collation (LIBERO-10, LIBERO-Spatial, LIBERO-Object),
Arabic dialect perturbations, and continuous action chunking.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from fastvla.data.collator import UnslothVLACollator
from fastvla.data.augmentations import RobustRoboticsVisionAugmentor, ArabicLanguagePerturber
from fastvla.adapters.token_compressor import SpatialPatchMerger
from fastvla.adapters.action_head import FlowMatchingActionHead


def test_multitask_collator_variable_task_lengths():
    """Verify UnslothVLACollator seamlessly handles heterogeneous multi-suite tasks."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros((3, 16), dtype=torch.long),
        "attention_mask": torch.ones((3, 16), dtype=torch.long),
    }

    vision_aug = RobustRoboticsVisionAugmentor(camera_dropout_prob=0.0)
    lang_aug = ArabicLanguagePerturber(dialect_prob=0.5, orthographic_noise_prob=0.3)

    collator = UnslothVLACollator(
        tokenizer=mock_tokenizer,
        vision_augmentor=vision_aug,
        language_perturber=lang_aug,
        action_dim=7,
        image_size=224,
    )

    # 3 distinct tasks from 3 distinct suites:
    # 1. LIBERO-10: Stove and moka pot
    # 2. LIBERO-Spatial: Bowl between plate and ramekin
    # 3. LIBERO-Object: Alphabet soup in basket
    batch_features = [
        {
            "images": {"agent": torch.rand(3, 224, 224), "wrist": torch.rand(3, 224, 224)},
            "actions": torch.randn(7),
            "instruction": "شغّل الموقد وضع إبريق الموكا عليه",
        },
        {
            "images": {"agent": torch.rand(3, 224, 224), "wrist": torch.rand(3, 224, 224)},
            "actions": torch.randn(7),
            "instruction": "التقط الوعاء الأسود الموجود بين الطبق وقالب الخَبز وضعه على الطبق",
        },
        {
            "images": {"agent": torch.rand(3, 224, 224), "wrist": torch.rand(3, 224, 224)},
            "actions": torch.randn(7),
            "instruction": "التقط حساء الحروف وضعه في السلة",
        },
    ]

    out = collator(batch_features)

    assert "pixel_values" in out
    # Batch shape: [B=3, Cams=2, C=3, H=224, W=224]
    assert out["pixel_values"].shape == (3, 2, 3, 224, 224)
    assert "labels" in out
    assert out["labels"].shape == (3, 7)
    assert "input_ids" in out
    assert "instructions" in out
    assert len(out["instructions"]) == 3


def test_multitask_patch_compressor_and_flow_loss():
    """Verify FastVLA multi-task pipeline trains end-to-end with patch compression & flow loss."""
    torch.manual_seed(42)
    B, vision_dim, hidden_dim, action_dim = 2, 1152, 512, 7

    compressor = SpatialPatchMerger(in_dim=vision_dim, out_dim=hidden_dim, merge_factor=2)
    action_head = FlowMatchingActionHead(
        input_dim=hidden_dim,
        action_dim=action_dim,
        hidden_dim=128,
        manifold="se3",
        rotation_format="7dof_euler",
    )

    # 512 raw patches per multi-camera sample (2 cams x 256) -> compressed to 128 tokens
    patches = torch.randn(B, 512, vision_dim)
    compressed = compressor(patches, num_cams=2)
    assert compressed.shape == (B, 128, hidden_dim)

    # Mean pooled conditioning
    cond = compressed.mean(dim=1)
    target_actions = torch.randn(B, action_dim)

    # Flow matching loss
    loss = action_head.compute_loss(cond, target_actions)
    assert loss is not None and not torch.isnan(loss) and loss.item() > 0.0

    loss.backward()
    for name, p in action_head.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
