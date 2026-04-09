import torch
import pytest
import numpy as np
from unittest.mock import MagicMock
from fastvla.data.collator import UnslothVLACollator

# Setup
tokenizer = MagicMock()
tokenizer.pad_token_id = 0
collator = UnslothVLACollator(tokenizer=tokenizer, action_dim=7)

def test_collator_nan_rejection():
    """Verify that the collator handles NaN values in actions if they occur."""
    batch = [
        {
            "images": [torch.randn(3, 224, 224)], # Correct key 'images'
            "input_ids": [1, 2, 3],
            "actions": [np.nan, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0]
        }
    ]
    
    # Must collate without crashing
    output = collator(batch)
    assert "pixel_values" in output
    assert torch.isnan(output["labels"]).any()

def test_collator_empty_sequences():
    """Verify collator behavior with minimum possible sequence lengths."""
    batch = [
        {
            "images": [torch.randn(3, 224, 224)],
            "input_ids": [1], # Minimum 1 token
            "actions": [0.1] * 7
        }
    ]
    output = collator(batch)
    assert output["input_ids"].shape == (1, 1)
    assert output["pixel_values"].shape == (1, 1, 3, 224, 224)

def test_collator_mismatched_cams():
    """
    Verify that the collator handles items with different numbers of images using list inputs.
    """
    batch = [
        {
            "images": [torch.zeros(3, 224, 224)], # 1 cam
            "actions": [0.0] * 7
        },
        {
            "images": [torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)], # 2 cams
            "actions": [0.0] * 7
        }
    ]
    
    # Should pad to 2 cameras automatically
    output = collator(batch)
    assert output["pixel_values"].shape == (2, 2, 3, 224, 224)

def test_collator_dict_cams():
    """Verify that the collator handles named camera dictionaries."""
    batch = [
        {
            "images": {"wrist": torch.zeros(3, 224, 224)},
            "actions": [0.0] * 7
        },
        {
            "images": {"wrist": torch.zeros(3, 224, 224), "ego": torch.zeros(3, 224, 224)},
            "actions": [0.0] * 7
        }
    ]
    
    output = collator(batch)
    assert output["pixel_values"].shape == (2, 2, 3, 224, 224)

def test_collator_invalid_action_dim():
    """Verify that the collator handles actions with wrong dimensions (updates action_dim)."""
    batch = [
        {
            "images": [torch.zeros(3, 224, 224)],
            "actions": [0.1] * 3 # WRONG DIM (7 expected)
        }
    ]
    
    # Current implementation warns and updates action_dim instead of crashing
    output = collator(batch)
    assert output["labels"].shape[-1] == 3
