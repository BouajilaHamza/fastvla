"""Pytest configuration and fixtures for FastVLA tests."""

import pytest
import torch
import numpy as np
from fastvla import FastVLAConfig

torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def test_config():
    """Create a test configuration (dummy mode)."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        num_attention_heads=4,
        vocab_size=500,
        action_dim=7,
        action_hidden_dim=32,
        gradient_checkpointing=False,
        use_peft=False,
        load_in_4bit=False,
    )
    config.batch_size = 2
    return config


@pytest.fixture
def test_batch():
    """Create a test batch of data."""
    batch_size = 2
    seq_len = 8
    images = torch.randn(batch_size, 3, 3, 32, 32)
    input_ids = torch.randint(0, 500, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    actions = torch.randn(batch_size, 7)

    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": actions,
    }
