"""
Unit tests for Episodic Prompt KV Caching in FastVLAModel.
"""

import pytest
import torch
from fastvla.model import FastVLAModel, FastVLAConfig


def test_episodic_prompt_caching_lifecycle_and_numerical_equivalence():
    """Verify that episodic prompt caching matches stateless forward pass exactly."""
    torch.manual_seed(42)
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        action_dim=7,
        vocab_size=100,
    )
    model = FastVLAModel(config)
    model.eval()

    B = 2
    pixel_values = torch.randn(B, 2, 3, 32, 32)
    prompt_ids = torch.randint(0, 100, (B, 16))

    # 1. Stateless baseline forward
    with torch.no_grad():
        action_stateless, _ = model(pixel_values=pixel_values, input_ids=prompt_ids)

    # 2. Set episodic prompt cache
    assert not model.has_episodic_prompt
    model.set_episodic_prompt(prompt_ids)
    assert model.has_episodic_prompt

    # 3. Cached forward without passing input_ids
    with torch.no_grad():
        action_cached, _ = model(pixel_values=pixel_values, input_ids=None)

    # Must be 100% numerically identical
    assert torch.allclose(action_stateless, action_cached, atol=1e-5), "Cached output mismatch with stateless reference!"

    # 4. Clear cache and verify fallback behavior
    model.clear_episodic_prompt()
    assert not model.has_episodic_prompt

    with pytest.raises(ValueError, match="Either input_ids or a cached episodic prompt"):
        model(pixel_values=pixel_values, input_ids=None)
