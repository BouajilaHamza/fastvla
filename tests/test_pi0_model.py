"""
Unit tests for Physical Intelligence π₀ / openpi single-GPU fine-tuning recipe in FastVLA.
Tests registry presets, PaliGemma-3B backbone configuration, flow-matching continuous
action expert coupling, and 2-step Midpoint ODE action prediction.
"""

import pytest
import torch
from fastvla.registry import VLAModelRegistry
from fastvla.model import FastVLAModel
from fastvla.config import FastVLAConfig
from fastvla.adapters.action_head import FlowMatchingActionHead


def test_pi0_registry_presets():
    """Verify π₀, openpi, and π₀-FAST model presets in the registry."""
    # 1. Base π₀ (Continuous Flow-Matching)
    pi0_cfg = VLAModelRegistry.get("pi0")
    assert pi0_cfg is not None
    assert pi0_cfg.name == "pi0-base"
    assert pi0_cfg.vision.model_type == "siglip"
    assert pi0_cfg.vision.image_size == 224
    assert pi0_cfg.vision.output_dim == 1152
    assert pi0_cfg.llm.model_type == "gemma"
    assert pi0_cfg.llm.model_name == "google/paligemma-3b-pt-224"
    assert "q_proj" in pi0_cfg.llm.lora_target_modules
    assert "v_proj" in pi0_cfg.llm.lora_target_modules
    assert pi0_cfg.action_head.head_type == "flow_matching"
    assert pi0_cfg.action_head.solver == "midpoint"

    # 2. Aliases check
    assert VLAModelRegistry.get("openpi") == pi0_cfg
    assert VLAModelRegistry.get("openpi/pi0-base") == pi0_cfg
    assert VLAModelRegistry.get("pi0-base") == pi0_cfg

    # 3. π₀-FAST (Autoregressive Discrete)
    pi0_fast_cfg = VLAModelRegistry.get("pi0-fast")
    assert pi0_fast_cfg is not None
    assert pi0_fast_cfg.action_head.head_type == "mlp_discrete"
    assert pi0_fast_cfg.action_head.num_bins == 256


def test_pi0_flow_matching_architecture():
    """Verify FastVLAModel integrates FlowMatchingActionHead for π₀."""
    batch_size = 2
    chunk_size = 16
    action_dof = 7
    total_action_dim = chunk_size * action_dof  # 112 continuous DoF

    config = FastVLAConfig(
        dummy=True,
        llm_name="google/paligemma-3b-pt-224",
        vision_encoder_name="google/paligemma-3b-pt-224",
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        num_attention_heads=4,
        vocab_size=500,
        action_dim=action_dof,
        chunk_size=chunk_size,
        head_type="flow_matching",
        action_type="continuous",
        action_hidden_dim=32,
    )

    model = FastVLAModel(config)
    assert isinstance(model.action_head, FlowMatchingActionHead)
    assert model.action_head.action_dim == total_action_dim

    # Verify forward and loss computation
    pixel_values = torch.randn(batch_size, 1, 3, 32, 32)
    input_ids = torch.randint(0, 500, (batch_size, 8))
    attention_mask = torch.ones(batch_size, 8)
    labels = torch.randn(batch_size, total_action_dim)

    action_preds, loss = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert action_preds.shape == (batch_size, total_action_dim)
    assert loss is not None
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

    # Verify backward pass computes gradients
    loss.backward()
    has_grads = any(p.grad is not None and p.grad.norm().item() > 0 for p in model.parameters())
    assert has_grads


def test_pi0_midpoint_ode_inference():
    """Verify 2-step Midpoint ODE rollout on π₀ flow matching action head."""
    batch_size = 2
    hidden_dim = 64
    action_dim = 32  # 16 steps x 2 DoF

    head = FlowMatchingActionHead(
        input_dim=hidden_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_inference_steps=2,
        solver="midpoint",
    )

    pooled_hidden = torch.randn(batch_size, hidden_dim)
    predicted_actions = head.predict(pooled_hidden, steps=2)

    assert predicted_actions.shape == (batch_size, action_dim)
    assert not torch.isnan(predicted_actions).any()
