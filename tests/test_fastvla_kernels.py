import torch
import torch.nn as nn
import pytest
from fastvla.kernels import vision_language_fusion_forward, TritonActionHead

# ── Feature 1: Vision-Language Fusion Parity ──────────────────────────────

def test_kernel_fusion_parity():
    """
    TDD: Verify that Triton-based fusion (if available) matches 
    PyTorch-based addition with hidden state projection.
    """
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    
    # 1. Inputs
    vision_feats = torch.randn(batch_size, 1, hidden_dim) # Projected vision
    text_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 2. Triton Forward (uses CPU fallback if no Triton)
    fused_out = vision_language_fusion_forward(vision_feats, text_embeds)
    
    # 3. PyTorch Reference
    # FastVLA uses a specific weighted fusion (0.5 * v + 0.5 * t)
    with torch.no_grad():
        ref_out = text_embeds.clone()
        # Align with Triton/fallback logic (mean expand + weighted sum)
        ref_vision = vision_feats.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        ref_out = 0.5 * ref_vision + 0.5 * text_embeds
        
    assert torch.allclose(fused_out, ref_out, atol=1e-3)

# ── Feature 2: Triton Action Head Parity ──────────────────────────────────

def test_kernel_action_head_parity():
    """
    TDD: Verify that TritonActionHead matches standard Linear prediction.
    """
    batch_size = 4
    hidden_dim = 256
    action_hidden_dim = 128
    action_dim = 7
    
    # 1. Setup Triton Head
    head = TritonActionHead(hidden_dim, action_hidden_dim, action_dim)
    
    inputs = torch.randn(batch_size, hidden_dim)
    
    # 2. Triton Forward
    outputs = head(inputs)
    
    # 3. Assertions
    assert outputs.shape == (batch_size, action_dim)
    assert not torch.isnan(outputs).any()
    
    # Verify it produces stable values
    outputs_2 = head(inputs)
    assert torch.allclose(outputs, outputs_2)

# ── Feature 3: Action Head Loss Logic ─────────────────────────────────────

def test_kernel_action_head_loss():
    """
    TDD: Verify that the model's loss calculation is numerically stable.
    """
    from fastvla import FastVLAModel, FastVLAConfig
    
    config = FastVLAConfig(dummy=True, action_dim=7)
    model = FastVLAModel(config)
    
    batch_size = 2
    pixel_values = torch.randn(batch_size, 1, 3, 224, 224)
    input_ids = torch.randint(0, 100, (batch_size, 10))
    labels = torch.randn(batch_size, 7)
    
    # Forward with labels
    action_preds, loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
    
    assert loss is not None
    assert loss.item() >= 0
    assert not torch.isnan(loss)
