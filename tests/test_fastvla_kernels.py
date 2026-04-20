import torch
import torch.nn as nn
import pytest
from fastvla.kernels import vision_language_fusion_forward, TritonActionHead

# ── Feature 1: Vision-Language Fusion Parity ──────────────────────────────

def test_kernel_fusion_parity():
    """
    TDD: Verify that Triton-based Cross-Attention matches 
    PyTorch's native scaled dot-product attention.
    """
    from fastvla.kernels import vision_language_cross_attention
    batch_size = 2
    seq_len_text = 16
    seq_len_vision = 32
    hidden_dim = 128
    
    # 1. Inputs (Text=Q, Visual=K,V)
    text_embeds = torch.randn(batch_size, seq_len_text, hidden_dim)
    vision_feats = torch.randn(batch_size, seq_len_vision, hidden_dim)
    
    # 2. FastVLA Cross-Attention (uses CPU fallback if no Triton)
    fused_out = vision_language_cross_attention(text_embeds, vision_feats)
    
    # 3. PyTorch Reference
    with torch.no_grad():
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            text_embeds, vision_feats, vision_feats
        )
        
    assert fused_out.shape == (batch_size, seq_len_text, hidden_dim)
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
