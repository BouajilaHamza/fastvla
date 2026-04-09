import torch
import torch.nn as nn
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastvla import FastVLAModel, FastVLAConfig, FastVLATrainer

# ── Feature 1: Checkpoint Bit-Perfection ──────────────────────────────────

def test_trainer_checkpoint_bit_perfection(tmp_path):
    """
    TDD: Verify that saving and loading a checkpoint restores model weights perfectly.
    This test is expected to FAIL because load_checkpoint is not yet implemented.
    """
    # 1. Setup
    config = FastVLAConfig(dummy=True, llm_hidden_size=64, vision_hidden_size=64)
    model = FastVLAModel(config)
    
    # Randomize a specific weight to track
    with torch.no_grad():
        model.vision_proj.weight.fill_(42.0)
    
    trainer = FastVLATrainer(
        model=model,
        train_dataset=MagicMock(__len__=lambda x: 1, __getitem__=lambda x, i: {"images": [torch.zeros(3, 224, 224)], "actions": [0.0]*7}),
        output_dir=str(tmp_path),
        use_mixed_precision=False
    )
    
    # 2. Save
    trainer.save_checkpoint(step=100)
    checkpoint_path = tmp_path / "checkpoint-100"
    assert checkpoint_path.exists()
    
    # 3. Modify original weights
    with torch.no_grad():
        model.vision_proj.weight.fill_(0.0)
    print(f"Weight before load: {model.vision_proj.weight[0,0].item()}")
    
    # 4. Load
    trainer.load_checkpoint(checkpoint_path)
    print(f"Weight after load: {model.vision_proj.weight[0,0].item()}")
        
    # 5. Verify bit-perfection
    assert torch.allclose(model.vision_proj.weight, torch.tensor(42.0), atol=1e-5)

# ── Feature 2: LoRA Trainable Parameters ──────────────────────────────────

def test_trainer_lora_trainable_parameters():
    """
    TDD: Verify that only LoRA parameters are trainable when use_peft=True.
    """
    config = FastVLAConfig(dummy=True, use_peft=True, lora_rank=8)
    # Note: FastVLAModel dummy currently doesn't implement PEFT wrapping in init for dummies
    # This test will highlight that gap.
    model = FastVLAModel(config)
    
    # Check if any parameter has requires_grad=True and is NOT in a lora layer
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    
    # In a proper PEFT setup, most weights should be frozen
    # This might fail if the dummy model isn't correctly freezing base weights
    non_lora_trainable = [p for p in trainable_params if "lora_" not in p and "action_head" not in p and "vision_proj" not in p]
    
    assert len(non_lora_trainable) == 0, f"Found non-LoRA trainable parameters: {non_lora_trainable}"
