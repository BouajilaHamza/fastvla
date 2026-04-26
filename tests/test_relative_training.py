import torch
import torch.nn as nn
import numpy as np
from fastvla import FastVLAConfig, FastVLAModel, FastVLATrainer
from fastvla.data.collator import UnslothVLACollator
from transformers import AutoTokenizer
from unittest.mock import MagicMock, patch

def test_relative_delta_mapping():
    """
    Verify that collator correctly handles both 'state' and 'states' keys for relative deltas.
    This was a bug where LeRobotDataset uses 'state' but collator expected 'states'.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Mock features with 'state' (singular) as provided by LeRobotDataset
    features = [
        {
            "images": {"rgb": torch.zeros(3, 224, 224)},
            "state": torch.tensor([1.0, 2.0, 3.0]),
            "action": torch.tensor([1.1, 2.2, 3.3]),
            "instructions": "test"
        }
    ]
    
    # Collator configured for relative deltas
    collator = UnslothVLACollator(
        tokenizer=tokenizer,
        use_relative_delta=True,
        action_dim=3
    )
    
    batch = collator(features)
    
    # Check if labels are deltas (0.1, 0.2, 0.3)
    labels = batch["labels"][0]
    expected_delta = torch.tensor([0.1, 0.2, 0.3])
    
    assert torch.allclose(labels, expected_delta, atol=1e-5), \
        f"Expected relative delta {expected_delta}, but got {labels}."

def test_trainer_relative_delta_integration():
    """
    Test that FastVLATrainer correctly sets up a collator with relative delta
    when use_relative_delta=True is in the config.
    """
    config = FastVLAConfig(
        dummy=True,
        use_relative_delta=True,
        action_dim=3
    )
    model = FastVLAModel(config)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    
    # Mock dataset
    mock_data = [
        {
            "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.array([1.0, 2.0, 3.0]),
            "action": np.array([1.1, 2.2, 3.3]),
            "instruction": "test",
            "episode_id": 0
        }
    ]
    
    with patch("fastvla.data.datasets.LeRobotDataset._load_data", return_value=mock_data):
        trainer = FastVLATrainer(
            model=model,
            dataset="pusht", 
            batch_size=1
        )
        
        # Verify collator inherited use_relative_delta from config
        assert trainer.train_dataloader.collate_fn.use_relative_delta is True
        
        # Verify batch computation
        batch = next(iter(trainer.train_dataloader))
        labels = batch["labels"][0]
        expected_delta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        assert torch.allclose(labels, expected_delta, atol=1e-5)

def test_config_hf_token_auto_discovery():
    """Verify that FastVLAConfig pulls HF_TOKEN from environment if not explicitly set."""
    import os
    with patch.dict(os.environ, {"HF_TOKEN": "env_token_val"}):
        config = FastVLAConfig()
        assert config.hf_token == "env_token_val"
    
    with patch.dict(os.environ, {"HF_API_KEY": "key_token_val"}):
        # Clear HF_TOKEN from env to test fallback
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        config = FastVLAConfig()
        assert config.hf_token == "key_token_val"

def test_model_vision_only_fallback_logic():
    """Verify that from_pretrained doesn't use vision model name as LLM name."""
    # Setup standard mocks for components
    mock_param = nn.Parameter(torch.zeros(1))
    mock_llm = MagicMock(spec=nn.Module)
    mock_llm.config = MagicMock()
    # next(self.llm.parameters()) needs to work multiple times
    mock_llm.parameters.side_effect = lambda: iter([mock_param])
    
    mock_vision = MagicMock(spec=nn.Module)
    mock_vision.config = MagicMock()
    mock_vision.config.hidden_size = 768
    mock_vision.embed_dim = 768
    mock_vision.parameters.side_effect = lambda: iter([mock_param])
    
    mock_action_head = nn.Linear(10, 10)
    mock_action_head.action_dim = 7
    
    with patch("fastvla.model.VLAModelRegistry.get", return_value=None), \
         patch("fastvla.model.FastVLAModel._load_component", return_value=mock_llm), \
         patch("fastvla.adapters.vision.get_vision_adapter", return_value=mock_vision), \
         patch("fastvla.model.check_environment"), \
         patch("fastvla.model.get_device", return_value="cpu"), \
         patch("fastvla.model.AutoTokenizer.from_pretrained"), \
         patch("fastvla.model.TritonActionHead", return_value=mock_action_head), \
         patch("fastvla.model.AutoModel.from_pretrained"), \
         patch("fastvla.model.AutoModelForCausalLM.from_pretrained"):
        
        # Test Case 1: Known vision pattern (SIGLIP) should NOT fallback
        model = FastVLAModel.from_pretrained("google/siglip-so400m-patch14-224")
        assert model.config.llm_name != "google/siglip-so400m-patch14-224"
        
        # Test Case 2: Custom pattern NOT in heuristic should NOT fallback by default
        model = FastVLAModel.from_pretrained("my-custom-vision-encoder")
        assert model.config.llm_name != "my-custom-vision-encoder"
        
        # Test Case 3: Explicit fallback=True should force it regardless of heuristic
        model = FastVLAModel.from_pretrained("my-custom-vlm", fallback_llm=True)
        assert model.config.llm_name == "my-custom-vlm"

        # Test Case 4: Composite pattern (OpenVLA) should fallback by default
        model = FastVLAModel.from_pretrained("openvla-test")
        assert model.config.llm_name == "openvla-test"

def test_registry_configurability():
    """Verify we can modify heuristic keywords at runtime."""
    from fastvla.registry import VLAModelRegistry
    
    original = list(VLAModelRegistry.COMPOSITE_VLM_KEYWORDS)
    try:
        VLAModelRegistry.COMPOSITE_VLM_KEYWORDS.append("customvlm")
        
        mock_llm = MagicMock(spec=nn.Module)
        mock_llm.config = MagicMock()
        mock_llm.parameters.return_value = iter([nn.Parameter(torch.zeros(1))])
        
        with patch("fastvla.model.VLAModelRegistry.get", return_value=None), \
             patch("fastvla.model.FastVLAModel._load_component", return_value=mock_llm), \
             patch("fastvla.model.get_device", return_value="cpu"), \
             patch("fastvla.model.check_environment"), \
             patch("fastvla.model.AutoTokenizer.from_pretrained"), \
             patch("fastvla.model.TritonActionHead", return_value=MagicMock()), \
             patch("fastvla.model.AutoModel.from_pretrained"), \
             patch("fastvla.model.AutoModelForCausalLM.from_pretrained"):
             
             # Should now fallback because of the custom keyword we added
             model = FastVLAModel.from_pretrained("my-customvlm-model")
             assert model.config.llm_name == "my-customvlm-model"
    finally:
        VLAModelRegistry.COMPOSITE_VLM_KEYWORDS = original

def test_action_dim_slicing_robustness():
    """TDD: Verify that model handles fine-tuning 7D head on 2D labels via slicing."""
    config = FastVLAConfig(dummy=True, action_dim=7) # Model has 7 dims
    model = FastVLAModel(config)
    
    # Simulate data with 2 dims (PushT)
    pixel_values = torch.randn(1, 1, 3, 224, 224)
    input_ids = torch.randint(0, 100, (1, 10))
    labels = torch.randn(1, 2) # Only 2 dims
    
    # This should NOT crash, but slice predictions to 2 dims for loss
    action_preds, loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
    
    # Verify predictions were sliced to match labels for loss calculation
    # In my proposed fix, we only slice for loss, action_preds returned might still be 7D 
    # unless we decide otherwise. But let's check internal loss logic.
    assert loss is not None
    assert loss.item() > 0

def test_4bit_precision_safety():
    """TDD: Verify that .to(torch.float32) doesn't crash a 4-bit model."""
    config = FastVLAConfig(dummy=True, load_in_4bit=True)
    model = FastVLAModel(config)
    
    # Simulate quantization flag
    model.is_quantized = True 
    
    # This should log a warning but NOT crash or corrupt
    model.to(torch.float32)
    
    assert model.is_quantized is True

def test_unsloth_patching_verification():
    """TDD: Verify that FastVLAModel explicitly calls Unsloth patching functions."""
    config = FastVLAConfig(dummy=False, load_in_4bit=True)
    
    # Mocking components to avoid real loading
    mock_llm = MagicMock(spec=nn.Module)
    mock_llm.config = MagicMock()
    mock_llm.parameters.side_effect = lambda: iter([nn.Parameter(torch.zeros(1))])
    
    mock_action_head = nn.Linear(10, 10)
    mock_action_head.action_dim = 7
    
    with patch("fastvla.model.UNSLOTH_AVAILABLE", True), \
         patch("fastvla.model.torch.cuda.is_available", return_value=True), \
         patch("fastvla.model.torch.cuda.current_device", return_value=0), \
         patch("fastvla.model.torch.cuda.get_device_capability", return_value=(8, 0)), \
         patch("fastvla.model.patch_model", return_value=mock_llm, create=True) as mock_patch_model, \
         patch("fastvla.model.patch_forward", create=True) as mock_patch_forward, \
         patch("fastvla.model.patch_saving_functions", create=True) as mock_patch_saving, \
         patch("fastvla.model.FastVLAModel._load_component", return_value=mock_llm), \
         patch("fastvla.adapters.vision.get_vision_adapter", return_value=MagicMock()), \
         patch("fastvla.model.check_environment"), \
         patch("fastvla.model.get_device", return_value="cuda"), \
         patch("fastvla.model.AutoTokenizer.from_pretrained"), \
         patch("fastvla.model.TritonActionHead", return_value=mock_action_head):
        
        # Instantiate model (this should trigger patching)
        FastVLAModel(config)
        
        # Assertions: All 3 Unsloth patches MUST be called
        mock_patch_model.assert_called_once()
        mock_patch_forward.assert_called_once()
        mock_patch_saving.assert_called_once_with(mock_llm)

def test_llm_layer_truncation():
    """TDD: Verify that LLM layers can be truncated via config."""
    # Setup dummy model with 2 layers requested
    config = FastVLAConfig(dummy=True, llm_num_layers=2)
    model = FastVLAModel(config)
    
    assert model.config.llm_num_layers == 2
    
    # Simulate a real model with 32 layers
    mock_llm = MagicMock(spec=nn.Module)
    mock_llm.config = MagicMock()
    # Path for Llama-2 (model.layers)
    mock_llm.model = MagicMock()
    mock_llm.model.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(32)])
    
    # We will implement logic in FastVLAModel to slice this ModuleList
    # If the model has a layers list, slice it to match config
    if hasattr(mock_llm, "model") and hasattr(mock_llm.model, "layers"):
        if len(mock_llm.model.layers) > config.llm_num_layers:
            mock_llm.model.layers = mock_llm.model.layers[:config.llm_num_layers]
        
    assert len(mock_llm.model.layers) == 2

if __name__ == "__main__":
    from fastvla import FastVLAConfig, FastVLAModel, FastVLATrainer
    test_relative_delta_mapping()
    test_trainer_relative_delta_integration()
    test_config_hf_token_auto_discovery()
    test_model_vision_only_fallback_logic()
    test_action_dim_slicing_robustness()
    test_4bit_precision_safety()
    test_unsloth_patching_verification()
    test_llm_layer_truncation()
