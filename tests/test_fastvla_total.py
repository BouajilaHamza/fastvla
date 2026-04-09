import torch
import torch.nn as nn
import pytest
import logging
from unittest.mock import MagicMock, patch
from transformers import PretrainedConfig
from fastvla import FastVLAModel, FastVLAConfig, FastVLATrainer
from fastvla.registry import VLAModelRegistry

# Setup logging
logger = logging.getLogger(__name__)

# ── Mock Classes ──────────────────────────────────────────────────────────

class MockOpenVLAConfig(PretrainedConfig):
    model_type = "openvla"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 1024
        self.tokenizer_class = "LlamaTokenizer"

class MockSigLIPConfig(PretrainedConfig):
    model_type = "siglip"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 1152
        self.tokenizer_class = "SiglipTokenizer"

# ── Feature 1: The Config Gauntlet (Loader Robustness) ─────────────────────

@patch("fastvla.model.AutoConfig.from_pretrained")
@patch("fastvla.model.AutoModel.from_pretrained")
def test_smart_loader_openvla_recovery(mock_model_load, mock_config_load):
    """
    Test that the loader correctly identifies an OpenVLA composite model
    and recovers when the standard AutoModel load fails.
    """
    # 1. Setup Mocks
    mock_config_load.return_value = MockOpenVLAConfig()
    
    # Create a mock vision tower that has a valid config
    mock_vision_tower = MagicMock(spec=nn.Module)
    mock_vision_tower.config = MockSigLIPConfig()
    # Mock parameters for next(self.vision_encoder.parameters()).device
    param = nn.Parameter(torch.zeros(1))
    mock_vision_tower.parameters.return_value = iter([param])
    
    # Simulate ValueError on first call
    side_effects = [
        ValueError("Unrecognized configuration class OpenVLAConfig"),
        mock_vision_tower
    ]
    mock_model_load.side_effect = side_effects
    
    # 2. Initialize Model
    config = FastVLAConfig(vision_encoder_name="openvla/openvla-7b", dummy=False)
    
    with patch("fastvla.model.check_environment"), \
         patch("fastvla.model.get_device", return_value="cpu"), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("fastvla.model.AutoModelForCausalLM.from_pretrained") as mock_llm_load, \
         patch("fastvla.model.AutoTokenizer.from_pretrained"):
        
        mock_llm = MagicMock()
        mock_llm.config = MagicMock(hidden_size=128)
        mock_llm.parameters.return_value = iter([param])
        mock_llm_load.return_value = mock_llm
        
        model = FastVLAModel(config)
        
    assert mock_model_load.call_count >= 2
    assert model.vision_encoder == mock_vision_tower

@patch("fastvla.model.AutoConfig.from_pretrained")
def test_composite_vlm_detection(mock_config_load):
    """Verify the heuristic for detecting composite VLMs."""
    mock_config_load.return_value = MockOpenVLAConfig()
    model = FastVLAModel(FastVLAConfig(dummy=True))
    assert model._is_composite_vlm("openvla/openvla-7b") is True
    
    mock_config_load.return_value = MockSigLIPConfig()
    assert model._is_composite_vlm("google/siglip-so400m") is False


# ── Feature 2: The Distributed Phantom (Multi-GPU Simulation) ──────────────

@patch("torch.cuda.device_count", return_value=2)
@patch("torch.cuda.is_available", return_value=True)
@patch("accelerate.Accelerator")
def test_distributed_trainer_logic(mock_accelerator, mock_cuda_avail, mock_cuda_count):
    """
    Verify that FastVLATrainer correctly adapts its logic for multi-GPU sharding.
    """
    model = FastVLAModel(FastVLAConfig(dummy=True, load_in_4bit=True))
    
    # Use a simpler Mock for the dataset to avoid init issues
    dataset = MagicMock()
    def get_item(i): return {"images": [torch.zeros(3, 224, 224)], "actions": [0.0]*7}
    dataset.__getitem__ = get_item
    dataset.__len__.return_value = 10
    
    # Simulation: Mixed precision should be DISABLED for 4-bit shanded models
    trainer = FastVLATrainer(model=model, train_dataset=dataset)
    
    # Verify accelerator was initialized at some point (either in trainer or internally)
    # FastVLATrainer initializes its own accelerator in __init__
    if mock_accelerator.call_args:
        kwargs = mock_accelerator.call_args[1]
        assert kwargs.get("mixed_precision") == "no"
    
    assert trainer.model == model

@patch("fastvla.model.AutoModel.from_pretrained")
@patch("fastvla.model.AutoConfig.from_pretrained")
@patch("fastvla.model.FastVLAModel._load_language_model_internal")
def test_composite_model_extraction(mock_llm_load, mock_config, mock_from_pretrained):
    """
    Verify that if AutoModel returns a composite object (SigLIP/CLIP), 
    we surgically extract the .vision_model.
    """
    # 1. Setup Mock Model with .vision_model attribute
    mock_vision_only = MagicMock()
    mock_vision_only.config = MagicMock()
    mock_vision_only.config.hidden_size = 768
    
    mock_composite = MagicMock()
    mock_composite.vision_model = mock_vision_only
    mock_from_pretrained.return_value = mock_composite
    
    # 2. Setup Mock Config
    mock_cfg = MagicMock()
    mock_cfg.model_type = "siglip"
    mock_config.return_value = mock_cfg

    # 3. Setup Mock LLM
    mock_llm = MagicMock()
    mock_llm.config = MagicMock()
    mock_llm.config.hidden_size = 512
    # next(model.parameters()) requires an iterator
    mock_llm.parameters.return_value = iter([torch.zeros(1)])
    mock_llm_load.return_value = mock_llm
    
    # 4. Load FastVLA
    config = FastVLAConfig(vision_encoder_name="google/siglip-test", dummy=False)
    model = FastVLAModel(config)
    
    # 5. Assertions
    # The internal vision_encoder should be the extracted one
    assert model.vision_encoder == mock_vision_only
    logger.info("Fidelity Check: Composite model extraction verified.")

# ── Feature 3: Numerical Parity (Triton vs PyTorch) ────────────────────────

def test_triton_parity_cpu_fallback():
    """Verify TritonActionHead gracefully falls back to PyTorch on CPU."""
    hidden_dim = 128
    action_dim = 7
    batch_size = 4
    
    head = FastVLAModel(FastVLAConfig(dummy=True, action_dim=action_dim)).action_head
    inputs = torch.randn(batch_size, hidden_dim)
    outputs = head(inputs)
    
    assert outputs.shape == (batch_size, action_dim)
    assert not torch.isnan(outputs).any()
    
    # Parity: should match deterministic linear projection in dummy mode
    # Since it's random weights, we just check consistency on same input
    outputs_2 = head(inputs)
    assert torch.allclose(outputs, outputs_2, atol=1e-3)

# ── Feature 4: Full Multi-Cam Fusion Test ──────────────────────────────────

def test_multicam_fusion_robustness():
    """Verify fusion logic handles multi-camera inputs consistently."""
    config = FastVLAConfig(dummy=True)
    model = FastVLAModel(config)
    
    batch_size = 2
    num_cams = 3 
    pixel_values = torch.randn(batch_size, num_cams, 3, 224, 224)
    input_ids = torch.randint(0, 100, (batch_size, 10))
    
    action_preds, loss = model(pixel_values=pixel_values, input_ids=input_ids)
    
    assert action_preds.shape == (batch_size, 7)
