import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
from fastvla import FastVLAModel, FastVLAConfig

class MockVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 128) 
        # config.hidden_size must be a real int
        self.config = MagicMock()
        self.config.hidden_size = 128
    def forward(self, pixel_values=None, **kwargs):
        out = MagicMock()
        out.last_hidden_state = torch.randn(1, 196, 128)
        return out

class MockSigLIPModel(nn.Module):
    """Mocks a composite SigLIP model with text and vision components."""
    def __init__(self):
        super().__init__()
        self.vision_model = MockVisionTower()
        self.text_model = MagicMock(spec=nn.Module)
        self.config = MagicMock()
        self.config.hidden_size = 128
        
        def forward(input_ids=None, pixel_values=None, **kwargs):
            if input_ids is None:
                raise ValueError("You have to specify input_ids")
            return self.vision_model(pixel_values=pixel_values, **kwargs)
        
        self.forward = forward

def test_siglip_surgical_extraction():
    """
    Verify that FastVLAModel correctly extracts the vision component 
    from a composite SigLIP model to avoid the 'input_ids' crash.
    """
    # 1. Setup Mock
    mock_composite = MockSigLIPModel()
    param = nn.Parameter(torch.zeros(1))
    
    with patch("fastvla.model.AutoModel.from_pretrained", return_value=mock_composite), \
         patch("fastvla.model.AutoConfig.from_pretrained"), \
         patch("fastvla.model.AutoModelForCausalLM.from_pretrained") as mock_llm_load, \
         patch("fastvla.model.AutoTokenizer.from_pretrained"):
        
        mock_llm = MagicMock(spec=nn.Module)
        mock_llm.config = MagicMock()
        mock_llm.config.hidden_size = 128
        mock_llm.parameters.side_effect = lambda: iter([param])
        mock_llm.get_input_embeddings = lambda: (lambda x: torch.randn(x.size(0), x.size(1), 128))
        
        # Mock LLM forward call to return hidden states
        class DummyOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states
        
        def mock_llm_forward(*args, **kwargs):
            return DummyOutput([torch.randn(1, 10, 128)])
        
        mock_llm.side_effect = mock_llm_forward
        mock_llm_load.return_value = mock_llm
        
        config = FastVLAConfig(vision_encoder_name="google/siglip-test", dummy=False)
        model = FastVLAModel(config)
        
        # 3. Verify Extraction (now wrapped in an adapter)
        from fastvla.adapters.vision import BaseVisionAdapter
        assert isinstance(model.vision_encoder, BaseVisionAdapter)
        assert model.vision_encoder.model == mock_composite.vision_model
        
        # 4. Verify Forward Pass Safety
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        model(pixel_values=pixel_values, input_ids=input_ids)

def test_peft_wrapped_vision_extraction():
    """Verify extraction works through PEFT/BitsAndBytes wrappers."""
    mock_vision = MockVisionTower()
    param = nn.Parameter(torch.zeros(1))
    
    # Mock a PEFT wrapper structure
    mock_inner = MagicMock(spec=nn.Module)
    mock_inner.vision_tower = mock_vision
    del mock_inner.base_model
    
    mock_base = MagicMock(spec=nn.Module)
    mock_base.model = mock_inner
    del mock_base.base_model
    
    mock_wrapper = MagicMock(spec=nn.Module)
    mock_wrapper.base_model = mock_base
    del mock_wrapper.model
    
    with patch("fastvla.model.AutoModel.from_pretrained", return_value=mock_wrapper), \
         patch("fastvla.model.AutoConfig.from_pretrained"), \
         patch("fastvla.model.AutoModelForCausalLM.from_pretrained") as mock_llm_load, \
         patch("fastvla.model.AutoTokenizer.from_pretrained"):
        
        mock_llm = MagicMock(spec=nn.Module)
        mock_llm.config = MagicMock()
        mock_llm.config.hidden_size = 128
        mock_llm.parameters.side_effect = lambda: iter([param])
        mock_llm_load.return_value = mock_llm
        
        config = FastVLAConfig(vision_encoder_name="custom-peft-vlm", dummy=False)
        model = FastVLAModel(config)
        
        from fastvla.adapters.vision import BaseVisionAdapter
        assert isinstance(model.vision_encoder, BaseVisionAdapter)
        assert model.vision_encoder.model == mock_vision
