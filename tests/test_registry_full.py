import pytest
from fastvla import FastVLAModel
from fastvla.registry import VLAModelRegistry


def test_registry_loading_smolvla():
    """Test that loading 'smolvla' pulls all correct fields from the registry."""
    # We use dummy=True to avoid actual heavy model loading
    model = FastVLAModel.from_pretrained("smolvla", dummy=True)
    config = model.config
    reg_config = VLAModelRegistry.get("smolvla")

    assert config.vision_encoder_name == reg_config.vision.model_name
    assert config.llm_name == reg_config.llm.model_name
    assert config.action_dim == reg_config.action_head.action_dim
    assert config.vision_hidden_size == reg_config.vision.output_dim
    assert config.lora_rank == reg_config.llm.lora_rank
    assert config.image_size == reg_config.vision.image_size


def test_registry_loading_openvla():
    """Test that loading 'openvla-7b' pulls all correct fields."""
    model = FastVLAModel.from_pretrained("openvla-7b", dummy=True)
    config = model.config
    reg_config = VLAModelRegistry.get("openvla-7b")

    assert config.vision_encoder_name == reg_config.vision.model_name
    assert config.action_dim == reg_config.action_head.action_dim
    assert config.load_in_4bit == reg_config.quantization_4bit
    assert config.image_size == reg_config.vision.image_size


def test_registry_overrides():
    """Test that manual kwargs override registry defaults."""
    # Override action_dim from 7 to 2
    model = FastVLAModel.from_pretrained("smolvla", dummy=True, action_dim=2)
    assert model.config.action_dim == 2

    # Override lora_rank
    model = FastVLAModel.from_pretrained("smolvla", dummy=True, lora_rank=32)
    assert model.config.lora_rank == 32


def test_unknown_model_fallback():
    """Test that an unknown model name falls back to using it as a HF path."""
    model_name = "facebook/opt-125m"
    model = FastVLAModel.from_pretrained(model_name, dummy=True)
    assert model.config.vision_encoder_name == model_name
    assert model.config.llm_name == model_name


if __name__ == "__main__":
    pytest.main([__file__])
