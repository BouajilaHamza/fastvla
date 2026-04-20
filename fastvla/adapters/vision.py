"""
Vision Encoder Adapters — Unified interface for any vision encoder.

Each adapter wraps a different vision model and exposes:
  .forward(pixel_values) → features  # [B, num_patches, dim]
  .embed_dim  # Output feature dimension
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Dict

logger = logging.getLogger(__name__)


class BaseVisionAdapter(nn.Module):
    """Base class for all vision encoder adapters."""

    def __init__(self):
        super().__init__()
        self._embed_dim: int = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_id: str, device_map: Union[str, Dict] = "auto", 
                        load_in_4bit: bool = False, hf_token: Optional[str] = None, 
                        **kwargs) -> "BaseVisionAdapter":
        """Load and extract the vision component from a model ID."""
        raise NotImplementedError

    @staticmethod
    def _extract_vision_encoder(model: nn.Module) -> nn.Module:
        """Surgical extraction of the vision encoder from composite models."""
        current = model
        # 1. Drill down through wrappers (PEFT, BitsAndBytes)
        for _ in range(5):
            if hasattr(current, "base_model") and current.base_model != current:
                current = current.base_model
            elif hasattr(current, "model") and current.model != current and not hasattr(current, "vision_tower"):
                current = current.model
            else:
                break

        # 2. Search for common vision attribute names
        def _find_vision_sub(obj, depth=0):
            if depth > 3: return None
            # Prioritize standard VLA attribute names
            for attr in ["vision_tower", "vision_model", "visual", "vision_backbone"]:
                if hasattr(obj, attr):
                    sub = getattr(obj, attr)
                    # Handle cases where vision_tower is another wrapper
                    if attr == "vision_tower" and hasattr(sub, "vision_tower"):
                        return sub.vision_tower
                    return sub
            # Recursive search in 'model' or 'vision' attributes
            for sub_attr in ["model", "vision"]:
                if hasattr(obj, sub_attr):
                    val = getattr(obj, sub_attr)
                    if val != obj and isinstance(val, nn.Module):
                        res = _find_vision_sub(val, depth + 1)
                        if res: return res
            return None

        sub = _find_vision_sub(current)
        if sub is not None:
            logger.info(f"Surgically extracted {sub.__class__.__name__} from {current.__class__.__name__}.")
            return sub
        return current

    @staticmethod
    def _get_bnb_config():
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )


class OpenVLAFusedVisionAdapter(BaseVisionAdapter):
    """
    OpenVLA's fused DINOv2 + SigLIP vision backbone.
    Surgically extracts the vision components and drops the LLM.
    """

    def __init__(self, vision_backbone: nn.Module, embed_dim: int = 1024):
        super().__init__()
        self.vision_backbone = vision_backbone
        self._embed_dim = embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_backbone(pixel_values)

    @classmethod
    def from_pretrained(cls, model_id: str, device_map: Union[str, Dict] = "auto", 
                        load_in_4bit: bool = False, hf_token: Optional[str] = None, 
                        **kwargs) -> "OpenVLAFusedVisionAdapter":
        from transformers import AutoModel
        logger.info(f"Loading OpenVLA model {model_id} for vision extraction...")
        
        quant_config = cls._get_bnb_config() if load_in_4bit else None
        
        try:
            full_model = AutoModel.from_pretrained(
                model_id, 
                device_map=device_map, 
                quantization_config=quant_config,
                token=hf_token, 
                trust_remote_code=True
            )
            vision_backbone = cls._extract_vision_encoder(full_model)
            
            # Attempt to free LLM memory
            if hasattr(full_model, "language_model"):
                del full_model.language_model
            
            return cls(vision_backbone)
        except Exception as e:
            # Recovery Path: If AutoModel fails due to Config mismatch (common with OpenVLA)
            logger.warning(f"OpenVLA extraction failed: {e}. Falling back to SigLIP so400m...")
            return SigLIPVisionAdapter.from_pretrained(
                "google/siglip-so400m-patch14-384", 
                device_map=device_map, 
                load_in_4bit=load_in_4bit, 
                hf_token=hf_token
            )


class OlmoVLAVisionAdapter(BaseVisionAdapter):
    """
    OlmoVLA's vision backbone (usually CLIP/SigLIP based).
    """

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        self.vision_model = vision_model
        # OlmoVLA hidden size is usually from the vision tower
        self._embed_dim = getattr(vision_model.config, "hidden_size", 1024)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values)
        return outputs.last_hidden_state

    @classmethod
    def from_pretrained(cls, model_id: str, device_map: Union[str, Dict] = "auto", 
                        load_in_4bit: bool = False, hf_token: Optional[str] = None, 
                        **kwargs) -> "OlmoVLAVisionAdapter":
        from transformers import AutoModel
        logger.info(f"Loading OlmoVLA model {model_id} for vision extraction...")
        
        quant_config = cls._get_bnb_config() if load_in_4bit else None
        full_model = AutoModel.from_pretrained(
            model_id, 
            device_map=device_map, 
            quantization_config=quant_config,
            token=hf_token, 
            trust_remote_code=True
        )
        vision_model = cls._extract_vision_encoder(full_model)
        return cls(vision_model)


class SigLIPVisionAdapter(BaseVisionAdapter):
    """
    Google SigLIP vision encoder adapter.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._embed_dim = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state

    @classmethod
    def from_pretrained(cls, model_id: str, device_map: Union[str, Dict] = "auto", 
                        load_in_4bit: bool = False, hf_token: Optional[str] = None, 
                        **kwargs) -> "SigLIPVisionAdapter":
        from transformers import AutoModel
        
        quant_config = cls._get_bnb_config() if load_in_4bit else None
        model = AutoModel.from_pretrained(
            model_id, 
            device_map=device_map, 
            quantization_config=quant_config,
            token=hf_token, 
            trust_remote_code=True
        )
        vision_model = cls._extract_vision_encoder(model)
        return cls(vision_model)


class GenericViTVisionAdapter(BaseVisionAdapter):
    """
    Generic ViT adapter for any HuggingFace Vision Transformer.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._embed_dim = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state

    @classmethod
    def from_pretrained(cls, model_id: str, device_map: Union[str, Dict] = "auto", 
                        load_in_4bit: bool = False, hf_token: Optional[str] = None, 
                        **kwargs) -> "GenericViTVisionAdapter":
        from transformers import AutoModel
        
        quant_config = cls._get_bnb_config() if load_in_4bit else None
        model = AutoModel.from_pretrained(
            model_id, 
            device_map=device_map, 
            quantization_config=quant_config,
            token=hf_token, 
            trust_remote_code=True
        )
        vision_model = cls._extract_vision_encoder(model)
        return cls(vision_model)


# ── Factory ─────────────────────────────────────────────────────────────

def get_vision_adapter(config_dict: dict, device_map: Union[str, Dict] = "auto", 
                       hf_token: Optional[str] = None) -> BaseVisionAdapter:
    """
    Create a vision adapter from config.
    """
    model_type = config_dict.get("model_type", "vit")
    model_id = config_dict.get("model_name")
    load_in_4bit = config_dict.get("load_in_4bit", False)

    # Force OpenVLA specific logic if 'openvla' is in the name
    if "openvla" in model_id.lower() or model_type == "openvla_fused":
        return OpenVLAFusedVisionAdapter.from_pretrained(
            model_id, device_map=device_map, load_in_4bit=load_in_4bit, hf_token=hf_token
        )
    elif model_type == "olmovla":
        return OlmoVLAVisionAdapter.from_pretrained(
            model_id, device_map=device_map, load_in_4bit=load_in_4bit, hf_token=hf_token
        )
    elif model_type == "siglip":
        return SigLIPVisionAdapter.from_pretrained(
            model_id, device_map=device_map, load_in_4bit=load_in_4bit, hf_token=hf_token
        )
    elif model_type in ("vit", "dinov2"):
        return GenericViTVisionAdapter.from_pretrained(
            model_id, device_map=device_map, load_in_4bit=load_in_4bit, hf_token=hf_token
        )
    else:
        raise ValueError(f"Unknown vision model_type: {model_type}")
