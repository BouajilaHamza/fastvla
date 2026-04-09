"""
Vision Encoder Adapters — Unified interface for any vision encoder.

Each adapter wraps a different vision model and exposes:
  .forward(pixel_values) → features  # [B, num_patches, dim]
  .embed_dim  # Output feature dimension
"""
import torch
import torch.nn as nn


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


class OpenVLAFusedVisionAdapter(BaseVisionAdapter):
    """
    OpenVLA's fused DINOv2 + SigLIP vision backbone.
    Input: [B, 6, 224, 224] (DINOv2 3ch + SigLIP 3ch concatenated)
    Output: [B, num_patches, 1024] (concatenated features)

    This wraps the actual OpenVLA model's vision backbone.
    """

    def __init__(self, model, freeze: bool = False):
        """
        Args:
            model: Full OpenVLA model (has .vision_backbone attribute)
            freeze: Whether to freeze the vision encoder
        """
        super().__init__()
        self.vision_backbone = model.vision_backbone
        self._embed_dim = 1024  # DINOv2-L (1024) + SigLIP-SO400M (varies)

        if freeze:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, 224, 224]
        Returns:
            features: [B, num_patches, embed_dim]
        """
        return self.vision_backbone(pixel_values)


class SigLIPVisionAdapter(BaseVisionAdapter):
    """
    Google SigLIP vision encoder adapter.
    Input: [B, 3, H, W]
    Output: [B, num_patches, 1152]
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-224",
                 freeze: bool = True, dtype: torch.dtype = torch.float16):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self._embed_dim = self.model.config.hidden_size
        self.dtype = dtype

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            features: [B, num_patches, embed_dim]
        """
        outputs = self.model(pixel_values=pixel_values.to(self.dtype))
        return outputs.last_hidden_state


class GenericViTVisionAdapter(BaseVisionAdapter):
    """
    Generic ViT adapter for any HuggingFace Vision Transformer.
    Works with: ViT, DINOv2, DeiT, etc.
    Input: [B, 3, H, W]
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224",
                 freeze: bool = True, dtype: torch.dtype = torch.float16):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self._embed_dim = self.model.config.hidden_size
        self.dtype = dtype

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            features: [B, num_patches, embed_dim]
        """
        outputs = self.model(pixel_values=pixel_values.to(self.dtype))
        return outputs.last_hidden_state


# ── Factory ─────────────────────────────────────────────────────────────

def get_vision_adapter(config: dict, model=None) -> BaseVisionAdapter:
    """
    Create a vision adapter from config.

    Args:
        config: VisionEncoderConfig.to_dict() or dict with 'model_type' key
        model: Full model (only needed for 'openvla_fused' type)
    """
    model_type = config.get("model_type", "vit")

    if model_type == "openvla_fused":
        if model is None:
            raise ValueError("model must be provided for openvla_fused adapter")
        return OpenVLAFusedVisionAdapter(
            model,
            freeze=config.get("freeze", False),
        )
    elif model_type == "siglip":
        return SigLIPVisionAdapter(
            model_name=config.get("model_name", "google/siglip-so400m-patch14-224"),
            freeze=config.get("freeze", True),
        )
    elif model_type in ("vit", "dinov2"):
        return GenericViTVisionAdapter(
            model_name=config.get("model_name", "google/vit-base-patch16-224"),
            freeze=config.get("freeze", True),
        )
    else:
        raise ValueError(f"Unknown vision model_type: {model_type}")
