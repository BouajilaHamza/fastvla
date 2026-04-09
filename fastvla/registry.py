"""
FastVLA Model Registry — Maps model names to their architecture configurations.
Enables one-line loading of any supported VLA model.

Usage:
    from fastvla.registry import VLAModelRegistry, register_model

    # Get config for a registered model
    config = VLAModelRegistry.get("openvla-7b")

    # Register a custom model
    register_model("my-vla", MyVLAModelConfig(...))
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class VisionEncoderConfig:
    """Configuration for a vision encoder."""
    model_type: str  # "openvla_fused", "siglip", "vit", "dinov2", "custom"
    model_name: str  # HF model name or timm model name
    num_channels: int  # Input channels (3 for RGB, 6 for fused)
    image_size: int  # Expected input image size
    output_dim: int  # Output feature dimension
    freeze: bool = True  # Whether to freeze during training
    dtype: str = "float16"  # Default dtype

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class LLMConfig:
    """Configuration for the language model backbone."""
    model_type: str  # "llama", "gemma", "qwen", "custom"
    model_name: str  # HF model name
    max_seq_length: int = 2048
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    dtype: str = "float16"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ActionHeadConfig:
    """Configuration for the action head."""
    head_type: str  # "mlp_discrete", "mlp_continuous", "flow_matching", "diffusion"
    action_dim: int = 7  # Output action dimensions
    hidden_dim: int = 256  # Hidden layer size
    num_bins: int = 256  # For discrete action heads
    use_triton: bool = True  # Use Triton kernels when on GPU

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ProjectorConfig:
    """Configuration for vision-to-LLM projection."""
    vision_dim: int  # Input from vision encoder
    llm_dim: int  # Output to LLM
    projector_type: str = "linear"  # "linear", "mlp", "qformer"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class VLAModelConfig:
    """Complete configuration for a VLA model."""
    name: str
    description: str
    vision: VisionEncoderConfig
    llm: LLMConfig
    action_head: ActionHeadConfig
    projector: Optional[ProjectorConfig] = None
    quantization_4bit: bool = True
    gradient_checkpointing: bool = True
    default_batch_size: int = 1
    min_vram_gb: float = 6.0  # Estimated minimum VRAM needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "vision": self.vision.to_dict(),
            "llm": self.llm.to_dict(),
            "action_head": self.action_head.to_dict(),
            "projector": self.projector.to_dict() if self.projector else None,
            "quantization_4bit": self.quantization_4bit,
            "gradient_checkpointing": self.gradient_checkpointing,
            "default_batch_size": self.default_batch_size,
            "min_vram_gb": self.min_vram_gb,
        }


# ── Pre-registered VLA Models ──────────────────────────────────────────

OPENVLA_7B = VLAModelConfig(
    name="openvla-7b",
    description="OpenVLA: 7B VLA with DINOv2+SigLIP fused vision encoder + LLaMA-2-7B",
    vision=VisionEncoderConfig(
        model_type="openvla_fused",
        model_name="openvla/openvla-7b",
        num_channels=6,  # DINOv2(3) + SigLIP(3) fused
        image_size=224,
        output_dim=1024,  # DINOv2-L + SigLIP-SO400M combined
        freeze=False,  # Fine-tune with LoRA
        dtype="float16",
    ),
    llm=LLMConfig(
        model_type="llama",
        model_name="meta-llama/Llama-2-7b-hf",
        max_seq_length=2048,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ),
    action_head=ActionHeadConfig(
        head_type="mlp_discrete",
        action_dim=7,
        hidden_dim=256,
        num_bins=256,
        use_triton=True,
    ),
    projector=ProjectorConfig(
        vision_dim=1024,
        llm_dim=4096,  # LLaMA-2-7B hidden size
        projector_type="linear",
    ),
    quantization_4bit=True,
    gradient_checkpointing=True,
    default_batch_size=1,
    min_vram_gb=6.0,
)

SMOLVLA = VLAModelConfig(
    name="smolvla",
    description="SmolVLA: Lightweight VLA with small vision encoder + SmolLM",
    vision=VisionEncoderConfig(
        model_type="siglip",
        model_name="google/siglip-so400m-patch14-224",
        num_channels=3,
        image_size=224,
        output_dim=1152,
        freeze=True,
        dtype="float16",
    ),
    llm=LLMConfig(
        model_type="llama",
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_seq_length=512,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ),
    action_head=ActionHeadConfig(
        head_type="mlp_continuous",
        action_dim=7,
        hidden_dim=128,
        use_triton=True,
    ),
    projector=ProjectorConfig(
        vision_dim=1152,
        llm_dim=576,  # SmolLM2-135M hidden size
        projector_type="mlp",
    ),
    quantization_4bit=False,
    gradient_checkpointing=True,
    default_batch_size=4,
    min_vram_gb=2.0,
)

PI0_BASE = VLAModelConfig(
    name="pi0-base",
    description="π₀ (Pi-Zero): SigLIP vision encoder + Gemma-2B + flow matching action head",
    vision=VisionEncoderConfig(
        model_type="siglip",
        model_name="google/siglip-so400m-patch14-384",
        num_channels=3,
        image_size=384,
        output_dim=1152,
        freeze=True,
        dtype="float16",
    ),
    llm=LLMConfig(
        model_type="gemma",
        model_name="google/gemma-2-2b",
        max_seq_length=1024,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
    ),
    action_head=ActionHeadConfig(
        head_type="flow_matching",
        action_dim=7,
        hidden_dim=256,
        use_triton=False,  # Flow matching doesn't benefit from Triton
    ),
    projector=ProjectorConfig(
        vision_dim=1152,
        llm_dim=2048,  # Gemma-2B hidden size
        projector_type="linear",
    ),
    quantization_4bit=True,
    gradient_checkpointing=True,
    default_batch_size=1,
    min_vram_gb=4.0,
)


# ── Registry ──────────────────────────────────────────────────────────

class VLAModelRegistry:
    """Registry of available VLA models."""

    _registry: Dict[str, VLAModelConfig] = {
        "openvla-7b": OPENVLA_7B,
        "smolvla": SMOLVLA,
        "pi0-base": PI0_BASE,
    }

    @classmethod
    def get(cls, name: str) -> Optional[VLAModelConfig]:
        """Get model config by name."""
        return cls._registry.get(name)

    @classmethod
    def list_models(cls) -> Dict[str, VLAModelConfig]:
        """List all registered models."""
        return dict(cls._registry)

    @classmethod
    def register(cls, name: str, config: VLAModelConfig):
        """Register a new model configuration."""
        cls._registry[name] = config

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered."""
        return name in cls._registry


def register_model(name: str, config: VLAModelConfig):
    """Convenience function to register a model."""
    VLAModelRegistry.register(name, config)
