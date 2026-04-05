"""
Optimization utilities for FastVLA, following Unsloth patterns.
Includes quantization-aware training, 8-bit optimizers, and memory optimizations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from peft import LoraConfig, TaskType

# Optional: bitsandbytes (GPU-only, not available on CPU)
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb

    BNB_AVAILABLE = True
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # type: ignore


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[Any]:
    """
    Create BitsAndBytes quantization configuration.
    Returns None if bitsandbytes is not installed (CPU environments).
    """
    if not load_in_4bit:
        return None
    if not BNB_AVAILABLE:
        return None

    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype_map.get(
            bnb_4bit_compute_dtype, torch.float16
        ),
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def get_8bit_optimizer(model: nn.Module, learning_rate: float = 5e-5):
    """
    Create optimizer. Uses 8-bit AdamW from bitsandbytes if available,
    otherwise falls back to standard AdamW.
    """
    param_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": learning_rate,
        }
    ]

    if BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    return optimizer


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing for memory efficiency.

    Args:
        model: The model to enable gradient checkpointing for
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    # Recursively enable for submodules
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()


def setup_mixed_precision_training():
    """
    Setup mixed precision training with autocast.

    Returns:
        Autocast context manager
    """
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)


class ActivationOffloader:
    """
    Offload activations to CPU to save GPU memory.
    Following Unsloth's memory optimization patterns.
    """

    def __init__(self, offload_enabled: bool = True):
        self.offload_enabled = offload_enabled
        self.offloaded_activations = {}

    def offload(self, name: str, tensor: torch.Tensor):
        """Offload activation to CPU."""
        if self.offload_enabled:
            self.offloaded_activations[name] = tensor.cpu()
            return None
        return tensor

    def load(self, name: str, device: str = "cuda"):
        """Load activation back to GPU."""
        if name in self.offloaded_activations:
            return self.offloaded_activations[name].to(device)
        return None

    def clear(self):
        """Clear all offloaded activations."""
        self.offloaded_activations.clear()


def apply_quantization_aware_training_hooks(model: nn.Module):
    """
    Apply quantization-aware training hooks to the model.

    Args:
        model: The model to apply hooks to
    """

    def quantize_hook(module, input, output):
        """Hook to quantize activations during forward pass."""
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            # Quantize to int8 for memory efficiency
            scale = output.abs().max() / 127.0
            quantized = (output / scale).round().clamp(-128, 127).to(torch.int8)
            return quantized, scale
        return output

    # Register hooks for linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(quantize_hook)


def get_peft_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    Create PEFT (LoRA) configuration.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA
        task_type: Task type for PEFT

    Returns:
        LoraConfig
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type,
    )


def estimate_memory_usage(
    model: nn.Module, batch_size: int, seq_length: int
) -> Dict[str, float]:
    """
    Estimate memory usage for the model.

    Args:
        model: The model
        batch_size: Batch size
        seq_length: Sequence length

    Returns:
        Dictionary with memory estimates in GB
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory (rough approximation)
    # 4-bit quantized: ~0.5 bytes per parameter
    # FP16: 2 bytes per parameter
    # FP32: 4 bytes per parameter

    param_memory_gb = total_params * 0.5 / (1024**3)  # Assume 4-bit quantization

    # Activation memory (rough estimate)
    # Hidden size * batch_size * seq_length * num_layers * 2 bytes (FP16)
    hidden_size = getattr(model.config, "hidden_size", 4096)
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    activation_memory_gb = (
        hidden_size * batch_size * seq_length * num_layers * 2 / (1024**3)
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_gb": param_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "total_memory_gb": param_memory_gb + activation_memory_gb,
    }
