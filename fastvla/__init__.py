"""FastVLA: Efficient Vision-Language-Action Models for Robotics."""

__version__ = "0.1.3"

from .model import FastVLAModel
from .config import FastVLAConfig
from .registry import (
    VLAModelRegistry,
    register_model,
    VLAModelConfig,
    VisionEncoderConfig,
    LLMConfig,
    ActionHeadConfig,
    ProjectorConfig,
)
from .utils import get_device, check_environment
from .data.collator import UnslothVLACollator
from .data.datasets import get_dataset, RoboticsDataset, LIBERODataset, FrankaKitchenDataset, LeRobotDataset
from .training import FastVLATrainer
from .optimization import (
    get_quantization_config,
    get_8bit_optimizer,
    enable_gradient_checkpointing,
    ActivationOffloader,
    get_peft_config,
    estimate_memory_usage,
)
from .benchmarking import PerformanceProfiler, compare_models, print_benchmark_results

# Print environment summary on first import so users immediately see
# what's available (Unsloth, BnB, Triton, GPU) and can diagnose mismatches.
_printed_env = False

def _print_env_once():
    global _printed_env
    if not _printed_env:
        _printed_env = True
        try:
            check_environment()
        except Exception:
            pass  # Never crash on env diagnostic

_print_env_once()

__all__ = [
    "FastVLAModel",
    "FastVLAConfig",
    "VLAModelRegistry",
    "register_model",
    "VLAModelConfig",
    "VisionEncoderConfig",
    "LLMConfig",
    "ActionHeadConfig",
    "ProjectorConfig",
    "get_device",
    "check_environment",
    "UnslothVLACollator",
    "get_dataset",
    "RoboticsDataset",
    "LIBERODataset",
    "FrankaKitchenDataset",
    "LeRobotDataset",
    "FastVLATrainer",
    "get_quantization_config",
    "get_8bit_optimizer",
    "enable_gradient_checkpointing",
    "ActivationOffloader",
    "get_peft_config",
    "estimate_memory_usage",
    "PerformanceProfiler",
    "compare_models",
    "print_benchmark_results",
]
