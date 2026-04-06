"""FastVLA: Efficient Vision-Language-Action Models for Robotics."""

__version__ = "0.1.0"

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
from .utils import get_device
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
