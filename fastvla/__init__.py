"""
FastVLA: Efficient Vision-Language-Action Models for Robotics.
Initialized with root-level stabilization for accelerate and unsloth.
"""

import logging
import torch

# ── 1. Break Accelerate Circular Imports ──────────────────────────────────
# We pre-import accelerate components to ensure the module is fully 
# initialized before transformers attempts to access its sub-modules.
try:
    import accelerate
    import accelerate.big_modeling
except ImportError:
    pass

# ── 2. Initialize Unsloth Patches (MUST be before Transformers) ────────────
# We apply unsloth patches at the library root so that any module importing
# fastvla (e.g. from fastvla import FastVLAConfig) will have patched transformers.
UNSLOTH_AVAILABLE = False
try:
    from unsloth import patch_model, patch_forward, patch_saving_functions
    patch_saving_functions()
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

# ── 3. Core FastVLA Imports ───────────────────────────────────────────────
# These modules may import transformers internally.
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

__version__ = "0.1.7.2"

# Print environment summary on first import
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
