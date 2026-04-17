"""
FastVLA: Efficient Vision-Language-Action Models for Robotics.
Strictly Lazy-Loaded to prevent environment-specific dependency loops.
"""

import sys
import importlib
import logging

__version__ = "0.1.8.1"

# ── 1. Native Lazy Loading (PEP 562) ───────────────────────────────────────

_import_structure = {
    "model": ["FastVLAModel"],
    "config": ["FastVLAConfig"],
    "training": ["FastVLATrainer"],
    "registry": ["VLAModelRegistry", "register_model"],
    "utils": ["get_device", "check_environment"],
    "data.datasets": ["get_dataset"],
    "data.collator": ["UnslothVLACollator"],
    "optimization": ["get_quantization_config", "get_8bit_optimizer"],
}

_submodule_objects = {}

def _ensure_env_stabilized():
    """Final safeguard: Ensures unsloth/accelerate are primed before submodules load."""
    if "fastvla._stabilized" in sys.modules:
        return

    # Phase A: Unsloth Patching (CRITICAL: MUST be before transformers/torch)
    try:
        import unsloth
        from unsloth import patch_saving_functions
        patch_saving_functions()
    except ImportError: pass

    # Phase B: Accelerate Circular Loop Break
    try:
        import accelerate
        import accelerate.big_modeling
    except ImportError: pass

    sys.modules["fastvla._stabilized"] = type("Stabilized", (), {})()

def __getattr__(name):
    if name == "__version__":
        return __version__

    target_submodule = None
    for sub, items in _import_structure.items():
        if name in items:
            target_submodule = sub
            break
    
    if target_submodule is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    # Load the submodule natively
    full_module_path = f"{__name__}.{target_submodule}"
    if target_submodule not in _submodule_objects:
        _ensure_env_stabilized()
        _submodule_objects[target_submodule] = importlib.import_module(full_module_path)
    
    return getattr(_submodule_objects[target_submodule], name)

def __dir__():
    return list(_import_structure.keys()) + list(globals().keys())
