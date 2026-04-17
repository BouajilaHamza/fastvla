"""
FastVLA: Efficient Vision-Language-Action Models for Robotics.
Strictly Lazy-Loaded to prevent environment-specific dependency loops.
"""

import sys
import importlib
import logging

__version__ = "0.1.8.1"

# ── 1. Lazy Module Proxy ───────────────────────────────────────────────────
# This mimics the internal 'transformers' lazy loading logic.
# It prevents any 'transformers' or 'torch' code from running until 
# a class is actually accessed.

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

class _LazyModule:
    def __init__(self, name, import_structure):
        self.name = name
        self.import_structure = import_structure
        self._modules = {}

    def __getattr__(self, name):
        # 1. Find which submodule contains the requested attribute
        target_submodule = None
        for sub, items in self.import_structure.items():
            if name in items:
                target_submodule = sub
                break
        
        if target_submodule is None:
            raise AttributeError(f"module {self.name} has no attribute {name}")

        # 2. Load the submodule (This is where the 'real' imports happen)
        full_module_path = f".{target_submodule}"
        if target_submodule not in self._modules:
            # TRIGGER STABILIZATION BEFORE REAL IMPORT
            self._ensure_env_stabilized()
            self._modules[target_submodule] = importlib.import_module(full_module_path, __package__)
        
        return getattr(self._modules[target_submodule], name)

    def _ensure_env_stabilized(self):
        """Final safeguard: Ensures unsloth/accelerate are primed before submodules load."""
        if "fastvla._stabilized" in sys.modules:
            return

        # Phase A: Accelerate Circular Loop Break
        try:
            import accelerate
            import accelerate.big_modeling
        except ImportError: pass

        # Phase B: Unsloth Patching (MUST be before transformers)
        try:
            import unsloth
            from unsloth import patch_saving_functions
            patch_saving_functions()
        except ImportError: pass

        sys.modules["fastvla._stabilized"] = type("Stabilized", (), {})()

# ── 2. Create the Proxy ────────────────────────────────────────────────────
# When you 'import fastvla', you are actually interacting with this object.
sys.modules[__name__] = _LazyModule(__name__, _import_structure)
