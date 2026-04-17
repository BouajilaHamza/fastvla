"""Utility functions for FastVLA."""

import torch

# ── 1. Root-Level Stabilization ───────────────────────────────────────────
# We pre-import unsloth to ensure it patches transformers before other 
# modules use it.
try:
    import unsloth as _  # noqa: F401
except ImportError:
    pass


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_memory_report() -> str:
    """Return a human-readable string of current GPU memory usage."""
    if not torch.cuda.is_available():
        return "GPU: N/A (CPU Mode)"
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    return (
        f"GPU Memory: {allocated:.2f}GB allocated, "
        f"{reserved:.2f}GB reserved, "
        f"{max_allocated:.2f}GB peak"
    )

def check_environment(require_cuda: bool = False, require_unsloth: bool = False) -> None:
    """
    Perform a rigorous health check of the runtime environment.
    
    Args:
        require_cuda: If True, raises EnvironmentCompatibilityError if CUDA is missing.
        require_unsloth: If True, raises EnvironmentCompatibilityError if Unsloth is missing.

    Raises:
        EnvironmentCompatibilityError: If a mandatory requirement is not met.
    """
    from .exceptions import EnvironmentCompatibilityError
    
    unsloth_available = False
    bnb_available = False
    triton_available = False

    try:
        import unsloth  # noqa: F401
        unsloth_available = True
    except ImportError:
        pass

    try:
        import bitsandbytes as bnb  # noqa: F401
        bnb_available = True
    except ImportError:
        pass

    try:
        import triton  # noqa: F401
        triton_available = True
    except ImportError:
        pass

    # 1. Device Check
    device = get_device()
    if require_cuda and device == "cpu":
        raise EnvironmentCompatibilityError(
            "CUDA-enabled GPU is required but not found. "
            "If using Kaggle, ensure 'GPU T4 x2' is enabled in the sidebar."
        )

    # 2. Dependency Check
    if require_unsloth and not unsloth_available:
        raise EnvironmentCompatibilityError(
            "Unsloth is required for 4-bit optimization but not installed. "
            "Please run '!pip install unsloth' first."
        )

    # 3. Environment Stability Check (The JIT/Dynamo test)
    # We don't perform destructive tests here, just log availability.
    # The actual stabilization logic is handled in __init__.py and model.py
