"""Utility functions for FastVLA."""

import torch


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

    # Mandatory checks
    if require_cuda and not torch.cuda.is_available():
        raise EnvironmentCompatibilityError(
            "CUDA is required but not available. Ensure you have a GPU and proper drivers."
        )
    
    if require_unsloth and not unsloth_available:
        raise EnvironmentCompatibilityError(
            "Unsloth is required for this operation. Install it with: pip install unsloth"
        )

    # Informative summary
    cuda_info = torch.version.cuda if torch.cuda.is_available() else "N/A"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    print("\n" + "=" * 60)
    print("FastVLA Health Check")
    print("=" * 60)
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {cuda_info}")
    print(f"  Device:     {device_name}")
    print(f"  Unsloth:    {'✓ Available' if unsloth_available else '✗ Missing'}")
    print(f"  BnB:        {'✓ Available' if bnb_available else '✗ Missing'}")
    print(f"  Triton:     {'✓ Available' if triton_available else '✗ Missing'}")
    
    if torch.cuda.is_available():
        print(f"  {get_gpu_memory_report()}")
    print("=" * 60 + "\n")
