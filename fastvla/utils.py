"""Utility functions for FastVLA."""

import torch


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_environment_summary() -> None:
    """Print a clear summary of the FastVLA runtime environment.

    This helps diagnose environment differences that cause silent bugs
    (e.g., Kaggle vs Lightning Studio, missing Unsloth, wrong CUDA version).
    """
    unsloth_available = False
    bnb_available = False
    triton_available = False

    try:
        import unsloth  # noqa: F401
        unsloth_available = True
    except ImportError:
        pass

    try:
        import bitsandbytes  # noqa: F401
        bnb_available = True
    except ImportError:
        pass

    try:
        import triton  # noqa: F401
        triton_available = True
    except ImportError:
        pass

    def _check(name: str, available: bool) -> str:
        if available:
            return f"✓ {name} installed"
        return f"✗ {name} NOT installed"

    cuda_info = "N/A"
    device_name = "N/A"
    if torch.cuda.is_available():
        cuda_info = torch.version.cuda or "N/A"
        device_name = torch.cuda.get_device_name(0)

    print("\n" + "=" * 60)
    print("FastVLA Environment Check")
    print("=" * 60)
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Python:     {torch.version.python if hasattr(torch.version, 'python') else 'N/A'}")
    print(f"  CUDA:       {cuda_info}")
    print(f"  Device:     {device_name}")
    print(f"  Unsloth:    {_check('Unsloth', unsloth_available)}")
    print(f"  BnB:        {_check('bitsandbytes', bnb_available)}")
    print(f"  Triton:     {_check('Triton', triton_available)}")

    # Warnings
    warnings = []
    if torch.cuda.is_available() and not unsloth_available:
        warnings.append(
            "⚠ Unsloth not installed — 4-bit QLoRA loading will NOT work.\n"
            "  Install: pip install unsloth\n"
            "  Or set load_in_4bit=False to use FP16/FP32 (higher VRAM)."
        )
    if torch.cuda.is_available() and not bnb_available:
        warnings.append(
            "⚠ bitsandbytes not installed — 4-bit quantization unavailable."
        )
    if torch.cuda.is_available() and not triton_available:
        warnings.append(
            "⚠ Triton not installed — custom Triton kernels disabled.\n"
            "  Install: pip install triton"
        )

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()
    print("=" * 60 + "\n")
