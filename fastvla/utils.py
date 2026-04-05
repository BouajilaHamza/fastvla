"""Utility functions for FastVLA."""

import torch


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
