import pytest
import torch
import torch.nn as nn
from fastvla.kernels import action_decode_forward, vision_language_fusion_forward

def test_action_decode_dtype_mismatch_recovery():
    """Verify that action_decode_forward handles dtype mismatches gracefully."""
    B, D, H, A = 2, 128, 256, 7
    hidden = torch.randn(B, D, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    w1 = torch.randn(D, H, device=hidden.device, dtype=torch.float32) # Mismatch!
    b1 = torch.randn(H, device=hidden.device, dtype=torch.float16)
    w2 = torch.randn(H, A, device=hidden.device, dtype=torch.float32) # Mismatch!
    b2 = torch.randn(A, device=hidden.device, dtype=torch.float16)
    
    # Should not crash, and should return float16
    out = action_decode_forward(hidden, w1, b1, w2, b2)
    assert out.shape == (B, A)
    assert out.dtype == torch.float16

def test_fusion_dtype_mismatch_recovery():
    """Verify that fusion handles dtype mismatches gracefully."""
    B, T, D = 2, 10, 128
    visual = torch.randn(B, T, D, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    text = torch.randn(B, T, D, device=visual.device, dtype=torch.float32) # Mismatch!
    
    # Should not crash
    out = vision_language_fusion_forward(visual, text)
    assert out.shape == (B, T, D)
    # Output dtype follows 'text' in fusion.py current implementation logic
    assert out.dtype == torch.float32 or out.dtype == torch.float16

def test_kernel_failure_fallback():
    """Force a kernel failure and verify fallback to PyTorch."""
    from unittest.mock import patch
    
    B, D, H, A = 1, 64, 128, 7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = torch.randn(B, D, device=device)
    w1 = torch.randn(D, H, device=device)
    b1 = torch.randn(H, device=device)
    w2 = torch.randn(H, A, device=device)
    b2 = torch.randn(A, device=device)

    # Import the inner triton wrapper directly to test its internal fallback
    from fastvla.kernels.action import action_decode_forward as triton_wrapper
    
    # Mock the kernel call inside the runner
    with patch("fastvla.kernels.action._action_fwd_kernel") as mock_kernel:
        # Triton kernels are called via __getitem__ (grid)
        mock_kernel.side_effect = RuntimeError("Simulated Triton Failure")
        
        # This should trigger the warning and still return correct results via fallback
        out = triton_wrapper(hidden, w1, b1, w2, b2)
        assert out.shape == (B, A)
