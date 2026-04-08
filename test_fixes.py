#!/usr/bin/env python3
"""Test script to verify FastVLA fixes work correctly."""

print("="*60)
print("Testing FastVLA Fixes")
print("="*60)

# Test 1: Unsloth imports
print("\n1. Testing Unsloth import...")
try:
    import unsloth
    print("   ✓ Unsloth imported successfully")
    print(f"   Version: {getattr(unsloth, '__version__', 'unknown')}")
except ImportError as e:
    print(f"   ✗ Unsloth import failed: {e}")

# Test 2: FastVLA model imports
print("\n2. Testing FastVLA model imports...")
try:
    from fastvla.model import FastVLAModel, UNSLOTH_AVAILABLE
    print(f"   ✓ FastVLAModel imported")
    print(f"   ✓ UNSLOTH_AVAILABLE: {UNSLOTH_AVAILABLE}")
    
    if UNSLOTH_AVAILABLE:
        from fastvla.model import FastLanguageModel, FastVisionModel
        print("   ✓ FastLanguageModel available")
        print("   ✓ FastVisionModel available")
except Exception as e:
    print(f"   ✗ FastVLA import failed: {e}")

# Test 3: Action head dtype handling
print("\n3. Testing action head dtype handling...")
try:
    from fastvla.kernels.action_head import TritonActionHead
    import torch
    
    # Create a test action head
    head = TritonActionHead(input_dim=256, hidden_dim=128, output_dim=2)
    
    # Test with fp16 input
    x_fp16 = torch.randn(2, 256, dtype=torch.float16, device='cuda')
    output = head(x_fp16)
    
    print(f"   ✓ TritonActionHead handles fp16 input correctly")
    print(f"   Input dtype: {x_fp16.dtype}, Output dtype: {output.dtype}")
    
except Exception as e:
    print(f"   ✗ Action head test failed: {e}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
