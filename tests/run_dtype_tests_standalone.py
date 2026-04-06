"""
Standalone dtype compatibility test - bypasses broken imports.
Tests ONLY the kernel modules directly without importing from fastvla.__init__
"""
import sys
import os
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("FASTVLA DTYPE COMPATIBILITY TEST SUITE (STANDALONE)")
print("=" * 80)

passed = 0
failed = 0
errors = []


def test(name, func):
    """Run a test function."""
    global passed, failed
    try:
        func()
        passed += 1
        print(f"✅ PASS: {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"❌ FAIL: {name}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


# ── TritonActionHead Tests ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("1. TRITON ACTION HEAD TESTS")
print("=" * 80)

# Import directly without going through fastvla.__init__
from fastvla.kernels import action_head


def test_triton_cpu_fp32_forward():
    head = action_head.TritonActionHead(input_dim=768, hidden_dim=256, output_dim=7)
    x = torch.randn(2, 768, dtype=torch.float32)
    with torch.no_grad():
        output = head(x)
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_triton_cpu_fp16_forward():
    """CRITICAL TEST: This was the original bug."""
    head = action_head.TritonActionHead(input_dim=768, hidden_dim=256, output_dim=7)
    x = torch.randn(2, 768, dtype=torch.float16)
    with torch.no_grad():
        output = head(x)  # This used to raise dtype mismatch error
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_triton_cpu_fp32_backward():
    head = action_head.TritonActionHead(input_dim=768, hidden_dim=256, output_dim=7)
    x = torch.randn(2, 768, dtype=torch.float32, requires_grad=True)
    output = head(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"


def test_triton_cpu_fp16_backward():
    """CRITICAL TEST: Backward with float16 input."""
    head = action_head.TritonActionHead(input_dim=768, hidden_dim=256, output_dim=7)
    x = torch.randn(2, 768, dtype=torch.float16, requires_grad=True)
    output = head(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"


test("TritonActionHead CPU float32 forward", test_triton_cpu_fp32_forward)
test("TritonActionHead CPU float16 forward (MIXED PRECISION - ORIGINAL BUG)", test_triton_cpu_fp16_forward)
test("TritonActionHead CPU float32 backward", test_triton_cpu_fp32_backward)
test("TritonActionHead CPU float16 backward (MIXED PRECISION)", test_triton_cpu_fp16_backward)


# ── DiscreteActionHead Tests ────────────────────────────────────────────

print("\n" + "=" * 80)
print("2. DISCRETE ACTION HEAD TESTS")
print("=" * 80)

# We need to import the adapters module directly
# First let's import just the module file directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "adapters_action_head",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastvla", "adapters", "action_head.py")
)
adapters_action_head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters_action_head)


def test_discrete_cpu_fp32_forward():
    head = adapters_action_head.DiscreteActionHead(input_dim=768, action_dim=7, hidden_dim=256, num_bins=256)
    x = torch.randn(2, 768, dtype=torch.float32)
    with torch.no_grad():
        output = head(x)
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output not in [0, 1]"


def test_discrete_cpu_fp16_forward():
    """CRITICAL TEST: Mixed precision with DiscreteActionHead."""
    head = adapters_action_head.DiscreteActionHead(input_dim=768, action_dim=7, hidden_dim=256, num_bins=256)
    x = torch.randn(2, 768, dtype=torch.float16)
    with torch.no_grad():
        output = head(x)  # This would fail without dtype fix
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output not in [0, 1]"


def test_discrete_cpu_fp32_backward():
    head = adapters_action_head.DiscreteActionHead(input_dim=768, action_dim=7, hidden_dim=256, num_bins=256)
    x = torch.randn(2, 768, dtype=torch.float32, requires_grad=True)
    output = head(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"


test("DiscreteActionHead CPU float32 forward", test_discrete_cpu_fp32_forward)
test("DiscreteActionHead CPU float16 forward (MIXED PRECISION)", test_discrete_cpu_fp16_forward)
test("DiscreteActionHead CPU float32 backward", test_discrete_cpu_fp32_backward)


# ── ContinuousActionHead Tests ──────────────────────────────────────────

print("\n" + "=" * 80)
print("3. CONTINUOUS ACTION HEAD TESTS")
print("=" * 80)


def test_continuous_cpu_fp32_forward():
    head = adapters_action_head.ContinuousActionHead(input_dim=768, action_dim=7, hidden_dim=256, use_triton=False)
    x = torch.randn(2, 768, dtype=torch.float32)
    with torch.no_grad():
        output = head(x)
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output not in [-1, 1]"


def test_continuous_cpu_fp16_forward():
    """CRITICAL TEST: Mixed precision with ContinuousActionHead."""
    head = adapters_action_head.ContinuousActionHead(input_dim=768, action_dim=7, hidden_dim=256, use_triton=False)
    x = torch.randn(2, 768, dtype=torch.float16)
    with torch.no_grad():
        output = head(x)  # This would fail without dtype fix
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output not in [-1, 1]"


def test_continuous_cpu_fp32_backward():
    head = adapters_action_head.ContinuousActionHead(input_dim=768, action_dim=7, hidden_dim=256, use_triton=False)
    x = torch.randn(2, 768, dtype=torch.float32, requires_grad=True)
    output = head(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"


test("ContinuousActionHead CPU float32 forward", test_continuous_cpu_fp32_forward)
test("ContinuousActionHead CPU float16 forward (MIXED PRECISION)", test_continuous_cpu_fp16_forward)
test("ContinuousActionHead CPU float32 backward", test_continuous_cpu_fp32_backward)


# ── FlowMatchingActionHead Tests ────────────────────────────────────────

print("\n" + "=" * 80)
print("4. FLOW MATCHING ACTION HEAD TESTS")
print("=" * 80)


def test_flow_cpu_fp16_forward():
    """CRITICAL TEST: Mixed precision with FlowMatchingActionHead."""
    head = adapters_action_head.FlowMatchingActionHead(input_dim=768, action_dim=7, hidden_dim=256)
    x = torch.randn(2, 768, dtype=torch.float16)
    with torch.no_grad():
        output = head(x)  # This would fail without dtype fix
    assert output.shape == (2, 7), f"Expected shape (2, 7), got {output.shape}"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output not in [-1, 1]"


test("FlowMatchingActionHead CPU float16 forward (MIXED PRECISION)", test_flow_cpu_fp16_forward)


# ── Fusion Kernel Tests ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("5. VISION-LANGUAGE FUSION TESTS")
print("=" * 80)

from fastvla.kernels import fusion


def test_fusion_cpu_fp32():
    visual = torch.randn(2, 196, 768, dtype=torch.float32)
    text = torch.randn(2, 196, 768, dtype=torch.float32)
    output = fusion.vision_language_fusion_forward(visual, text)
    assert output.shape == (2, 196, 768)
    assert not torch.isnan(output).any()


def test_fusion_cpu_fp16():
    visual = torch.randn(2, 196, 768, dtype=torch.float16)
    text = torch.randn(2, 196, 768, dtype=torch.float16)
    output = fusion.vision_language_fusion_forward(visual, text)
    assert output.shape == (2, 196, 768)
    assert not torch.isnan(output).any()


def test_fusion_mixed_dtype():
    """Fusion with mismatched dtypes should handle gracefully."""
    visual = torch.randn(2, 196, 768, dtype=torch.float16)
    text = torch.randn(2, 196, 768, dtype=torch.float32)
    output = fusion.vision_language_fusion_forward(visual, text)
    assert output.shape == (2, 196, 768)
    assert not torch.isnan(output).any()


test("Fusion CPU float32", test_fusion_cpu_fp32)
test("Fusion CPU float16", test_fusion_cpu_fp16)
test("Fusion mixed dtype (mismatched)", test_fusion_mixed_dtype)


# ── Data Collator Tests ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("6. DATA COLLATOR TESTS")
print("=" * 80)

spec2 = importlib.util.spec_from_file_location(
    "collator",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastvla", "data", "collator.py")
)
collator_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(collator_module)


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
        }


def test_collator_complete_data():
    """Collator with complete data."""
    tokenizer = MockTokenizer()
    collator = collator_module.UnslothVLACollator(tokenizer=tokenizer, action_dim=7)
    features = [
        {
            "images": {"cam1": torch.randn(3, 224, 224)},
            "actions": torch.randn(7),
            "instructions": "pick up the block",
        }
        for _ in range(2)
    ]
    batch = collator(features)
    assert "pixel_values" in batch, "Missing pixel_values"
    assert "input_ids" in batch, "Missing input_ids"
    assert "labels" in batch, "Missing labels"
    assert batch["pixel_values"].shape == (2, 1, 3, 224, 224)
    assert batch["labels"].shape == (2, 7)


def test_collator_missing_instructions():
    """Collator handles missing instructions gracefully."""
    tokenizer = MockTokenizer()
    collator = collator_module.UnslothVLACollator(tokenizer=tokenizer, action_dim=7)
    features = [
        {
            "images": {"cam1": torch.randn(3, 224, 224)},
            "actions": torch.randn(7),
            # No "instructions" key
        }
        for _ in range(2)
    ]
    batch = collator(features)
    assert "input_ids" in batch, "Missing input_ids even with fallback"
    assert "attention_mask" in batch, "Missing attention_mask even with fallback"
    assert batch["input_ids"].shape[0] == 2, f"Wrong batch size: {batch['input_ids'].shape}"


def test_collator_missing_images_raises():
    """Collator raises error when images are missing."""
    tokenizer = MockTokenizer()
    collator = collator_module.UnslothVLACollator(tokenizer=tokenizer, action_dim=7)
    features = [
        {
            "actions": torch.randn(7),
            "instructions": "pick up the block",
            # No "images" key
        }
        for _ in range(2)
    ]
    try:
        collator(features)
        raise AssertionError("Should have raised ValueError for missing images")
    except ValueError as e:
        assert "missing required keys" in str(e), f"Wrong error message: {e}"


test("Collator with complete data", test_collator_complete_data)
test("Collator with missing instructions (graceful fallback)", test_collator_missing_instructions)
test("Collator with missing images (raises error)", test_collator_missing_images_raises)


# ── Mixed Precision Integration Tests ───────────────────────────────────

print("\n" + "=" * 80)
print("7. MIXED PRECISION INTEGRATION TESTS")
print("=" * 80)


def test_autocast_simulation_cpu():
    """Simulate autocast behavior on CPU (float16 input to float32 model)."""
    head = action_head.TritonActionHead(input_dim=256, hidden_dim=64, output_dim=5)
    x_fp16 = torch.randn(4, 256, dtype=torch.float16)
    
    # This should NOT raise dtype mismatch
    with torch.no_grad():
        output = head(x_fp16)
    
    assert output.shape == (4, 5), f"Expected shape (4, 5), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_dtype_consistency():
    """Verify float16 and float32 produce similar outputs."""
    head = action_head.TritonActionHead(input_dim=128, hidden_dim=32, output_dim=3)
    torch.manual_seed(42)
    
    x_fp32 = torch.randn(2, 128, dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)
    
    with torch.no_grad():
        out_fp32 = head(x_fp32)
        out_fp16 = head(x_fp16).to(torch.float32)
    
    # Outputs should be numerically similar (allowing for precision loss)
    assert torch.allclose(out_fp32, out_fp16, rtol=1e-2, atol=1e-2), (
        f"float16/float32 outputs differ: max diff = {(out_fp32 - out_fp16).abs().max()}"
    )


test("Autocast simulation (float16 input to float32 model)", test_autocast_simulation_cpu)
test("Dtype consistency (float16 vs float32 outputs)", test_dtype_consistency)


# ── Summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"\nTotal tests: {passed + failed}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")

if failed > 0:
    print("\n" + "-" * 80)
    print("FAILED TESTS:")
    print("-" * 80)
    for name, error in errors:
        print(f"\n  ❌ {name}")
        print(f"     {error}")
    print("\n" + "=" * 80)
    print("OVERALL RESULT: FAILED")
    print("=" * 80)
    sys.exit(1)
else:
    print("\n" + "=" * 80)
    print("OVERALL RESULT: ALL TESTS PASSED ✅")
    print("=" * 80)
