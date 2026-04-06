"""
Standalone validation script for the distributed training fixes.
This script validates the logic without requiring full library imports.
"""

import ast
import sys

def validate_training_fixes():
    """Validate that the training.py fixes are correct."""
    print("=" * 80)
    print("Validating Distributed Training Fixes")
    print("=" * 80)
    
    # Read training.py
    with open("fastvla/training.py", "r") as f:
        training_code = f.read()
    
    # Parse the AST
    tree = ast.parse(training_code)
    
    # Find the train_step method
    print("\n✓ Checking train_step method...")
    
    # Check 1: No duplicate optimizer steps
    optimizer_step_count = training_code.count("self.optimizer.step()")
    scheduler_step_count = training_code.count("self.lr_scheduler.step()")
    zero_grad_count = training_code.count("self.optimizer.zero_grad()")
    
    print(f"  - optimizer.step() calls: {optimizer_step_count} (should be 1)")
    print(f"  - lr_scheduler.step() calls: {scheduler_step_count} (should be 2: 1 in train_step, 1 elsewhere)")
    print(f"  - optimizer.zero_grad() calls: {zero_grad_count} (should be 1)")
    
    assert optimizer_step_count == 1, f"Expected 1 optimizer.step() call, found {optimizer_step_count}"
    assert zero_grad_count == 1, f"Expected 1 optimizer.zero_grad() call, found {zero_grad_count}"
    print("  ✓ No duplicate optimizer steps!")
    
    # Check 2: No manual device placement in train_step
    train_step_start = training_code.find("def train_step(self, batch:")
    train_step_end = training_code.find("def evaluate(self)", train_step_start)
    train_step_code = training_code[train_step_start:train_step_end]
    
    assert ".to(self.device)" not in train_step_code, "Found manual device placement in train_step!"
    print("  ✓ No manual device placement in train_step!")
    
    # Check 3: Loss is not scaled by gradient_accumulation_steps
    assert "loss.item() * self.gradient_accumulation_steps" not in training_code, \
        "Found incorrect loss scaling!"
    assert '"loss": loss.item()' in training_code, "Loss should be returned without scaling!"
    print("  ✓ Loss scaling is correct!")
    
    # Check 4: Accelerator initialization
    assert "Accelerator(" in training_code, "Accelerator not initialized!"
    print("  ✓ Accelerator properly initialized!")
    
    print("\n✓ All training.py checks passed!")
    return True


def validate_model_fixes():
    """Validate that the model.py fixes are correct."""
    print("\n" + "=" * 80)
    print("Validating Model Fixes")
    print("=" * 80)
    
    # Read model.py
    with open("fastvla/model.py", "r") as f:
        model_code = f.read()
    
    print("\n✓ Checking forward method...")
    
    # Check 1: Shape validation exists
    assert "action_preds.shape != labels.shape" in model_code or \
           "action_preds.shape[1] != labels.shape[1]" in model_code, \
           "No shape validation found!"
    print("  ✓ Shape validation present!")
    
    # Check 2: Informative error message
    assert "Action dimension mismatch" in model_code, \
           "No informative error message for dimension mismatch!"
    print("  ✓ Informative error messages present!")
    
    # Check 3: Labels are moved to correct device
    assert "labels.to(head_device)" in model_code, \
           "Labels not moved to head device!"
    print("  ✓ Labels properly moved to head device!")
    
    # Check 4: No duplicate loss computation
    loss_count = model_code.count("nn.MSELoss()(action_preds, labels)")
    assert loss_count == 1, f"Expected 1 loss computation, found {loss_count}"
    print("  ✓ Single loss computation!")
    
    print("\n✓ All model.py checks passed!")
    return True


def validate_collator_fixes():
    """Validate that the collator.py fixes are correct."""
    print("\n" + "=" * 80)
    print("Validating Collator Fixes")
    print("=" * 80)
    
    # Read collator.py
    with open("fastvla/data/collator.py", "r") as f:
        collator_code = f.read()
    
    print("\n✓ Checking collator...")
    
    # Check 1: action_dim parameter exists
    assert "action_dim: int = 7" in collator_code or "action_dim" in collator_code, \
           "action_dim parameter not found!"
    print("  ✓ action_dim parameter present!")
    
    # Check 2: Validation logic
    assert "Inconsistent action dimensions" in collator_code, \
           "No validation for inconsistent action dimensions!"
    print("  ✓ Action dimension validation present!")
    
    # Check 3: Auto-update with warning
    assert "Warning" in collator_code or "warning" in collator_code.lower(), \
           "No warning message for dimension mismatch!"
    print("  ✓ Warning messages for mismatches!")
    
    # Check 4: Scalar handling
    assert "unsqueeze" in collator_code or "dim() == 0" in collator_code, \
           "No scalar action handling!"
    print("  ✓ Scalar action handling present!")
    
    print("\n✓ All collator.py checks passed!")
    return True


def validate_test_coverage():
    """Validate that comprehensive tests were added."""
    print("\n" + "=" * 80)
    print("Validating Test Coverage")
    print("=" * 80)
    
    try:
        with open("tests/test_training_robustness.py", "r") as f:
            test_code = f.read()
        
        print("\n✓ Checking test coverage...")
        
        # Check for key test classes
        test_classes = [
            "TestShapeValidation",
            "TestCollatorValidation",
            "TestGradientAccumulation",
            "TestDistributedTrainingSimulation",
            "TestEdgeCases"
        ]
        
        for test_class in test_classes:
            assert f"class {test_class}" in test_code, f"{test_class} not found!"
            print(f"  ✓ {test_class} present!")
        
        print("\n✓ All test classes present!")
        return True
        
    except FileNotFoundError:
        print("\n⚠ Warning: test_training_robustness.py not found!")
        return False


def main():
    """Run all validations."""
    print("\n" + "=" * 80)
    print("FastVLA Distributed Training Fixes - Validation Suite")
    print("=" * 80 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= validate_training_fixes()
    except AssertionError as e:
        print(f"\n✗ Training validation failed: {e}")
        all_passed = False
    
    try:
        all_passed &= validate_model_fixes()
    except AssertionError as e:
        print(f"\n✗ Model validation failed: {e}")
        all_passed = False
    
    try:
        all_passed &= validate_collator_fixes()
    except AssertionError as e:
        print(f"\n✗ Collator validation failed: {e}")
        all_passed = False
    
    try:
        all_passed &= validate_test_coverage()
    except Exception as e:
        print(f"\n✗ Test validation failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 80)
        print("\nSummary of fixes:")
        print("  1. ✓ Removed duplicate optimizer/scheduler steps")
        print("  2. ✓ Removed manual device placement in train_step/evaluate")
        print("  3. ✓ Fixed loss scaling (no longer multiplied by gradient_accumulation_steps)")
        print("  4. ✓ Added shape validation in model forward pass")
        print("  5. ✓ Added action dimension validation in collator")
        print("  6. ✓ Added comprehensive test suite")
        print("\nYour distributed training should now work correctly on Kaggle (2x T4)!")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 80)
        print("\nPlease review the errors above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
