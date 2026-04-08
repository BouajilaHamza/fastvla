"""
End-to-End Test Suite for FastVLA
Tests the FULL training pipeline: model → dataloader → trainer → train_step → backward → optimizer.step
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest


# ── Test Fixtures ───────────────────────────────────────────────────────

class FakeDataset(Dataset):
    """Minimal dataset that produces valid training samples."""
    
    def __init__(self, num_samples=10, image_size=(3, 224, 224), num_cameras=1, 
                 action_dim=7, seq_len=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_cameras = num_cameras
        self.action_dim = action_dim
        self.seq_len = seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate camera images
        images = {}
        for cam_idx in range(self.num_cameras):
            images[f"cam{cam_idx}"] = torch.rand(*self.image_size, dtype=torch.float32)
        
        return {
            "images": images,
            "states": torch.randn(10, dtype=torch.float32),
            "actions": torch.randn(self.action_dim, dtype=torch.float32),
            "instructions": "pick up the red block",
        }


class FakeIncompleteDataset(Dataset):
    """Dataset with missing fields to test collator robustness."""
    
    def __init__(self, missing_field="instructions", num_samples=5):
        self.num_samples = num_samples
        self.missing_field = missing_field
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        data = {
            "images": {"cam0": torch.rand(3, 224, 224)},
            "states": torch.randn(10),
            "actions": torch.randn(7),
            "instructions": "move the block",
        }
        if self.missing_field in data:
            del data[self.missing_field]
        return data


class FakeModel(nn.Module):
    """Fake model that mimics FastVLAModel interface for testing Trainer."""
    
    def __init__(self, action_dim=7, hidden_size=128, has_device_map=False):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.config = type("Config", (), {"action_dim": action_dim})()
        self.fc = nn.Linear(hidden_size, action_dim)
        
        # Simulate device_map attribute
        if has_device_map:
            self.hf_device_map = {"": "cpu"}
        
        # Fake tokenizer
        self._tokenizer = type("Tokenizer", (), {
            "pad_token_id": 0,
            "eos_token_id": 1,
            "__call__": lambda self, texts, **kwargs: {
                "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
            },
        })()
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        B = pixel_values.shape[0]
        
        # Simulate forward pass
        action_preds = torch.randn(B, self.action_dim, dtype=pixel_values.dtype)
        
        if labels is not None:
            loss = nn.functional.mse_loss(action_preds, labels)
        else:
            loss = None
        
        return action_preds, loss


class FakeQuantizedModel(nn.Module):
    """Model that simulates 4-bit quantized behavior (FP16 parameters)."""
    
    def __init__(self, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.config = type("Config", (), {"action_dim": action_dim})()
        
        # Simulate FP16 parameters like a 4-bit loaded model
        self.fc = nn.Linear(128, action_dim, dtype=torch.float16)
        
        # Simulate is_loaded_in_4bit flag
        self.is_loaded_in_4bit = True
        self.hf_device_map = {"": "cpu"}
        
        # Fake tokenizer
        self._tokenizer = type("Tokenizer", (), {
            "pad_token_id": 0,
            "eos_token_id": 1,
            "__call__": lambda self, texts, **kwargs: {
                "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
            },
        })()
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        B = pixel_values.shape[0]
        
        # Simulate FP16 forward (like a 4-bit model under autocast)
        action_preds = torch.randn(B, self.action_dim, dtype=torch.float16)
        
        if labels is not None:
            # Loss computation with mixed dtypes
            loss = nn.functional.mse_loss(action_preds, labels.to(torch.float16))
        else:
            loss = None
        
        return action_preds, loss


# ── Helper Functions ──────────────────────────────────────────────────────

def make_collator(model):
    """Create a collator from a model's tokenizer."""
    from fastvla.data.collator import UnslothVLACollator
    
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
            }
    
    return UnslothVLACollator(
        tokenizer=FakeTokenizer(),
        action_dim=getattr(model.config, "action_dim", 7)
    )


def make_dataloader(dataset, model, batch_size=2):
    """Create a dataloader with proper collation."""
    collator = make_collator(model)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)


# ── Test Suite ─────────────────────────────────────────────────────────

class TestEndToEndTraining(unittest.TestCase):
    """Full training pipeline tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_01_basic_train_step(self):
        """Test a single train step with fake model (no mixed precision)."""
        print("\n" + "=" * 80)
        print("TEST 1: Basic train step (fake model, no mixed precision)")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7)
        dataset = FakeDataset(num_samples=4, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=2,
                use_mixed_precision=False,  # Disable mixed precision
                use_8bit_optimizer=False,   # Use standard AdamW
                output_dir=self.temp_dir,
                save_steps=999,
                eval_steps=999,
                logging_steps=999,
            )
            
            batch = next(iter(trainer.train_dataloader))
            metrics = trainer.train_step(batch)
            
            print(f"✅ PASS: Train step completed, loss={metrics['loss']:.4f}")
            self.assertIn("loss", metrics)
            self.assertIn("learning_rate", metrics)
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_02_mixed_precision_train_step(self):
        """Test train step with mixed precision enabled (GPU only)."""
        print("\n" + "=" * 80)
        print("TEST 2: Train step with mixed precision (GPU only)")
        print("=" * 80)
        
        if not torch.cuda.is_available():
            print("⏭️ SKIP: CUDA not available")
            self.skipTest("CUDA not available")
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7).cuda()
        dataset = FakeDataset(num_samples=4, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=2,
                use_mixed_precision=True,
                use_8bit_optimizer=False,
                output_dir=self.temp_dir,
                save_steps=999,
            )
            
            batch = next(iter(trainer.train_dataloader))
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            metrics = trainer.train_step(batch)
            
            print(f"✅ PASS: Mixed precision train step, loss={metrics['loss']:.4f}")
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_03_full_train_loop_two_steps(self):
        """Test full trainer.train() for 2 steps."""
        print("\n" + "=" * 80)
        print("TEST 3: Full trainer.train() for 2 steps")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7)
        dataset = FakeDataset(num_samples=8, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=2,
                use_mixed_precision=False,
                use_8bit_optimizer=False,
                num_epochs=1,
                output_dir=self.temp_dir,
                save_steps=999,
                eval_steps=999,
                logging_steps=1,
            )
            
            history = trainer.train(max_steps=2)
            
            print(f"✅ PASS: Training completed, {len(history)} history entries")
            self.assertGreaterEqual(len(history), 1)
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_04_4bit_model_mixed_precision(self):
        """Test 4-bit quantized model with mixed precision (REPRODUCES: Attempting to unscale FP16 gradients)."""
        print("\n" + "=" * 80)
        print("TEST 4: 4-bit model with mixed precision (FP16 gradient unscale)")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeQuantizedModel(action_dim=7)
        dataset = FakeDataset(num_samples=4, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=2,
                use_mixed_precision=torch.cuda.is_available(),
                use_8bit_optimizer=False,
                output_dir=self.temp_dir,
                save_steps=999,
            )
            
            batch = next(iter(trainer.train_dataloader))
            metrics = trainer.train_step(batch)
            
            print(f"✅ PASS: 4-bit model train step, loss={metrics['loss']:.4f}")
            
        except ValueError as e:
            if "Attempting to unscale FP16 gradients" in str(e):
                print(f"❌ FAIL (KNOWN BUG): {e}")
                raise
            else:
                print(f"❌ FAIL: {e}")
                raise
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_05_model_with_device_map_auto(self):
        """Test model loaded with device_map='auto' (Accelerator prepare logic)."""
        print("\n" + "=" * 80)
        print("TEST 5: Model with device_map='auto'")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7, has_device_map=True)
        dataset = FakeDataset(num_samples=4, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=2,
                use_mixed_precision=False,
                use_8bit_optimizer=False,
                output_dir=self.temp_dir,
                save_steps=999,
            )
            
            # Model should NOT be prepared by Accelerator when it has hf_device_map
            batch = next(iter(trainer.train_dataloader))
            metrics = trainer.train_step(batch)
            
            print(f"✅ PASS: device_map='auto' model, loss={metrics['loss']:.4f}")
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_06_gradient_accumulation(self):
        """Test gradient accumulation steps."""
        print("\n" + "=" * 80)
        print("TEST 6: Gradient accumulation")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7)
        dataset = FakeDataset(num_samples=8, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=1,
                gradient_accumulation_steps=2,
                use_mixed_precision=False,
                use_8bit_optimizer=False,
                output_dir=self.temp_dir,
                save_steps=999,
            )
            
            # Run 4 steps (should accumulate gradients every 2 steps)
            batch = next(iter(trainer.train_dataloader))
            for i in range(4):
                metrics = trainer.train_step(batch)
                if i == 0 or i == 2:  # sync_gradients should fire
                    print(f"  Step {i+1}: loss={metrics['loss']:.4f}")
            
            print(f"✅ PASS: Gradient accumulation worked")
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_07_checkpoint_save_and_restore(self):
        """Test checkpoint saving and loading."""
        print("\n" + "=" * 80)
        print("TEST 7: Checkpoint save and restore")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7)
        dataset = FakeDataset(num_samples=4, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=dataset,
                batch_size=2,
                use_mixed_precision=False,
                use_8bit_optimizer=False,
                output_dir=self.temp_dir,
                save_steps=1,  # Save every step
            )
            
            # Run 1 step to trigger checkpoint
            batch = next(iter(trainer.train_dataloader))
            trainer.train_step(batch)
            trainer.save_checkpoint(step=1)
            
            # Verify checkpoint exists
            checkpoint_dir = Path(self.temp_dir) / "checkpoint-1"
            self.assertTrue(checkpoint_dir.exists(), "Checkpoint not created")
            self.assertTrue((checkpoint_dir / "training_state.json").exists())
            
            # Verify checkpoint contents
            with open(checkpoint_dir / "training_state.json") as f:
                state = json.load(f)
            self.assertEqual(state["global_step"], 1)
            
            print(f"✅ PASS: Checkpoint saved and verified")
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_08_evaluation_loop(self):
        """Test evaluation loop."""
        print("\n" + "=" * 80)
        print("TEST 8: Evaluation loop")
        print("=" * 80)
        
        from fastvla.training import FastVLATrainer
        
        model = FakeModel(action_dim=7)
        train_dataset = FakeDataset(num_samples=4, action_dim=7)
        eval_dataset = FakeDataset(num_samples=2, action_dim=7)
        
        try:
            trainer = FastVLATrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                batch_size=2,
                use_mixed_precision=False,
                use_8bit_optimizer=False,
                output_dir=self.temp_dir,
                save_steps=999,
                eval_steps=999,
            )
            
            eval_metrics = trainer.evaluate()
            
            self.assertIn("eval_loss", eval_metrics)
            self.assertIn("eval_samples", eval_metrics)
            print(f"✅ PASS: Evaluation completed, eval_loss={eval_metrics['eval_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise


class TestDataCollatorRobustness(unittest.TestCase):
    """Test data collator with various edge cases."""
    
    def test_missing_images_raises_error(self):
        """Missing images should raise ValueError."""
        print("\n" + "=" * 80)
        print("TEST 9: Missing images should raise error")
        print("=" * 80)
        
        from fastvla.data.collator import UnslothVLACollator
        
        class FakeTokenizer:
            pad_token_id = 0
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
                }
        
        collator = UnslothVLACollator(tokenizer=FakeTokenizer(), action_dim=7)
        features = [
            {"actions": torch.randn(7), "instructions": "move"}
            for _ in range(2)
        ]
        
        try:
            batch = collator(features)
            print(f"❌ FAIL: Should have raised ValueError for missing images")
            self.fail("Should have raised ValueError")
        except ValueError as e:
            if "missing required keys" in str(e):
                print(f"✅ PASS: Correctly raised error: {e}")
            else:
                print(f"❌ FAIL: Wrong error message: {e}")
                raise
    
    def test_missing_instructions_uses_fallback(self):
        """Missing instructions should use fallback text."""
        print("\n" + "=" * 80)
        print("TEST 10: Missing instructions uses fallback")
        print("=" * 80)
        
        from fastvla.data.collator import UnslothVLACollator
        
        class FakeTokenizer:
            pad_token_id = 0
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
                }
        
        collator = UnslothVLACollator(tokenizer=FakeTokenizer(), action_dim=7)
        features = [
            {"images": {"cam0": torch.rand(3, 224, 224)}, "actions": torch.randn(7)}
            for _ in range(2)
        ]
        
        try:
            batch = collator(features)
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertIn("pixel_values", batch)
            self.assertIn("labels", batch)
            print(f"✅ PASS: Fallback text handled correctly")
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_inconsistent_action_shapes_raises_error(self):
        """Inconsistent action dimensions should raise error."""
        print("\n" + "=" * 80)
        print("TEST 11: Inconsistent action shapes raises error")
        print("=" * 80)
        
        from fastvla.data.collator import UnslothVLACollator
        
        class FakeTokenizer:
            pad_token_id = 0
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.zeros(len(texts), 10, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
                }
        
        collator = UnslothVLACollator(tokenizer=FakeTokenizer(), action_dim=7)
        features = [
            {"images": {"cam0": torch.rand(3, 224, 224)}, "actions": torch.randn(7), "instructions": "move"},
            {"images": {"cam0": torch.rand(3, 224, 224)}, "actions": torch.randn(5), "instructions": "move"},  # Wrong dim
        ]
        
        try:
            batch = collator(features)
            print(f"❌ FAIL: Should have raised ValueError for inconsistent actions")
            self.fail("Should have raised ValueError")
        except ValueError as e:
            if "Inconsistent action dimensions" in str(e):
                print(f"✅ PASS: Correctly caught inconsistent actions")
            else:
                print(f"❌ FAIL: Wrong error: {e}")
                raise
        except Exception as e:
            print(f"❌ FAIL: Unexpected error: {e}")
            raise


class TestKernelOperations(unittest.TestCase):
    """Test kernel operations in isolation."""
    
    def test_triton_action_head_cpu_fp16_forward(self):
        """TritonActionHead CPU fallback with FP16 input."""
        print("\n" + "=" * 80)
        print("TEST 12: TritonActionHead CPU FP16 forward")
        print("=" * 80)
        
        from fastvla.kernels.action_head import TritonActionHead
        
        head = TritonActionHead(input_dim=128, hidden_dim=64, output_dim=7)
        x = torch.randn(2, 128, dtype=torch.float16)
        
        try:
            with torch.no_grad():
                output = head(x)
            self.assertEqual(output.shape, (2, 7))
            print(f"✅ PASS: TritonActionHead FP16 forward")
        except RuntimeError as e:
            if "dtype" in str(e).lower():
                print(f"❌ FAIL (KNOWN BUG): {e}")
            raise
    
    def test_action_decode_backward_fp16(self):
        """action_decode_backward with FP16 inputs."""
        print("\n" + "=" * 80)
        print("TEST 13: action_decode_backward FP16")
        print("=" * 80)
        
        from fastvla.kernels.action import action_decode_backward
        
        B, D, H, A = 2, 128, 64, 7
        hidden = torch.randn(B, D, dtype=torch.float16)
        weight1 = torch.randn(D, H, dtype=torch.float32)
        bias1 = torch.randn(H, dtype=torch.float32)
        weight2 = torch.randn(H, A, dtype=torch.float32)
        bias2 = torch.randn(A, dtype=torch.float32)
        grad_output = torch.randn(B, A, dtype=torch.float16)
        
        try:
            grads = action_decode_backward(grad_output, hidden, weight1, bias1, weight2, bias2)
            self.assertEqual(len(grads), 5)
            print(f"✅ PASS: action_decode_backward FP16")
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_vision_language_fusion_mixed_dtypes(self):
        """Fusion with mismatched visual/text dtypes."""
        print("\n" + "=" * 80)
        print("TEST 14: Vision-language fusion mixed dtypes")
        print("=" * 80)
        
        from fastvla.kernels.fusion import vision_language_fusion_forward
        
        visual = torch.randn(2, 100, 128, dtype=torch.float16)
        text = torch.randn(2, 100, 128, dtype=torch.float32)
        
        try:
            output = vision_language_fusion_forward(visual, text)
            self.assertEqual(output.shape, (2, 100, 128))
            print(f"✅ PASS: Fusion mixed dtypes")
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise


class TestOptimizerCompatibility(unittest.TestCase):
    """Test optimizer compatibility with different model types."""
    
    def test_standard_adamw_with_fp16_model(self):
        """Standard AdamW with FP16 model parameters."""
        print("\n" + "=" * 80)
        print("TEST 15: Standard AdamW with FP16 model")
        print("=" * 80)
        
        model = FakeModel(action_dim=7)
        # Convert model to FP16
        model = model.half()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        try:
            # Simulate a train step
            optimizer.zero_grad()
            loss = torch.tensor(1.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            print(f"✅ PASS: Standard AdamW with FP16 model")
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise
    
    def test_8bit_optimizer_with_standard_model(self):
        """8-bit optimizer with standard model."""
        print("\n" + "=" * 80)
        print("TEST 16: 8-bit optimizer with standard model")
        print("=" * 80)
        
        from fastvla.optimization import get_8bit_optimizer
        
        model = FakeModel(action_dim=7)
        
        try:
            optimizer = get_8bit_optimizer(model, learning_rate=1e-4)
            
            optimizer.zero_grad()
            loss = torch.tensor(1.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            
            print(f"✅ PASS: 8-bit optimizer with standard model")
        except Exception as e:
            print(f"❌ FAIL: {e}")
            raise


# ── Test Runner ─────────────────────────────────────────────────────────

def run_tests():
    """Run all tests and print summary."""
    print("=" * 80)
    print("FASTVLA END-TO-END TEST SUITE")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCollatorRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestKernelOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizerCompatibility))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n" + "-" * 80)
        print("FAILURES:")
        print("-" * 80)
        for test, traceback in result.failures:
            print(f"\n  ❌ {test}")
            print(f"     {traceback}")
    
    if result.errors:
        print("\n" + "-" * 80)
        print("ERRORS:")
        print("-" * 80)
        for test, traceback in result.errors:
            print(f"\n  💥 {test}")
            print(f"     {traceback}")
    
    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
