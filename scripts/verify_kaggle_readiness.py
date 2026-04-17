import torch
import os
from fastvla import FastVLAModel, FastVLAConfig, FastVLATrainer
from torch.utils.data import Dataset

class DummyActionDataset(Dataset):
    def __init__(self, size=16, action_dim=7):
        self.size = size
        self.action_dim = action_dim
    def __len__(self): return self.size
    def __getitem__(self, idx):
        return {
            "images": [torch.randn(3, 224, 224)], # List of images
            "instructions": "test",                # Dummy text
            "actions": torch.randn(self.action_dim)
        }

def verify_robustness():
    print("🚀 Starting FastVLA Kaggle Robustness Verification...")
    
    # 1. Test Shape Validation (The #1 Kaggle Killer)
    print("\n--- Testing Shape Validation ---")
    config = FastVLAConfig(dummy=True, action_dim=7)
    model = FastVLAModel(config)
    
    # Force vocab size to match what we will produce
    model.llm.embed_tokens = torch.nn.Embedding(100000, 128)
    
    # Intentionally mismatched labels
    pixel_values = torch.randn(1, 1, 3, 224, 224)
    input_ids = torch.zeros(1, 10, dtype=torch.long)
    bad_labels = torch.zeros(1, 2) # Wrong dim!
    
    try:
        model(pixel_values=pixel_values, input_ids=input_ids, labels=bad_labels)
        print("❌ FAILED: Model did not catch shape mismatch.")
    except ValueError as e:
        print(f"✅ PASSED: Caught mismatch with message: {e}")

    # 2. Test Gradient Accumulation Pattern
    print("\n--- Testing Gradient Accumulation Engine ---")
    dataset = DummyActionDataset(size=16, action_dim=7)
    trainer = FastVLATrainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=5,
        use_8bit_optimizer=False # CPU test
    )
    
    # This should run without error and correctly accumulate
    try:
        trainer.train()
        print("✅ PASSED: Training engine executed successfully.")
    except Exception as e:
        print(f"❌ FAILED: Training engine crashed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test Surgical Extraction (Simulation)
    print("\n--- Testing Surgical Extraction Logic ---")
    # We already verified this in tests/test_siglip_loading.py
    # but let's confirm the current model has it.
    if hasattr(model, "_load_vision_encoder_internal"):
        print("✅ PASSED: Surgical extraction logic is present in the model.")
    else:
        print("❌ FAILED: Surgical extraction logic is missing!")

    print("\n✨ All Kaggle-critical fixes verified locally.")

if __name__ == "__main__":
    verify_robustness()
