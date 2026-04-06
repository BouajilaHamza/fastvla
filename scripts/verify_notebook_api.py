import torch
import torch.nn as nn
from fastvla import FastVLAModel, FastVLATrainer, get_dataset

def test_notebook_replica():
    print("🧪 Running Notebook API Replica Test...")
    
    # 1. Load Dummy Model (matches notebook Step 2)
    print("Loading dummy model...")
    model = FastVLAModel.from_pretrained(
        dummy=True,
        load_in_4bit=True,
        use_peft=True,
        gradient_checkpointing=True,
    )
    
    # 2. Setup Dummy Dataset (matches notebook Step 3 call style)
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            return {
                "pixel_values": torch.randn(1, 1, 3, 224, 224), # [B, num_cam, C, H, W]
                "input_ids": torch.randint(0, 100, (1, 16)),
                "labels": torch.randn(1, 7)
            }
    
    dataset = DummyDataset()
    
    # 3. Create Trainer with ALL problematic arguments (matches notebook Step 3)
    print("Initializing Trainer with notebook arguments...")
    trainer = FastVLATrainer(
        model=model,
        dataset=dataset, # Was failing
        batch_size=1,
        lr=1e-4,         # Was failing
        max_steps=5      # Was missing in some versions
    )
    
    # 4. Run training (matches notebook Step 3)
    print("Starting Training test...")
    results = trainer.train(max_steps=2)
    print(f"✅ API Test Passed! Steps run: {len(results)}")
    
    # 5. Inference (matches notebook Step 4)
    print("Testing Inference...")
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    prompt = "push the block"
    action = model.predict(dummy_image, prompt)
    print(f"✅ Inference Passed! Action shape: {action.shape}")

if __name__ == "__main__":
    test_notebook_replica()
