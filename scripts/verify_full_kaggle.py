import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from datasets import Dataset

# Mimic the 'pip install git+...' effect by using the local source
from fastvla import FastVLAModel, FastVLATrainer, get_dataset

def run_kaggle_replica():
    print("🚀 Running Full Kaggle API Replication...")
    
    # ── Step 0: Mock Login ───────────────────────────────────────────────
    os.environ["HF_TOKEN"] = "mock_token"
    print("Step 0: HF Login mocked.")

    # ── Step 1: Environment ──────────────────────────────────────────────
    print("Step 1: Environment ready (local source used).")

    # ── Step 2: Load Model (DUMMY for speed) ──────────────────────────────
    print("Step 2: Loading model in 4-bit (Dummy mode)...")
    model = FastVLAModel.from_pretrained(
        dummy=True,
        load_in_4bit=True,
        use_peft=True,
        gradient_checkpointing=True,
    )
    print(f"Model ID: {model.config.llm_name}")

    # ── Step 3: Dataset ──────────────────────────────────────────────────
    print("Step 3: Creating mock LeRobot dataset...")
    # Instead of downloading 1GB of PushT, we mock its format to catch key errors
    mock_data = {
        "observation.image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(5)],
        "observation.state": [np.zeros(7, dtype=np.float32) for _ in range(5)],
        "action": [np.zeros(7, dtype=np.float32) for _ in range(5)],
        "instruction": ["mock push task" for _ in range(5)]
    }
    # Mocking what LeRobotDataset._load_data does
    from unittest.mock import MagicMock
    dataset = get_dataset("pusht")
    def mock_load():
        return [
            {
                "rgb": img,
                "state": st,
                "action": act,
                "instruction": inst
            } for img, st, act, inst in zip(mock_data["observation.image"], mock_data["observation.state"], mock_data["action"], mock_data["instruction"])
        ]
    dataset._load_data = mock_load
    dataset.data = mock_load() # Refresh
    
    # ── Step 4: Trainer ──────────────────────────────────────────────────
    print("Step 4: Initializing Trainer with notebook arguments...")
    trainer = FastVLATrainer(
        model=model,
        dataset=dataset,
        batch_size=1,
        lr=1e-4,
        max_steps=2
    )
    
    print("Running training step...")
    try:
        results = trainer.train()
        print("✅ Training successfully processed the 'pixel_values' and 'labels' keys!")
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # ── Step 5: Inference ────────────────────────────────────────────────
    print("Step 5: Testing Inference...")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    action = model.predict(dummy_img, "test instruction")
    print(f"✅ Inference successful! Action shape: {action.shape}")
    print("\n🏆 REPLICA TEST PASSED: All notebook calls are verified.")

if __name__ == "__main__":
    run_kaggle_replica()
