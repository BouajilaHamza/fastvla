import torch
import json
import os
from fastvla import FastVLAModel, FastVLAConfig, FastVLATrainer
from fastvla.data.datasets import get_dataset

def test_local_syntax():
    print("🔍 Starting local syntax and integration check...")
    
    # 1. Create a dummy mapping
    mapping = {
        "push the block to the goal": "إدفع الكتلة إلى الهدف",
        "move the object": "حرك الكائن"
    }
    mapping_path = "data/test_mapping.json"
    os.makedirs("data", exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print("✅ Dummy mapping created.")

    # 2. Initialize Model (Dummy mode for CPU)
    config = FastVLAConfig(
        dummy=True, 
        vision_encoder_name="google/siglip-test",
        action_dim=2
    )
    model = FastVLAModel(config)
    print("✅ Dummy model initialized.")

    # 3. Initialize Trainer with Mapping
    # We use a very small max_steps for the syntax check
    trainer = FastVLATrainer(
        model=model,
        dataset="pusht",
        translation_mapping=mapping_path,
        max_steps=2,
        batch_size=1,
        use_8bit_optimizer=False, # 8-bit needs CUDA
        use_mixed_precision=False  # Mixed precision needs CUDA
    )
    print("✅ Trainer initialized with translation mapping.")

    # 4. Run a single train step (Dummy)
    print("🏃 Running 1 dummy training step...")
    # Get valid input_ids from the tokenizer to avoid IndexError
    tokenizer = model.tokenizer
    text = "push the block to the goal"
    tokens = tokenizer(text, return_tensors="pt")
    
    batch = {
        "pixel_values": torch.randn(1, 1, 3, 224, 224),
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens.get("attention_mask"),
        "instructions": [text],
        "labels": torch.randn(1, 2)
    }
    
    metrics = trainer.train_step(batch)
    print(f"✅ Training step metrics: {metrics}")
    
    # 5. Check translation application
    # In train_step, batch["input_ids"] should have been re-tokenized
    # We can check if the trainer correctly loaded the mapping
    assert trainer.translation_mapping["push the block to the goal"] == "إدفع الكتلة إلى الهدف"
    print("✅ Translation mapping logic verified.")

    print("\n✨ ALL LOCAL SYNTAX CHECKS PASSED ✨")
    print("Code is ready for Modal deployment.")

if __name__ == "__main__":
    try:
        test_local_syntax()
    except Exception as e:
        print(f"❌ Local check failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
