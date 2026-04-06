"""
Example: FastVLA Distributed Training on Kaggle (2x T4 GPUs)

This script demonstrates the correct way to set up distributed training
for Kaggle's dual T4 GPU environment.

Key points:
1. Action dimensions MUST match between model and dataset
2. Let Accelerator handle device placement (don't do it manually)
3. Use mixed precision (fp16) for T4 GPUs
4. Use 8-bit optimizer for memory efficiency
"""

import torch
from torch.utils.data import Dataset
from fastvla import FastVLAModel, FastVLATrainer


class MyRoboticsDataset(Dataset):
    """
    Example robotics dataset with 7-dimensional actions.
    
    IMPORTANT: action_dim MUST match the model's action_dim config!
    """
    
    def __init__(self, data_path, num_samples=100):
        self.data_path = data_path
        self.num_samples = num_samples
        # This MUST match your model's action_dim
        self.action_dim = 7
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Example: Replace with your actual data loading logic
        return {
            # Single camera image [C, H, W]
            "images": {
                "rgb": torch.randn(3, 224, 224)
            },
            # Action vector [action_dim] - MUST be 7D!
            "actions": torch.randn(self.action_dim),
            # Text instruction
            "instructions": "pick up the object and place it in the drawer"
        }


def train_on_kaggle():
    """Main training function for Kaggle environment."""
    
    print("=" * 80)
    print("FastVLA Distributed Training - Kaggle Setup")
    print("=" * 80)
    
    # ── Configuration ──────────────────────────────────────────────
    ACTION_DIM = 7  # Change this to match your dataset!
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 2  # Effective batch size = 8
    MAX_STEPS = 1000
    LEARNING_RATE = 5e-5
    
    print(f"\nConfiguration:")
    print(f"  Action Dimension: {ACTION_DIM}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Max Steps: {MAX_STEPS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ── Load Model ─────────────────────────────────────────────────
    print("\nLoading model...")
    
    # For testing - use dummy model
    # Replace with your actual model loading
    model = FastVLAModel.from_pretrained(
        dummy=True,  # Remove for real training
        vocab_size=50257,
        action_dim=ACTION_DIM,  # MUST match dataset!
        max_sequence_length=512,
    )
    
    # For real training, use:
    # model = FastVLAModel.from_pretrained(
    #     vision_encoder_name="google/vit-base-patch16-224",
    #     llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     action_dim=ACTION_DIM,
    #     load_in_4bit=True,
    #     use_peft=True,
    #     lora_rank=16,
    #     lora_alpha=32,
    # )
    
    print("✓ Model loaded successfully")
    
    # ── Load Dataset ───────────────────────────────────────────────
    print("\nLoading dataset...")
    
    train_dataset = MyRoboticsDataset(
        data_path="/path/to/your/data",
        num_samples=1000
    )
    
    eval_dataset = MyRoboticsDataset(
        data_path="/path/to/your/data",
        num_samples=100
    )
    
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    # ── Create Trainer ─────────────────────────────────────────────
    print("\nCreating trainer...")
    
    trainer = FastVLATrainer(
        model=model,
        dataset=train_dataset,  # or train_dataset
        eval_dataset=eval_dataset,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_steps=MAX_STEPS,
        # Distributed training settings
        use_mixed_precision=True,  # Auto-uses fp16 for T4
        use_8bit_optimizer=True,   # Memory efficient
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_grad_norm=1.0,
        # Checkpointing
        output_dir="./checkpoints",
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
    )
    
    print("✓ Trainer created")
    
    # ── Train ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")
    
    results = trainer.train()
    
    # ── Results ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nTotal steps: {trainer.global_step}")
    print(f"Final loss: {results[-1]['loss']:.4f}" if results else "No results")
    print(f"Checkpoints saved to: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    results = train_on_kaggle()
