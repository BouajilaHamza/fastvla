"""
Example training script for FastVLA.
Demonstrates how to use the FastVLA API for training on robotics tasks.
"""
import torch
from torch.utils.data import DataLoader
from fastvla import (
    FastVLAModel,
    FastVLAConfig,
    UnslothVLACollator,
    get_dataset,
    FastVLATrainer,
)


def main():
    """Main training function."""
    # Configuration
    config = FastVLAConfig(
        vision_encoder_name="google/vit-base-patch16-224",
        llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use smaller model for demo
        max_sequence_length=512,
        action_dim=7,
        load_in_4bit=True,
        use_peft=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    
    # Load model
    print("Loading model...")
    model = FastVLAModel.from_pretrained(
        config=config,
        load_in_4bit=True,
        gradient_checkpointing=True,
        use_peft=True,
    )
    
    # Load tokenizer
    tokenizer = model.tokenizer
    
    # Load dataset (example with LIBERO)
    print("Loading dataset...")
    dataset = get_dataset(
        dataset_name="libero",
        data_path="./data/libero",  # Update with your data path
        image_keys=["rgb"],
        state_key="state",
        action_key="action",
        instruction_key="instruction",
    )
    
    # Create data collator
    collator = UnslothVLACollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True,
    )
    
    # Create data loaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = FastVLATrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        use_8bit_optimizer=True,
        use_mixed_precision=True,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="./checkpoints",
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
    )
    
    # Train
    print("Starting training...")
    history = trainer.train(num_epochs=3, max_steps=1000)
    
    print("Training completed!")
    print(f"Training history: {history[-5:]}")  # Print last 5 entries


if __name__ == "__main__":
    main()

