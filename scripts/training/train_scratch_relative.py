#!/usr/bin/env python3
"""
Scratch relative BC fine-tuning entrypoint script for FastVLA.
Matches the README Quickstart invocation:
    modal run scripts/training/train_scratch_relative.py --bc-epochs 10
Can also be run locally directly via Python / CLI.
"""

import os
import argparse
import logging
import torch

from fastvla import FastVLAModel, FastVLATrainer, get_dataset
from fastvla.config import FastVLAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="FastVLA BC Fine-Tuning")
    parser.add_argument("--model-name", type=str, default="openvla-7b", help="Model registry preset or path")
    parser.add_argument("--dataset-name", type=str, default="pusht", help="Dataset name or path")
    parser.add_argument("--bc-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--dummy", action="store_true", help="Use lightweight dummy backbone for quick testing")
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB tracking")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting FastVLA BC Fine-Tuning on {args.dataset_name} for {args.bc_epochs} epochs...")

    # Configure FastVLA model
    if args.dummy or not torch.cuda.is_available():
        logger.info("Initializing in dummy/CPU mode...")
        model = FastVLAModel.from_pretrained(dummy=True, action_dim=7, chunk_size=1)
    else:
        model = FastVLAModel.from_pretrained(args.model_name, action_dim=7, chunk_size=1)

    # Initialize dataset
    try:
        train_dataset = get_dataset(args.dataset_name)
    except Exception as e:
        logger.warning(f"Failed loading {args.dataset_name} ({e}), creating dummy dataset for execution test...")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 16
            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "states": torch.randn(7),
                    "actions": torch.randn(7),
                    "instructions": "push block to target",
                }
        train_dataset = DummyDataset()

    trainer = FastVLATrainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.bc_epochs,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb and bool(os.environ.get("WANDB_API_KEY")),
    )

    history = trainer.train()
    logger.info("BC Fine-tuning completed successfully.")
    return history

if __name__ == "__main__":
    main()
