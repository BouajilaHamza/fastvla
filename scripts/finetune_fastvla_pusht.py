"""
Fine-tune FastVLA (Our Optimized Model) on PushT.
This script uses our custom library components to beat the OpenVLA baseline.
"""

import argparse
import torch
import numpy as np
from fastvla.model import FastVLAModel
from fastvla.config import FastVLAConfig
from fastvla.training import FastVLATrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fastvla-pusht-optimized")
    args = parser.parse_args()

    print("\n🚀 FastVLA Optimized Fine-Tuning")
    print(f"   Steps: {args.steps} | Batch: {args.batch} | LR: {args.lr}")
    print(f"   Chunk Size: {args.chunk_size} | Loss: {args.loss}\n")

    # 1. Config with PushT stats
    config = FastVLAConfig(
        llm_name="openvla/openvla-7b",
        vision_encoder_name="openvla/openvla-7b",
        load_in_4bit=True,
        use_peft=True,
        action_dim=2, # PushT is 2D
        chunk_size=args.chunk_size,
        loss_type=args.loss,
        norm_min=[12.0, 25.0],
        norm_max=[511.0, 511.0],
    )

    # 2. Load Model
    model = FastVLAModel(config)

    # 3. Trainer
    trainer = FastVLATrainer(
        model=model,
        dataset="lerobot/pusht_image",
        lr=args.lr,
        batch_size=args.batch,
        max_steps=args.steps,
        output_dir=args.output_dir,
        use_8bit_optimizer=True,
        save_steps=200,
        logging_steps=10,
    )

    # 4. Train
    trainer.train()

    print(f"\n✅ Training complete. Checkpoints saved to {args.output_dir}")

if __name__ == "__main__":
    main()
