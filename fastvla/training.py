"""
Training loop for FastVLA models.
Includes training, evaluation, and checkpointing utilities.
CPU/GPU auto-selected.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from tqdm import tqdm
from pathlib import Path
import json
from .optimization import get_8bit_optimizer
from .utils import get_device


class FastVLATrainer:
    """
    Trainer for FastVLA models with Unsloth-style optimizations.
    Auto-detects CPU/GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        dataset: Optional[torch.utils.data.Dataset] = None, # Alias for train_dataset
        data_collator: Optional[Any] = None,
        batch_size: int = 1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_8bit_optimizer: bool = True,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
        output_dir: str = "./checkpoints",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ):
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        
        # Handle dataset aliases
        if train_dataset is None:
            train_dataset = dataset
            
        # Auto-create dataloaders if datasets are provided
        if train_dataloader is None and train_dataset is not None:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data_collator
            )
        
        if eval_dataloader is None and eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator
            )

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        if self.train_dataloader is None:
            raise ValueError("You must provide either 'train_dataloader' or 'train_dataset'.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        if optimizer is None:
            if use_8bit_optimizer:
                self.optimizer = get_8bit_optimizer(model, learning_rate=5e-5)
            else:
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=5e-5,
                    weight_decay=0.01,
                )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        # Mixed precision only meaningful on GPU
        self.use_mixed_precision = use_mixed_precision and self.device == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = []

        # Setup mixed precision scaler (GPU only)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                action_preds, loss = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
        else:
            action_preds, loss = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )

        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.optimizer.zero_grad()

        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                action_preds, loss = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )

                total_loss += loss.item()
                num_samples += batch["pixel_values"].size(0)

        avg_loss = (
            total_loss / len(self.eval_dataloader)
            if len(self.eval_dataloader) > 0
            else 0.0
        )

        return {
            "eval_loss": avg_loss,
            "eval_samples": num_samples,
        }

    def save_checkpoint(self, step: Optional[int] = None):
        if step is None:
            step = self.global_step

        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "pytorch_model.bin")

        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "training_history": self.training_history,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}")

    def train(self, num_epochs: int = 1, max_steps: Optional[int] = None):
        self.model.train()

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                self.global_step += 1

                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{metrics['learning_rate']:.2e}",
                        }
                    )
                    self.training_history.append(
                        {
                            "step": self.global_step,
                            "loss": avg_loss,
                            "learning_rate": metrics["learning_rate"],
                        }
                    )

                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    print(
                        f"\nStep {self.global_step} - Eval Loss: {eval_metrics.get('eval_loss', 0.0):.4f}"
                    )
                    self.training_history[-1].update(eval_metrics)

                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()

                if max_steps is not None and self.global_step >= max_steps:
                    break

            self.save_checkpoint()

            if max_steps is not None and self.global_step >= max_steps:
                break

        print("Training completed!")
        return self.training_history
