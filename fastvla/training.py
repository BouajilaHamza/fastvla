"""
Training loop for FastVLA models.
Refactored for distributed training stability, structured logging,
and robust precision management.
"""

import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator

from .optimization import get_8bit_optimizer
from .data.collator import UnslothVLACollator

# Setup logging
logger = logging.getLogger(__name__)

class FastVLATrainer:
    """
    Production-grade trainer for FastVLA.
    Handles DDP, Mixed Precision, and 4-bit optimization automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collator: Optional[Any] = None,
        batch_size: int = 1,
        learning_rate: float = 5e-5,
        max_steps: Optional[int] = None,
        num_epochs: int = 1,
        use_8bit_optimizer: bool = True,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "./checkpoints",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ):
        # 1. 4-bit & Distributed Detection
        self.is_4bit = getattr(model, "is_loaded_in_4bit", False)
        
        # 2. Initialize Accelerator
        mixed_precision = "no"
        if use_mixed_precision and not self.is_4bit:
            if torch.cuda.is_available():
                mixed_precision = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        
        # 3. Model Preparation
        # For 4-bit, we skip accelerator.prepare(model) to avoid hook conflicts
        if self.is_4bit or hasattr(model, "hf_device_map"):
            self.model = model
            logger.info("4-bit or Sharded model detected; skipping Accelerator model wrapping.")
        else:
            self.model = self.accelerator.prepare(model)

        # 4. Data Setup
        if data_collator is None:
            tokenizer = getattr(model, "tokenizer", None)
            action_dim = getattr(model.config, "action_dim", 7)
            data_collator = UnslothVLACollator(tokenizer=tokenizer, action_dim=action_dim)

        if train_dataloader is None and train_dataset is not None:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        
        if train_dataloader is not None:
            self.train_dataloader = self.accelerator.prepare(train_dataloader)
        else:
            raise ValueError("No training data provided.")

        if eval_dataloader is None and eval_dataset is not None:
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        
        self.eval_dataloader = self.accelerator.prepare(eval_dataloader) if eval_dataloader else None

        # 5. Optimizer
        self.optimizer = get_8bit_optimizer(model, learning_rate=learning_rate) if use_8bit_optimizer else \
                         torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        # 6. Attributes
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.global_step = 0
        self.training_history = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        
        # Consistent with Accelerate's recommended pattern for DDP
        with self.accelerator.accumulate(self.model):
            action_preds, loss = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                # 4-bit gradients need specific handling to avoid GradScaler unscaling errors
                if self.is_4bit:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def evaluate(self) -> Dict[str, float]:
        if not self.eval_dataloader:
            return {}
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                _, loss = self.model(**batch)
                total_loss += loss.item()
        return {"eval_loss": total_loss / len(self.eval_dataloader)}

    def save_checkpoint(self, step: int):
        if not self.accelerator.is_main_process:
            return
        
        path = self.output_dir / f"checkpoint-{step}"
        path.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(path)
        else:
            torch.save(unwrapped_model.state_dict(), path / "pytorch_model.bin")
            
        logger.info(f"Checkpoint saved to {path}")

    def train(self):
        logger.info(f"Starting Training: {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)
            
            for batch in progress_bar:
                metrics = self.train_step(batch)
                self.global_step += 1
                
                if self.global_step % self.logging_steps == 0:
                    progress_bar.set_postfix(metrics)
                    if self.accelerator.is_main_process:
                        self.training_history.append({"step": self.global_step, **metrics})
                
                if self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if self.accelerator.is_main_process:
                        logger.info(f"Step {self.global_step}: {eval_metrics}")
                
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(self.global_step)
                
                if self.max_steps and self.global_step >= self.max_steps:
                    break
            
            if self.max_steps and self.global_step >= self.max_steps:
                break

        self.save_checkpoint(self.global_step)
        logger.info("Training Complete.")
        return self.training_history
