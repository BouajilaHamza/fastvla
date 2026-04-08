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
from .data.collator import UnslothVLACollator
from .utils import get_device
from accelerate import Accelerator


def _detect_4bit_model(model: nn.Module) -> bool:
    """Reliable 4-bit model detection.

    Checks explicit quantization flags only. Does NOT scan parameter dtypes
    because standard FP16 models produce FP16 params without being quantized.

    Check order (first match wins):
    1. Wrapper flag: model.is_loaded_in_4bit
    2. Config flag: model.config.load_in_4bit
    3. Submodule flags: model.llm.is_loaded_in_4bit, etc.
    4. HF quantizer: model.hf_quantizer or submodule.hf_quantizer
    5. BitsAndBytes quantization_config on config
    """
    # 1. Wrapper flag (set by FastVLAModel.__init__)
    if getattr(model, "is_loaded_in_4bit", False):
        return True

    # 2. Config flag (FastVLAConfig.load_in_4bit)
    config = getattr(model, "config", None)
    if getattr(config, "load_in_4bit", False):
        return True

    # 3. Submodule flags (HuggingFace sets these on the inner model)
    for submodule_name in ("llm", "vision_encoder", "model", "base_model"):
        submodule = getattr(model, submodule_name, None)
        if submodule is not None:
            if getattr(submodule, "is_loaded_in_4bit", False):
                return True
            # HF quantizer flag
            if getattr(submodule, "hf_quantizer", None) is not None:
                return True

    # 4. HF quantizer on the wrapper itself
    if getattr(model, "hf_quantizer", None) is not None:
        return True

    # 5. BitsAndBytes quantization config (HF native 4-bit path)
    if config is not None:
        bnb_config = getattr(config, "quantization_config", None)
        if bnb_config is not None and getattr(bnb_config, "load_in_4bit", False):
            return True

    return False


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
        lr: Optional[float] = None,
        learning_rate: float = 5e-5,
        max_steps: Optional[int] = None,
        num_epochs: int = 1,
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
        # Prefer 'lr' over 'learning_rate' if both provided
        self.learning_rate = lr if lr is not None else learning_rate
        
        # Handle dataset aliases
        if train_dataset is None:
            train_dataset = dataset

        # ── Auto-initialize Data Collator ──────────────────────────────
        if data_collator is None:
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None:
                # Get action_dim from model config if available
                action_dim = getattr(model.config, "action_dim", 7)
                data_collator = UnslothVLACollator(
                    tokenizer=tokenizer,
                    action_dim=action_dim
                )
            else:
                print("⚠️ Warning: No tokenizer found on model. Using default collator.")

        # ── Initialize Accelerator ────────────────────────────────────
        # Bulletproof 4-bit detection (checks all possible locations)
        is_4bit_model = _detect_4bit_model(model)

        mixed_precision = "no"
        if use_mixed_precision and not is_4bit_model:
            if torch.cuda.is_available():
                # Use bf16 for Ampere+ GPUs (compute capability >= 8.0),
                # fp16 for older GPUs (T4, V100). bf16 avoids gradient
                # overflow issues that plague fp16 on newer hardware.
                if torch.cuda.get_device_capability()[0] >= 8:
                    mixed_precision = "bf16"
                else:
                    mixed_precision = "fp16"
            else:
                mixed_precision = "no"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        self.device = self.accelerator.device
        self.is_4bit_model = is_4bit_model  # Store for use in train_step

        # Print clear status so the user knows exactly what's happening
        if is_4bit_model:
            print(f"\nFastVLATrainer: Mixed precision DISABLED (4-bit model detected)")
            print(f"  → Using direct gradient clipping (no GradScaler)")
            print(f"  → 4-bit models produce FP16 gradients which GradScaler cannot handle\n")
        elif use_mixed_precision and torch.cuda.is_available():
            print(f"\nFastVLATrainer: Mixed precision ENABLED (fp16)")
            print(f"  → Using GradScaler for gradient clipping\n")
        else:
            print(f"\nFastVLATrainer: Full precision mode\n")

        # Setup optimizer
        if optimizer is None:
            if use_8bit_optimizer:
                optimizer = get_8bit_optimizer(model, learning_rate=self.learning_rate)
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=0.01,
                )
        self.optimizer = optimizer

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

        # ── Prepare for Distributed Training ──────────────────────────
        # Only skip Accelerator wrapping if the model is genuinely dispatched
        # across devices (hf_device_map) or is actually 4-bit (not just has the attribute).
        if hasattr(model, "hf_device_map") or getattr(model, "is_loaded_in_4bit", False):
            self.model = model
        else:
            self.model = self.accelerator.prepare(model)
            
        self.optimizer = self.accelerator.prepare(self.optimizer)
        
        if train_dataloader is not None:
            self.train_dataloader = self.accelerator.prepare(train_dataloader)
        else:
            self.train_dataloader = None
        
        if eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)
        else:
            self.eval_dataloader = None

        if lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler)
        else:
            self.lr_scheduler = None

        if self.train_dataloader is None:
             raise ValueError("You must provide either 'train_dataloader' or 'train_dataset'.")
             
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm 
        self.gradient_accumulation_steps = gradient_accumulation_steps # Restore missing attribute
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        # Forward pass (Mixed precision handled by Accelerator)
        with self.accelerator.accumulate(self.model):
            action_preds, loss = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )

            # Backward pass
            self.accelerator.backward(loss)

            # Step optimizer & scheduler only when gradients are synchronized
            if self.accelerator.sync_gradients:
                # CRITICAL: 4-bit models have FP16 gradients which GradScaler cannot handle
                # Use direct clip_grad_norm_ for 4-bit models to avoid "Attempting to unscale FP16 gradients" error
                if self.is_4bit_model:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                else:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
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

    def train(
        self,
        num_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.model.train()
        
        # Priority: method argument > __init__ attribute > default (1 for epochs)
        if num_epochs is None:
            num_epochs = self.num_epochs if self.num_epochs is not None else 1
        if max_steps is None:
            max_steps = self.max_steps

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

                if self.accelerator.is_main_process and self.global_step % self.logging_steps == 0:
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
                    if self.accelerator.is_main_process:
                        print(
                            f"\nStep {self.global_step} - Eval Loss: {eval_metrics.get('eval_loss', 0.0):.4f}"
                        )
                        self.training_history[-1].update(eval_metrics)

                if self.accelerator.is_main_process and self.global_step % self.save_steps == 0:
                    self.save_checkpoint()

                if max_steps is not None and self.global_step >= max_steps:
                    break

            if self.accelerator.is_main_process:
                self.save_checkpoint()

            if max_steps is not None and self.global_step >= max_steps:
                break

        print("Training completed!")
        return self.training_history
