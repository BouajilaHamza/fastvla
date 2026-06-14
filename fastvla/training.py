import torch
import torch.nn as nn
import logging
import json
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Union
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator

from .optimization import get_8bit_optimizer
from .data.collator import UnslothVLACollator
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """Handles data manipulation, including instruction translation."""

    def __init__(
        self, translation_mapping: Optional[Union[str, Dict[str, str]]] = None
    ):
        self.translation_mapping = translation_mapping
        if isinstance(translation_mapping, str):
            with open(translation_mapping, "r", encoding="utf-8") as f:
                self.translation_mapping = json.load(f)

    def process_batch(
        self, batch: Dict[str, Any], tokenizer: Any, device: torch.device
    ) -> Dict[str, Any]:
        if self.translation_mapping and "instructions" in batch:
            translated = [
                self.translation_mapping.get(inst, inst)
                for inst in batch["instructions"]
            ]
            if tokenizer:
                new_tokens = tokenizer(
                    translated,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                batch["input_ids"] = new_tokens["input_ids"]
                batch["attention_mask"] = new_tokens.get("attention_mask")
        return batch


class CheckpointManager:
    """Handles saving and loading of model weights and trainer state."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self, step: int, global_step: int, model: nn.Module, accelerator: Accelerator
    ):
        if not accelerator.is_main_process and not getattr(
            model.config, "dummy", False
        ):
            return

        path = self.output_dir / f"checkpoint-{step}"
        path.mkdir(parents=True, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "save_pretrained"):
            try:
                unwrapped_model.save_pretrained(path, safe_serialization=True)
            except Exception as e:
                logger.warning(
                    f"save_pretrained failed, falling back to torch.save: {e}"
                )
                torch.save(unwrapped_model.state_dict(), path / "pytorch_model.bin")
        else:
            torch.save(unwrapped_model.state_dict(), path / "pytorch_model.bin")

        torch.save(
            {"step": step, "global_step": global_step}, path / "trainer_state.pt"
        )
        logger.info(f"Checkpoint saved to {path}")

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        accelerator: Accelerator,
    ) -> int:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path {checkpoint_path} does not exist."
            )

        unwrapped_model = accelerator.unwrap_model(model)
        weight_files = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.pt",
            "adapter_model.bin",
        ]
        found_file = next(
            (
                checkpoint_path / wf
                for wf in weight_files
                if (checkpoint_path / wf).exists()
            ),
            None,
        )

        if found_file:
            logger.info(f"Loading weights from {found_file}...")
            if found_file.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(found_file, device="cpu")
            else:
                state_dict = torch.load(found_file, map_location="cpu")

            try:
                unwrapped_model.load_state_dict(state_dict, strict=False)
            except Exception:
                if hasattr(unwrapped_model, "load_adapter"):
                    unwrapped_model.load_adapter(checkpoint_path, "default")
                else:
                    unwrapped_model.load_state_dict(state_dict, strict=False)
        else:
            if hasattr(unwrapped_model, "from_pretrained"):
                unwrapped_model.from_pretrained(checkpoint_path)

        state_path = checkpoint_path / "trainer_state.pt"
        global_step = 0
        if state_path.exists():
            state = torch.load(state_path)
            global_step = state.get("global_step", 0)

        logger.info(f"Checkpoint loading complete. Resumed at step {global_step}")
        return global_step


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
        num_workers: int = 4,
        pin_memory: bool = True,
        max_grad_norm: float = 1.0,
        output_dir: str = "./checkpoints",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        translation_mapping: Optional[Union[str, Dict[str, str]]] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, str]] = None,
        lr: Optional[float] = None,
        use_wandb: bool = False,
        wandb_project: str = "fastvla",
        checkpoint_callback: Optional[callable] = None,
    ):
        if train_dataset is None:
            train_dataset = dataset
        if lr is not None:
            learning_rate = lr

        self.checkpoint_callback = checkpoint_callback
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb
            import os

            wandb.init(
                project=wandb_project,
                entity=os.environ.get("WANDB_ENTITY"),
                config={
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                },
                reinit=True
            )
            if wandb.run:
                print(f"📊 WandB Run started: {wandb.run.get_url()}")

        # Resolve dataset if string provided
        from .data.datasets import get_dataset

        self.is_4bit = getattr(model, "is_quantized", False) or "4bit" in str(getattr(model, "config", ""))
        
        image_size = getattr(model.config, "image_size", 224)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        dataset_kwargs = {
            "chunk_size": getattr(model.config, "chunk_size", 1),
            "image_size": image_size,
            "hf_token": getattr(model.config, "hf_token", None),
        }
        if isinstance(train_dataset, str):
            train_dataset = get_dataset(train_dataset, **dataset_kwargs)
        if isinstance(eval_dataset, str):
            eval_dataset = get_dataset(eval_dataset, **dataset_kwargs)

        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.warmup_steps = getattr(model.config, "warmup_steps", 1000)
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # DataLoader parallelism. The default single-process loader (num_workers=0)
        # serializes image decoding/resizing on the main process and starves the
        # GPU; overlap it with worker processes and pinned host memory.
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        mixed_precision = None
        if use_mixed_precision and not self.is_4bit:
            if torch.cuda.is_available():
                mixed_precision = (
                    "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
                )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        self.model = (
            model
            if self.is_4bit or hasattr(model, "hf_device_map")
            else self.accelerator.prepare(model)
        )

        if data_collator is None:
            data_collator = UnslothVLACollator(
                tokenizer=getattr(model, "tokenizer", None),
                action_dim=getattr(model.config, "action_dim", 7),
                image_size=getattr(model.config, "image_size", 224),
                chunk_size=getattr(model.config, "chunk_size", 1),
                use_relative_delta=getattr(model.config, "use_relative_delta", False),
                norm_min=torch.tensor(model.config.norm_min)
                if getattr(model.config, "norm_min", None)
                else None,
                norm_max=torch.tensor(model.config.norm_max)
                if getattr(model.config, "norm_max", None)
                else None,
            )

        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
        }

        if train_dataloader is None and train_dataset is not None:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data_collator,
                **loader_kwargs,
            )

        if train_dataloader is None:
            raise ValueError("No training data provided.")
        self.train_dataloader = self.accelerator.prepare(train_dataloader)

        if eval_dataloader is None and eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
                **loader_kwargs,
            )
        self.eval_dataloader = (
            self.accelerator.prepare(eval_dataloader) if eval_dataloader else None
        )

        weight_decay = getattr(model.config, "weight_decay", 0.01)
        if use_8bit_optimizer:
            base_optimizer = get_8bit_optimizer(
                model, learning_rate=learning_rate, weight_decay=weight_decay
            )
        else:
            base_optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        self.optimizer = self.accelerator.prepare(base_optimizer)

        # Setup Scheduler
        total_steps = max_steps if max_steps else (len(self.train_dataloader) * num_epochs)
        self.scheduler = self.accelerator.prepare(
            get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        )

        self.max_steps, self.num_epochs, self.max_grad_norm = (
            max_steps,
            num_epochs,
            max_grad_norm,
        )
        self.save_steps, self.eval_steps, self.logging_steps = (
            save_steps,
            eval_steps,
            logging_steps,
        )
        self.global_step = 0
        self.training_history = []

        # Delegated Concerns (SOLID)
        self.data_orchestrator = DataOrchestrator(translation_mapping)
        self.checkpoint_manager = CheckpointManager(output_dir)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def translation_mapping(self):
        return self.data_orchestrator.translation_mapping

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = self.data_orchestrator.process_batch(
            batch, getattr(self.model, "tokenizer", None), self.accelerator.device
        )

        with self.accelerator.accumulate(self.model):
            _, loss = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.is_4bit:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                else:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}

    def evaluate(self) -> Dict[str, float]:
        if not self.eval_dataloader:
            return {}
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_main_process,
            ):
                _, loss = self.model(**batch)
                total_loss += loss.item()
        return {"eval_loss": total_loss / len(self.eval_dataloader)}

    def save_checkpoint(self, step: int):
        self.checkpoint_manager.save(
            step, self.global_step, self.model, self.accelerator
        )
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(self.model, step, phase="bc")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        self.global_step = self.checkpoint_manager.load(
            checkpoint_path, self.model, self.accelerator
        )

    def train(self):
        logger.info(f"Starting Training: {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                disable=not self.accelerator.is_main_process,
            )
            for batch in progress_bar:
                metrics = self.train_step(batch)
                self.global_step += 1

                if self.global_step % self.logging_steps == 0:
                    progress_bar.set_postfix(metrics)
                    if self.accelerator.is_main_process:
                        self.training_history.append(
                            {"step": self.global_step, **metrics}
                        )
                        if self.use_wandb:
                            import wandb

                            wandb.log(metrics, step=self.global_step)

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
