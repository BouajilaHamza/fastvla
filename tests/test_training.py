import torch
from fastvla import FastVLAModel, FastVLATrainer

def test_trainer_initialization():
    """Test that the trainer can be initialized with different argument styles."""
    model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 5
        def __getitem__(self, idx):
            return {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                "instructions": "test"
            }
            
    dataset = DummyDataset()
    
    # ── Test with 'train_dataset' and 'learning_rate' (Standard style) ──────
    trainer = FastVLATrainer(
        model=model,
        train_dataset=dataset,
        learning_rate=1e-4,
        batch_size=1,
        max_steps=1
    )
    
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-4
    assert trainer.train_dataloader is not None
    assert trainer.max_steps == 1

def test_trainer_train_step():
    """Test one single training step with the auto-initialized collator."""
    # Use a vocab size that matches our GPT2 dummy tokenizer to avoid IndexError
    model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 5
        def __getitem__(self, idx):
            return {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                "instructions": "test"
            }
            
    dataset = DummyDataset()
    trainer = FastVLATrainer(
        model=model,
        train_dataset=dataset,
        max_steps=1
    )
    
    # Get one batch from internal dataloader
    batch = next(iter(trainer.train_dataloader))
    
    # Verify keys from UnslothVLACollator
    assert "pixel_values" in batch
    assert "labels" in batch
    assert "input_ids" in batch
    
    # Run step
    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)

def test_trainer_convergence_dummy():
    """Verify the train() method doesn't crash."""
    model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx):
            return {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                "instructions": "test"
            }
            
    dataset = DummyDataset()
    trainer = FastVLATrainer(model=model, train_dataset=dataset, max_steps=1, logging_steps=1)
    
    history = trainer.train()
    assert len(history) > 0
