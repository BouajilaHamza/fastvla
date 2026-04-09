import torch
import pytest
from fastvla import FastVLAModel, FastVLATrainer
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __len__(self): return 10
    def __getitem__(self, idx):
        return {
            "pixel_values": torch.randn(1, 1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 10)),
            "actions": torch.randn(2)
        }

def test_positional_from_pretrained():
    """Verify that FastVLAModel supports positional model_id."""
    # This should not raise TypeError
    model = FastVLAModel.from_pretrained("smolvla", dummy=True)
    assert model.config.llm_name == "HuggingFaceTB/SmolLM2-135M-Instruct"

def test_trainer_aliases():
    """Verify that FastVLATrainer supports 'dataset' and 'lr' aliases."""
    model = FastVLAModel.from_pretrained(dummy=True)
    dataset = DummyDataset()
    
    # This should not raise TypeError or ValueError
    trainer = FastVLATrainer(
        model=model,
        dataset=dataset,
        lr=2e-4
    )
    
    assert trainer.optimizer.param_groups[0]["lr"] == 2e-4
    assert len(trainer.train_dataloader) > 0

def test_hook_removal_mechanism():
    """Verify that stabilize_distributed_hooks doesn't crash on dummy model."""
    model = FastVLAModel.from_pretrained(dummy=True)
    # Trigger hook removal logic manually
    model._stabilize_distributed_hooks() 
    # Should complete without error
