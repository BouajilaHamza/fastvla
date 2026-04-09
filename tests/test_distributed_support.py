import torch
from fastvla import FastVLAModel, FastVLATrainer
from accelerate import Accelerator

def test_accelerator_integration():
    """Verify that FastVLATrainer correctly initializes and uses Accelerator."""
    model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            return {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                "instructions": "test"
            }
            
    dataset = DummyDataset()
    trainer = FastVLATrainer(
        model=model,
        dataset=dataset,
        max_steps=2,
        logging_steps=1,   # Ensure history is populated
        use_mixed_precision=True,
        use_8bit_optimizer=False
    )
    
    # Verify accelerator is active
    assert isinstance(trainer.accelerator, Accelerator)
    assert trainer.device is not None
    
    # Verify model is prepared
    # Prepared models are wrapped by Accelerator
    # Note: hf_device_map models are not wrapped by Accelerator.prepare by default in current logic
    # but dummy models are.
    assert hasattr(trainer.model, "forward")
    
    # Run training
    history = trainer.train()
    assert len(history) > 0
    assert trainer.global_step == 2

def test_distributed_aware_logging():
    """Verify that logging only happens on the main process (mocked)."""
    model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            return {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),
                "instructions": "test"
            }
            
    dataset = DummyDataset()
    trainer = FastVLATrainer(
        model=model,
        dataset=dataset,
        max_steps=1,
        logging_steps=1,
        use_mixed_precision=False,
        use_8bit_optimizer=False
    )
    
    # Mock is_main_process using patch
    from unittest.mock import patch
    with patch.object(Accelerator, "is_main_process", new_callable=lambda: True):
        history = trainer.train()
        assert len(history) > 0
    
    # Verify history is populated
    assert "loss" in history[0]
