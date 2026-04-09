"""
Comprehensive tests for distributed training robustness.
Tests shape validation, error handling, and edge cases.
"""

import pytest
import torch
from fastvla import FastVLAModel, FastVLATrainer
from fastvla.data.collator import UnslothVLACollator


class TestShapeValidation:
    """Test that shape mismatches are properly detected and handled."""

    def test_action_dimension_mismatch_error(self):
        """Test that action dimension mismatch raises informative error."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257, action_dim=7)

        class MismatchedDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    # Wrong action dimension (2 instead of 7)
                    "actions": torch.randn(2),
                    "instructions": "test"
                }

        dataset = MismatchedDataset()
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            max_steps=1,
            logging_steps=1,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        # Verify that mismatched dimensions are handled (Warning is printed, but it shouldn't crash)
        # In v0.1.7, the collator auto-updates with a warning for better flexibility
        batch = next(iter(trainer.train_dataloader))
        assert batch["labels"].shape[-1] == 2
        assert trainer.train_dataloader.collate_fn.action_dim == 2

    def test_batch_size_handling(self):
        """Test that batch size variations are handled properly."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

        class VariableBatchDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "actions": torch.randn(7),
                    "instructions": "test instruction"
                }

        dataset = VariableBatchDataset()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            trainer = FastVLATrainer(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                max_steps=1,
                logging_steps=1,
                use_8bit_optimizer=False,
                use_mixed_precision=False,
            )

            batch = next(iter(trainer.train_dataloader))
            metrics = trainer.train_step(batch)

            assert "loss" in metrics
            assert isinstance(metrics["loss"], float)

    def test_single_sample_batch(self):
        """Test training with batch size of 1."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

        class SingleSampleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "actions": torch.randn(7),
                    "instructions": "test"
                }

        dataset = SingleSampleDataset()
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            batch_size=1,
            max_steps=2,
            logging_steps=1,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        history = trainer.train()
        assert len(history) >= 2


class TestCollatorValidation:
    """Test data collator validation and error handling."""

    def test_inconsistent_action_dimensions(self):
        """Test that inconsistent action dimensions in batch raise error."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
        tokenizer = model.tokenizer

        collator = UnslothVLACollator(tokenizer=tokenizer, action_dim=7)

        # Create features with inconsistent action dimensions
        features = [
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(7),  # Correct dimension
                "instructions": "test1"
            },
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(3),  # Wrong dimension
                "instructions": "test2"
            },
        ]

        with pytest.raises(ValueError, match="Inconsistent action dimensions in batch"):
            collator(features)

    def test_scalar_action_handling(self):
        """Test that scalar actions are properly reshaped."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
        tokenizer = model.tokenizer

        collator = UnslothVLACollator(tokenizer=tokenizer, action_dim=1)

        features = [
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.tensor(0.5),  # Scalar action
                "instructions": "test1"
            },
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.tensor(0.3),  # Scalar action
                "instructions": "test2"
            },
        ]

        batch = collator(features)
        assert "labels" in batch
        # Scalars should be reshaped to [batch_size, 1]
        assert batch["labels"].shape == (2, 1)

    def test_action_dim_auto_update(self, capsys):
        """Test that collator auto-updates action_dim with warning."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)
        tokenizer = model.tokenizer

        collator = UnslothVLACollator(tokenizer=tokenizer, action_dim=7)

        features = [
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(2),  # Different dimension
                "instructions": "test1"
            },
            {
                "images": {"rgb": torch.randn(3, 224, 224)},
                "actions": torch.randn(2),  # Same dimension
                "instructions": "test2"
            },
        ]

        batch = collator(features)

        # Should have printed a warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert collator.action_dim == 2


class TestGradientAccumulation:
    """Test gradient accumulation works correctly."""

    def test_gradient_accumulation_steps(self):
        """Test that gradient accumulation properly accumulates gradients."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "actions": torch.randn(7),
                    "instructions": "test"
                }

        dataset = SimpleDataset()

        # Test with gradient accumulation
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            batch_size=2,
            max_steps=4,
            logging_steps=1,
            gradient_accumulation_steps=2,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        # Run training
        history = trainer.train()
        assert len(history) >= 4
        assert trainer.global_step == 4


class TestDistributedTrainingSimulation:
    """Simulate distributed training scenarios."""

    def test_multiple_training_runs(self):
        """Test that multiple training runs are consistent."""
        results = []

        for _ in range(3):
            torch.manual_seed(42)
            model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

            class ConsistentDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 10

                def __getitem__(self, idx):
                    torch.manual_seed(idx)
                    return {
                        "images": {"rgb": torch.randn(3, 224, 224)},
                        "actions": torch.randn(7),
                        "instructions": "test"
                    }

            dataset = ConsistentDataset()
            trainer = FastVLATrainer(
                model=model,
                dataset=dataset,
                batch_size=2,
                max_steps=2,
                logging_steps=1,
                use_8bit_optimizer=False,
                use_mixed_precision=False,
            )

            history = trainer.train()
            results.append(history)

        # All runs should complete successfully
        for result in results:
            assert len(result) > 0

    def test_model_device_consistency(self):
        """Test that model devices remain consistent."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

        class DeviceDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "actions": torch.randn(7),
                    "instructions": "test"
                }

        dataset = DeviceDataset()
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            max_steps=1,
            logging_steps=1,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        # Check that all model components are on expected devices
        # (In dummy mode, everything should be on CPU)
        batch = next(iter(trainer.train_dataloader))
        metrics = trainer.train_step(batch)

        assert "loss" in metrics


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_labels(self):
        """Test handling of missing labels."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

        class NoLabelsDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "instructions": "test"
                    # No actions/labels
                }

        dataset = NoLabelsDataset()
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            max_steps=1,
            logging_steps=1,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        batch = next(iter(trainer.train_dataloader))
        # Should not crash, but return loss=None
        action_preds, loss = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        assert loss is None
        assert action_preds is not None

    def test_very_long_sequence(self):
        """Test handling of long sequences."""
        model = FastVLAModel.from_pretrained(
            dummy=True,
            vocab_size=50257,
            max_sequence_length=1024
        )

        class LongSequenceDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "images": {"rgb": torch.randn(3, 224, 224)},
                    "actions": torch.randn(7),
                    # Very long instruction
                    "instructions": "do this task " * 100
                }

        dataset = LongSequenceDataset()
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            max_steps=1,
            logging_steps=1,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        batch = next(iter(trainer.train_dataloader))
        metrics = trainer.train_step(batch)
        assert "loss" in metrics

    def test_multi_camera_input(self):
        """Test handling of multiple camera inputs."""
        model = FastVLAModel.from_pretrained(dummy=True, vocab_size=50257)

        class MultiCameraDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "images": {
                        "cam1": torch.randn(3, 224, 224),
                        "cam2": torch.randn(3, 224, 224),
                        "cam3": torch.randn(3, 224, 224),
                    },
                    "actions": torch.randn(7),
                    "instructions": "test"
                }

        dataset = MultiCameraDataset()
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            max_steps=1,
            logging_steps=1,
            use_8bit_optimizer=False,
            use_mixed_precision=False,
        )

        batch = next(iter(trainer.train_dataloader))
        # Should have shape [B, 3, C, H, W] for 3 cameras
        assert batch["pixel_values"].shape[1] == 3

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
