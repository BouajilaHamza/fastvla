"""Integration tests for the FastVLA model."""

import torch
import pytest
from fastvla import FastVLAModel, FastVLAConfig


class TestFastVLAModel:
    """Tests for the FastVLA model."""

    @pytest.fixture
    def dummy_config(self):
        return FastVLAConfig(
            dummy=True,
            vision_hidden_size=64,
            llm_hidden_size=64,
            llm_num_layers=2,
            num_attention_heads=4,
            vocab_size=500,
            action_dim=7,
            action_hidden_dim=32,
            gradient_checkpointing=False,
            use_peft=False,
            load_in_4bit=False,
        )

    @pytest.fixture
    def dummy_batch(self, dummy_config):
        batch_size = 2
        seq_len = 8
        return {
            "pixel_values": torch.randn(batch_size, 2, 3, 32, 32),
            "input_ids": torch.randint(
                0, dummy_config.vocab_size, (batch_size, seq_len)
            ),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randn(batch_size, dummy_config.action_dim),
        }

    def test_forward_pass(self, dummy_config, dummy_batch):
        """Test forward pass through the model."""
        model = FastVLAModel(dummy_config)

        outputs = model(
            pixel_values=dummy_batch["pixel_values"],
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
            labels=dummy_batch["labels"],
        )

        action_preds, loss = outputs
        assert action_preds.shape == (2, dummy_config.action_dim)
        assert loss is not None
        assert not torch.isnan(loss)

    def test_training_step(self, dummy_config, dummy_batch):
        """Test a single training step."""
        model = FastVLAModel(dummy_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        _, loss = model(
            pixel_values=dummy_batch["pixel_values"],
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
            labels=dummy_batch["labels"],
        )
        loss.backward()
        optimizer.step()

        for param in model.action_head.parameters():
            assert param.grad is not None

    def test_from_pretrained_dummy(self):
        """Test loading a dummy model via from_pretrained."""
        model = FastVLAModel.from_pretrained(
            dummy=True,
            vision_encoder_name="dummy",
            llm_name="dummy",
            gradient_checkpointing=False,
            use_peft=False,
        )
        assert model is not None
        assert model.config.dummy is True


if __name__ == "__main__":
    pytest.main([__file__])
