import torch
import pytest
from fastvla import FastVLAModel, FastVLAConfig
from fastvla.rl.grpo import GRPOTrainer


def test_grpo_trainer_cycle():
    """Verify end-to-end GRPO sampling, buffer storage, and group-relative update."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        num_attention_heads=4,
        vocab_size=500,
        action_dim=7,
        action_hidden_dim=32,
        use_peft=False,
    )
    model = FastVLAModel(config)

    trainer = GRPOTrainer(
        model=model,
        learning_rate=1e-4,
        group_size=4,
        beta=0.01,
        use_8bit_optimizer=False,  # CPU-compatible AdamW for unit testing
    )

    # 1. Action Selection
    obs = torch.randn(1, 1, 3, 32, 32)
    input_ids = torch.randint(0, 500, (1, 8))
    action, log_prob, value = trainer.select_action(obs, input_ids)

    assert action.shape == (7,)
    assert log_prob.numel() == 1
    assert value is None

    # 2. Buffer Storage: Create 1 group of 4 transitions
    for i in range(4):
        a, lp, _ = trainer.select_action(obs, input_ids)
        trainer.store_transition(
            obs=obs,
            input_ids=input_ids,
            action=a,
            log_prob=lp,
            reward=float(i % 2),
            value=None,
            terminal=False,
        )

    assert len(trainer.buffer) == 4

    # 3. Update Policy with Group Advantage & KL penalty
    metrics = trainer.update()
    assert "policy_loss" in metrics
    assert "total_loss" in metrics
    assert "reward_std" in metrics
    assert len(trainer.buffer) == 0  # Buffer cleared after update
