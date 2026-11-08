import torch
import pytest
from contextlib import contextmanager
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
    assert "mean_group_std" in metrics
    assert len(trainer.buffer) == 0  # Buffer cleared after update


def test_grpo_group_trajectory_sampling():
    """Verify parallel sampling of G continuous trajectory candidates in one pass."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        num_attention_heads=4,
        vocab_size=500,
        action_dim=32,  # 16-step continuous 2D trajectory chunk (16 x 2 = 32 DoF)
        action_hidden_dim=32,
        use_peft=False,
    )
    model = FastVLAModel(config)

    trainer = GRPOTrainer(
        model=model,
        group_size=8,
        action_std=0.05,
        use_8bit_optimizer=False,
    )

    obs = torch.randn(1, 1, 3, 32, 32)
    input_ids = torch.randint(0, 500, (1, 8))

    actions, log_probs, mu = trainer.sample_group_trajectories(obs, input_ids, group_size=8)

    assert actions.shape == (8, 32), f"Expected [8, 32], got {actions.shape}"
    assert log_probs.shape == (8,), f"Expected [8], got {log_probs.shape}"
    assert mu.shape == (32,), f"Expected [32], got {mu.shape}"

    # Verify diversity across samples
    diff = torch.norm(actions[0] - actions[1])
    assert diff > 0.0, "Exploration samples must differ"


def test_grpo_zero_copy_adapter_kl():
    """Verify zero-copy reference model evaluation via adapter disable."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        num_attention_heads=4,
        vocab_size=500,
        action_dim=16,
        action_hidden_dim=32,
        use_peft=False,
    )
    model = FastVLAModel(config)

    # Attach mock adapter disable context manager
    adapter_disabled_calls = []

    @contextmanager
    def mock_disable_adapter():
        adapter_disabled_calls.append(True)
        yield

    model.disable_adapter = mock_disable_adapter

    trainer = GRPOTrainer(
        model=model,
        group_size=4,
        beta=0.05,
        use_8bit_optimizer=False,
    )

    assert trainer.has_adapter is True
    assert trainer.ref_model is None, "Zero-copy mode must not allocate duplicate ref_model"

    obs = torch.randn(1, 1, 3, 32, 32)
    input_ids = torch.randint(0, 500, (1, 8))

    actions, log_probs, _ = trainer.sample_group_trajectories(obs, input_ids, group_size=4)
    rewards = [0.1, 0.4, 0.8, 0.2]
    trainer.store_group(obs, input_ids, actions, log_probs, rewards)

    metrics = trainer.update()
    assert len(adapter_disabled_calls) > 0, "Reference pass must trigger disable_adapter()"
    assert "kl_penalty" in metrics
    assert metrics["kl_penalty"] >= 0.0


def test_grpo_adaptive_action_std():
    """Verify reward-conditioned adaptive exploration scheduling."""
    config = FastVLAConfig(
        dummy=True,
        vision_hidden_size=64,
        llm_hidden_size=64,
        llm_num_layers=2,
        num_attention_heads=4,
        vocab_size=500,
        action_dim=7,
        use_peft=False,
    )
    model = FastVLAModel(config)
    trainer = GRPOTrainer(model=model, action_std=0.05, min_action_std=0.01, use_8bit_optimizer=False)

    # Low reward -> does not decay
    std = trainer.update_action_std(mean_reward=0.2, threshold=0.6, decay=0.90)
    assert std == 0.05

    # High reward -> decays
    std = trainer.update_action_std(mean_reward=0.8, threshold=0.6, decay=0.90)
    assert abs(std - 0.045) < 1e-5


if __name__ == "__main__":
    test_grpo_trainer_cycle()
    print("✅ test_grpo_trainer_cycle passed.")
    test_grpo_group_trajectory_sampling()
    print("✅ test_grpo_group_trajectory_sampling passed.")
    test_grpo_zero_copy_adapter_kl()
    print("✅ test_grpo_zero_copy_adapter_kl passed.")
    test_grpo_adaptive_action_std()
    print("✅ test_grpo_adaptive_action_std passed.")
    print("\n🎉 ALL GRPO UNIT TESTS PASSED!")
