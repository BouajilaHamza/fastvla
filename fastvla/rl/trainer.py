"""
High-level RL Trainer API for FastVLA.
Routes to specific RL algorithms (PPO, DPO, etc.) based on configuration.
"""

import torch.nn as nn
from accelerate import Accelerator
from .ppo import PPOTrainer


class RLTrainer:
    """
    Unified entry point for RL training in FastVLA.
    """

    def __init__(
        self, model: nn.Module, algo: str = "ppo", learning_rate: float = 1e-5, **kwargs
    ):
        self.model = model
        self.algo = algo.lower()
        self.accelerator = Accelerator()

        # Ensure model has value head if using PPO
        if self.algo == "ppo" and not getattr(self.model, "value_head", None):
            print("⚠️ ValueHead not found in model. Initializing default ValueHead...")
            from ..adapters.value_head import ValueHead

            llm_hidden_size = self.model.config.llm_hidden_size
            self.model.value_head = ValueHead(llm_hidden_size).to(self.model.device)
            self.model.config.use_rl = True

        if self.algo == "ppo":
            self.trainer = PPOTrainer(
                model=self.model,
                learning_rate=learning_rate,
                accelerator=self.accelerator,
                **kwargs,
            )
        else:
            raise ValueError(f"RL Algorithm '{algo}' not yet implemented.")

    def select_action(self, obs, input_ids):
        """Select an action using the current policy."""
        return self.trainer.select_action(obs, input_ids)

    def store_transition(
        self, obs, input_ids, action, log_prob, reward, value, terminal
    ):
        """Store experience in the buffer."""
        self.trainer.buffer.add(
            obs, input_ids, action, log_prob, reward, value, terminal
        )

    def train_step(self):
        """Perform one RL update cycle."""
        return self.trainer.update()
