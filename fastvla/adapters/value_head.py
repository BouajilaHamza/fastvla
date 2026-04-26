"""Value Head Adapter for FastVLA RL."""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Lightweight Value Head for Reinforcement Learning (Critic).
    Predicts a scalar value from the LLM's final hidden states.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.v_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, input_dim]
        Returns:
            values: [B, 1]
        """
        # Ensure input dtype matches layer weights dtype
        hidden_states = hidden_states.to(self.v_proj[0].weight.dtype)
        return self.v_proj(hidden_states)
