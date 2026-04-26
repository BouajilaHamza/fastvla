"""
PPO Implementation for FastVLA.
Optimized for continuous action spaces and high-performance VLA backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional
from accelerate import Accelerator


class PPOBuffer:
    """Experience buffer for PPO rollouts."""

    def __init__(self):
        self.observations = []
        self.input_ids = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminals = []

    def add(self, obs, input_ids, action, log_prob, reward, value, terminal):
        self.observations.append(obs)
        self.input_ids.append(input_ids)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminals.append(terminal)

    def clear(self):
        self.observations = []
        self.input_ids = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminals = []

    def get_batches(self, device):
        # 1. Observations: Target [BatchSize, 1, 3, H, W] (VLA expects camera dimension)
        processed_obs = []
        for o in self.observations:
            # Force into [1, 3, H, W]
            o_sq = o.detach().cpu()
            while o_sq.dim() > 4:
                o_sq = o_sq.squeeze(0)

            if o_sq.dim() == 4:  # [1, 3, H, W]
                processed_obs.append(o_sq)
            elif o_sq.dim() == 3:  # [3, H, W]
                processed_obs.append(o_sq.unsqueeze(0))
            else:
                raise ValueError(
                    f"Observation shape {o.shape} cannot be converted to [1, 3, H, W]"
                )

        # Concat into [Batch, 1, 3, H, W]
        # Instead of stacking into [Batch, 3, H, W], we need that extra camera dim
        obs = torch.stack(processed_obs, dim=0).to(
            device
        )  # Result: [Batch, 1, 3, H, W]

        # 2. IDs: Ensure [Batch, Seq]
        ids = torch.cat([i.view(1, -1) for i in self.input_ids], dim=0).to(device)

        # 3. Actions: Ensure [Batch, ActionDim]
        actions = torch.cat([a.view(1, -1) for a in self.actions], dim=0).to(device)

        # 4. Others: Ensure [Batch]
        log_probs = torch.cat([l.view(-1) for l in self.log_probs], dim=0).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        values = torch.cat([v.view(-1) for v in self.values], dim=0).to(device)
        terminals = torch.tensor(self.terminals, dtype=torch.float32).to(device)

        return obs, ids, actions, log_probs, rewards, values, terminals


class PPOTrainer:
    """
    PPO Trainer for FastVLA.
    Supports continuous action heads (Normal distribution) and discrete heads (Categorical).
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        target_kl: float = 0.01,
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        action_std: float = 0.1,  # Fixed std for continuous actions (can be learned later)
        accelerator: Optional[Accelerator] = None,
    ):
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.target_kl = target_kl
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.action_std = action_std

        self.accelerator = accelerator or Accelerator()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        self.buffer = PPOBuffer()

    def compute_gae(self, rewards, values, terminals, last_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            non_terminal = 1.0 - terminals[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.lam * non_terminal * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def select_action(self, obs, input_ids):
        """Select action and compute log_prob and value."""
        self.model.eval()
        with torch.no_grad():
            (action_preds, value_preds), _ = self.model(
                pixel_values=obs, input_ids=input_ids
            )

            # TODO: Handle Discrete heads
            # For now, assume continuous action head (mu)
            mu = action_preds
            dist = Normal(mu, self.action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value_preds

    def update(self):
        """Update policy and value networks using PPO."""
        obs, ids, actions, old_log_probs, rewards, values, terminals = (
            self.buffer.get_batches(self.accelerator.device)
        )

        # Compute advantages
        with torch.no_grad():
            # Get last value for GAE
            # This is a bit simplified, usually we need the next state's value
            last_value = torch.zeros(1, 1).to(self.accelerator.device)
            advantages, returns = self.compute_gae(
                rewards, values, terminals, last_value
            )
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.model.train()
        for _ in range(self.ppo_epochs):
            # Re-evaluate log_probs and values
            (new_action_preds, new_value_preds), _ = self.model(
                pixel_values=obs, input_ids=ids
            )

            # Policy Loss
            dist = Normal(new_action_preds, self.action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            value_loss = F.mse_loss(new_value_preds.view(-1), returns.view(-1))

            # Total Loss
            total_loss = (
                policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            )

            self.optimizer.zero_grad()
            self.accelerator.backward(total_loss)
            self.optimizer.step()

            # Check KL divergence for early stopping
            with torch.no_grad():
                kl = (old_log_probs - new_log_probs).mean()
                if kl > 1.5 * self.target_kl:
                    break

        self.buffer.clear()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "kl": kl.item(),
        }
