"""
Group Relative Policy Optimization (GRPO) Implementation for FastVLA.
Removes the need for a separate Value Head by using group-relative rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class GRPOTrainer:
    def __init__(
        self,
        model,
        learning_rate=1e-6,
        eps_clip=0.2,
        beta=0.01,  # KL penalty coefficient
        group_size=8,
        action_std=0.05,
        accelerator=None,
        num_warmup_steps=100,
        num_training_steps=1000,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.eps_clip = eps_clip
        self.beta = beta
        self.group_size = group_size
        self.action_std = action_std
        self.accelerator = accelerator
        
        # Buffer for GRPO (groups of trajectories)
        self.buffer = []

    def select_action(self, obs, input_ids):
        action_std = self.action_std
        
        with torch.no_grad():
            # Get policy mean (mu)
            (action_preds, _), _ = self.model(pixel_values=obs, input_ids=input_ids)
            mu = action_preds[0]
            
            # Sample action
            dist = Normal(mu, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
        return action, log_prob, None # No value in GRPO

    def store_transition(self, obs, input_ids, action, log_prob, reward, value, terminal):
        # Value is ignored in GRPO
        self.buffer.append({
            "obs": obs,
            "input_ids": input_ids,
            "action": action,
            "log_prob": log_prob,
            "reward": reward
        })

    def update(self):
        if len(self.buffer) == 0:
            return {}

        # GRPO Logic:
        # 1. Organize buffer into groups
        # 2. Compute group-relative advantages: Adv = (R - mean(R_group)) / std(R_group)
        # 3. Policy update with KL penalty
        
        # Convert buffer to tensors
        obs = torch.stack([x["obs"] for x in self.buffer]).to(self.model.device).squeeze(1)
        input_ids = torch.stack([x["input_ids"] for x in self.buffer]).to(self.model.device).squeeze(1)
        old_actions = torch.stack([x["action"] for x in self.buffer]).to(self.model.device)
        old_log_probs = torch.stack([x["log_prob"] for x in self.buffer]).to(self.model.device)
        rewards = torch.tensor([x["reward"] for x in self.buffer], dtype=torch.float32).to(self.model.device)

        # Compute Group Advantages
        # Assuming the buffer is filled in groups of `group_size`
        num_groups = len(rewards) // self.group_size
        if num_groups == 0:
            return {"error": "Not enough samples for a group update"}

        advantages = torch.zeros_like(rewards)
        group_stds = []
        for i in range(num_groups):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            group_rewards = rewards[start:end]
            
            mean = group_rewards.mean()
            std = group_rewards.std()
            group_stds.append(std.item())
            advantages[start:end] = (group_rewards - mean) / (std + 1e-8)

        mean_group_std = np.mean(group_stds)

        # Policy Update
        action_std = self.action_std
        
        # Re-run model for current log_probs
        # Use accelerator if available
        (action_preds, _), _ = self.model(pixel_values=obs, input_ids=input_ids)
        mu = action_preds
        dist = Normal(mu, action_std)
        new_log_probs = dist.log_prob(old_actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        # Ratio for surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Total Loss
        total_loss = policy_loss # GRPO usually includes a KL penalty to the reference model here
        
        self.optimizer.zero_grad()
        if self.accelerator:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.buffer = [] # Clear buffer
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": mean_group_std,
            "lr": self.scheduler.get_last_lr()[0]
        }
