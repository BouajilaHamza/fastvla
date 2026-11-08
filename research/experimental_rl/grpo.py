"""
Group Relative Policy Optimization (GRPO) for FastVLA.

Critic-free Reinforcement Learning for continuous trajectory Vision-Language-Action policies.
Features:
1. Zero-Copy Reference Policy: Evaluates reference policy KL penalties via PEFT
   `model.disable_adapter()` without duplicating 3B/7B weights in VRAM.
2. Parallel Group Trajectory Sampling: Samples G continuous trajectory candidates
   in a single forward pass, expanding token features to amortize vision compute.
3. Multi-Step Trajectory Support: Native support for 16-step continuous action chunks
   (32 DoF for Push-T, 112 DoF for Franka/LIBERO).
4. Stable Group-Relative Advantage Normalization: Computes advantages per group
   without an actor-critic value head, eliminating value-loss explosion.
5. Adaptive Exploration Scheduling: Dynamically controls Gaussian action noise
   to prevent premature entropy collapse.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Dict, Any, Tuple, List, Union

from ..optimization import get_8bit_optimizer


class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO) Trainer for FastVLA models."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-5,
        eps_clip: float = 0.2,
        beta: float = 0.01,  # KL penalty coefficient
        group_size: int = 4,  # Number of trajectory samples per prompt/observation
        action_std: float = 0.05,  # Continuous exploration Gaussian noise std
        min_action_std: float = 0.01,
        use_8bit_optimizer: bool = True,
        accelerator: Optional[Any] = None,
        num_warmup_steps: int = 50,
        num_training_steps: int = 1000,
    ):
        self.model = model
        self.ref_model = ref_model
        self.eps_clip = eps_clip
        self.beta = beta
        self.group_size = group_size
        self.action_std = float(action_std)
        self.min_action_std = float(min_action_std)
        self.accelerator = accelerator

        # Determine if zero-copy reference evaluation is supported (PEFT LoRA models)
        self.has_adapter = False
        if hasattr(self.model, "disable_adapter"):
            self.has_adapter = True
        elif hasattr(self.model, "llm") and hasattr(self.model.llm, "disable_adapter"):
            self.has_adapter = True

        # Only deepcopy reference model if explicitly required and adapters are not available
        if self.ref_model is None:
            if not self.has_adapter:
                # Fallback for non-PEFT models (e.g. dummy test models)
                self.ref_model = copy.deepcopy(model)
                for p in self.ref_model.parameters():
                    p.requires_grad = False
                self.ref_model.eval()

        # Optimizer: Paged 8-bit AdamW or standard PyTorch AdamW
        if use_8bit_optimizer:
            self.optimizer = get_8bit_optimizer(self.model, learning_rate=learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
            )

        from transformers import get_cosine_schedule_with_warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Replay buffer holding groups of transitions
        self.buffer: List[Dict[str, Any]] = []

    def _get_device(self) -> torch.device:
        """Infer model device safely."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _forward_model(
        self,
        obs: torch.Tensor,
        input_ids: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        use_ref: bool = False,
    ) -> torch.Tensor:
        """Forward pass through active policy or zero-copy reference model."""
        target_model = self.ref_model if use_ref and self.ref_model is not None else self.model

        # Ensure correct observation tensor shape [B, num_cams, C, H, W]
        pixel_values = obs
        if pixel_values is not None:
            if pixel_values.dim() == 3:  # [C, H, W]
                pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
            elif pixel_values.dim() == 4:  # [B, C, H, W]
                pixel_values = pixel_values.unsqueeze(1)  # [B, 1, C, H, W]
            elif pixel_values.dim() == 6:  # [B, 1, num_cams, C, H, W]
                pixel_values = pixel_values.squeeze(1)

        # Ensure input_ids is 2D [B, seq_len]
        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            elif input_ids.dim() == 3 and input_ids.shape[1] == 1:
                input_ids = input_ids.squeeze(1)

        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if states is not None:
            if states.dim() == 1:
                states = states.unsqueeze(0)
            elif states.dim() == 3 and states.shape[1] == 1:
                states = states.squeeze(1)
            kwargs["states"] = states

        # Zero-Copy Reference evaluation via adapter disabling
        if use_ref and self.has_adapter and self.ref_model is None:
            disable_ctx = (
                self.model.disable_adapter()
                if hasattr(self.model, "disable_adapter")
                else self.model.llm.disable_adapter()
            )
            with disable_ctx, torch.no_grad():
                outputs = self.model(**kwargs)
        else:
            outputs = target_model(**kwargs)

        preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return preds

    def select_action(
        self,
        obs: torch.Tensor,
        input_ids: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Select a single continuous trajectory action sample.
        
        Returns:
            action: Sampled action tensor [action_dim] or [chunk_size * action_dim]
            log_prob: Scalar log probability under current policy distribution
            value: None (GRPO is critic-free)
        """
        device = self._get_device()
        with torch.no_grad():
            mu = self._forward_model(obs.to(device), input_ids.to(device), states.to(device) if states is not None else None)
            if mu.dim() > 1 and mu.shape[0] == 1:
                mu = mu.squeeze(0)

            if deterministic:
                action = mu
                dist = Normal(mu, self.action_std)
                log_prob = dist.log_prob(action).sum()
            else:
                dist = Normal(mu, self.action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

        return action.detach(), log_prob.detach(), None

    def sample_group_trajectories(
        self,
        obs: torch.Tensor,
        input_ids: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        group_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample G trajectory candidates for the same observation.
        
        Amortizes visual and prompt encoding by running the forward pass once
        and expanding the predicted mean across the group size.

        Returns:
            actions: [G, action_dim]
            log_probs: [G]
            mu: [1, action_dim]
        """
        G = group_size or self.group_size
        device = self._get_device()

        with torch.no_grad():
            mu = self._forward_model(obs.to(device), input_ids.to(device), states.to(device) if states is not None else None)
            if mu.dim() > 1 and mu.shape[0] == 1:
                mu = mu.squeeze(0)

            # Expand mean to [G, action_dim]
            mu_expanded = mu.unsqueeze(0).expand(G, *mu.shape)
            dist = Normal(mu_expanded, self.action_std)
            actions = dist.sample()  # [G, action_dim]
            log_probs = dist.log_prob(actions).sum(dim=-1)  # [G]

        return actions.detach(), log_probs.detach(), mu.detach()

    def store_transition(
        self,
        obs: torch.Tensor,
        input_ids: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: Any = None,
        terminal: bool = False,
        states: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a single rollout transition into the replay buffer."""
        o = obs.detach().cpu()
        if o.dim() in [4, 5] and o.shape[0] == 1:
            o = o.squeeze(0)

        i_ids = input_ids.detach().cpu()
        if i_ids.dim() == 2 and i_ids.shape[0] == 1:
            i_ids = i_ids.squeeze(0)

        st = states.detach().cpu() if states is not None else None
        if st is not None and st.dim() == 2 and st.shape[0] == 1:
            st = st.squeeze(0)

        act = action.detach().cpu()
        if act.dim() > 1 and act.shape[0] == 1:
            act = act.squeeze(0)

        lp = log_prob.detach().cpu()
        if lp.dim() > 0 and lp.numel() == 1:
            lp = lp.squeeze()

        self.buffer.append({
            "obs": o,
            "input_ids": i_ids,
            "states": st,
            "action": act,
            "log_prob": lp,
            "reward": float(reward),
        })

    def store_group(
        self,
        obs: torch.Tensor,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: List[float],
        states: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a full group of G rollouts for the same observation."""
        for a, lp, r in zip(actions, log_probs, rewards):
            self.store_transition(
                obs=obs,
                input_ids=input_ids,
                action=a,
                log_prob=lp,
                reward=r,
                states=states,
            )

    def update_action_std(self, mean_reward: float, threshold: float = 0.6, decay: float = 0.95) -> float:
        """Adaptive exploration scheduling: decay std only when rewards pass threshold."""
        if mean_reward >= threshold:
            self.action_std = max(self.min_action_std, self.action_std * decay)
        return self.action_std

    def update(self) -> Dict[str, float]:
        """Execute Group Relative Policy Optimization update.
        
        1. Groups transitions by `group_size`.
        2. Computes group-relative advantage A_i = (R_i - mean(R)) / (std(R) + eps).
        3. Computes clipped surrogate objective with KL penalty against reference policy.
        """
        if len(self.buffer) == 0:
            return {}

        num_samples = len(self.buffer)
        num_groups = num_samples // self.group_size
        if num_groups == 0:
            return {"error": f"Insufficient samples: {num_samples} < group_size {self.group_size}"}

        # Truncate buffer to exact multiple of group_size
        usable_samples = num_groups * self.group_size
        active_buffer = self.buffer[:usable_samples]

        device = self._get_device()

        obs = torch.stack([x["obs"] for x in active_buffer]).to(device)
        input_ids = torch.stack([x["input_ids"] for x in active_buffer]).to(device)
        old_actions = torch.stack([x["action"] for x in active_buffer]).to(device)
        old_log_probs = torch.stack([x["log_prob"] for x in active_buffer]).to(device)
        rewards = torch.tensor([x["reward"] for x in active_buffer], dtype=torch.float32, device=device)

        has_states = active_buffer[0]["states"] is not None
        states = torch.stack([x["states"] for x in active_buffer]).to(device) if has_states else None

        # 1. Compute Group-Relative Advantages
        advantages = torch.zeros_like(rewards)
        group_stds = []
        for i in range(num_groups):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            g_rewards = rewards[start:end]

            mean_r = g_rewards.mean()
            std_r = g_rewards.std()
            group_stds.append(std_r.item())

            # If all rewards in group are identical, advantage is 0 (neutral)
            if std_r < 1e-6:
                advantages[start:end] = 0.0
            else:
                advantages[start:end] = (g_rewards - mean_r) / (std_r + 1e-8)

        # 2. Forward pass under current policy
        mu = self._forward_model(obs, input_ids, states=states, use_ref=False)
        dist = Normal(mu, self.action_std)
        new_log_probs = dist.log_prob(old_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        # 3. PPO Clipped Surrogate Loss
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. Reference Policy KL Regularization: D_KL(pi_theta || pi_ref)
        kl_penalty = torch.tensor(0.0, device=device)
        kl_div_val = 0.0
        if self.beta > 0.0:
            with torch.no_grad():
                ref_mu = self._forward_model(obs, input_ids, states=states, use_ref=True)
            # Analytical KL for two Gaussians with identical variance sigma^2:
            # KL = 0.5 * ||mu - mu_ref||^2 / sigma^2
            kl_div = 0.5 * torch.sum(((mu - ref_mu) / self.action_std) ** 2, dim=-1).mean()
            kl_penalty = self.beta * kl_div
            kl_div_val = float(kl_div.item())

        total_loss = policy_loss + kl_penalty

        # 5. Backpropagation
        self.optimizer.zero_grad()
        if self.accelerator:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()

        # Gradient clipping for RL stability
        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Clear buffer after successful update
        self.buffer = []

        return {
            "policy_loss": float(policy_loss.item()),
            "kl_penalty": float(kl_penalty.item()),
            "kl_divergence": kl_div_val,
            "entropy": float(entropy.item()),
            "total_loss": float(total_loss.item()),
            "mean_reward": float(rewards.mean().item()),
            "mean_group_std": float(np.mean(group_stds)),
            "action_std": self.action_std,
            "lr": float(self.scheduler.get_last_lr()[0]),
            "num_groups_updated": num_groups,
        }
