"""Action Head Adapters for FastVLA — Different action representations."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_device
from ..kernels.fused_flow_loss import fused_flow_interpolate, fused_flow_mse_loss
from ..geometry.se3 import se3_flow_interpolate


class BaseActionHead(nn.Module):
    """Base class for all action heads."""

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(
        self, hidden_states: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Optional head-specific training objective computed directly from the
        pooled hidden states (the conditioning) and the target actions.

        Heads that need a bespoke objective (e.g. flow matching, which trains a
        velocity field rather than regressing the decoded action) override this.
        Returning ``None`` signals the model to fall back to the generic
        regression loss (MSE/L1) on the decoded action predictions.
        """
        return None


class DiscreteActionHead(BaseActionHead):
    """
    MLP action head with discretized action bins (OpenVLA style).
    Each action dimension is discretized into `num_bins` bins.
    Output: [B, action_dim] decoded to the bin centre in [0, 1].

    Training and inference share the *same* discretized value via a
    straight-through estimator: the forward value is always the hard argmax
    bin, while gradients flow through the differentiable soft-argmax. This
    removes the train/inference distribution shift that arises when training
    on a soft expectation but deploying with a hard argmax.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_bins: int = 256,
        use_triton: bool = True,
    ):
        super().__init__(input_dim, action_dim)
        self.num_bins = num_bins
        self.use_triton = use_triton and get_device() == "cuda"

        # Bin prediction layers: output logits for each bin per action dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim * num_bins)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure input dtype matches layer weights for mixed precision compatibility
        hidden_states = hidden_states.to(self.fc1.weight.dtype)
        h = F.relu(self.fc1(hidden_states))
        logits = self.fc2(h)  # [B, action_dim * num_bins]
        return logits.view(-1, self.action_dim, self.num_bins)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, input_dim] (pooled LLM output)
        Returns:
            actions: [B, action_dim] (discrete bin centres in [0, 1])
        """
        logits = self._logits(hidden_states)

        # Hard argmax bin centre — the value actually used at train and inference.
        bin_idx = logits.argmax(dim=-1)
        hard = bin_idx.to(logits.dtype) / (self.num_bins - 1)

        if self.training or logits.requires_grad:
            # Straight-through estimator: forward = hard, gradient = soft-argmax.
            probs = F.softmax(logits, dim=-1)
            bin_values = torch.linspace(
                0, 1, self.num_bins, device=logits.device, dtype=logits.dtype
            )
            soft = (probs * bin_values).sum(dim=-1)
            return hard + (soft - soft.detach())

        return hard

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expose per-bin logits for a cross-entropy objective."""
        return self._logits(hidden_states)

    def compute_loss(
        self, hidden_states: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy over discretized bins — the principled objective for a
        classification-style action head. Uses fused Triton cross-entropy when available.
        """
        logits = self._logits(hidden_states)  # [B, action_dim, num_bins]
        action_dim = logits.shape[1]
        tgt = targets[..., :action_dim].to(logits.device).clamp(0.0, 1.0)
        bin_idx = (tgt * (self.num_bins - 1)).round().long()

        from ..kernels import fused_cross_entropy_loss
        return fused_cross_entropy_loss(
            logits.reshape(-1, self.num_bins), bin_idx.reshape(-1)
        )

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE loss between predicted and target actions."""
        return F.mse_loss(predictions, targets)


class ContinuousActionHead(BaseActionHead):
    """
    MLP action head for continuous action output.
    Outputs continuous values in [-1, 1] via Tanh.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        use_triton: bool = True,
    ):
        super().__init__(input_dim, action_dim)
        self.use_triton = use_triton and get_device() == "cuda"

        if self.use_triton:
            # Use Triton action head
            from ..kernels.action_head import TritonActionHead

            self.triton_head = TritonActionHead(input_dim, hidden_dim, action_dim)
            self.fc1 = None
            self.fc2 = None
        else:
            # Standard PyTorch MLP
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
            self.triton_head = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, input_dim]
        Returns:
            actions: [B, action_dim] in [-1, 1]
        """
        if self.triton_head is not None:
            return self.triton_head(hidden_states)

        # Ensure input dtype matches layer weights dtype for mixed precision compatibility
        hidden_states = hidden_states.to(self.fc1.weight.dtype)
        h = F.relu(self.fc1(hidden_states))
        return torch.tanh(self.fc2(h))

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE loss."""
        return F.mse_loss(predictions, targets)


class FlowMatchingActionHead(BaseActionHead):
    """
    Conditional Flow Matching action head (π₀ style).

    Trains a time-conditioned velocity field ``v_θ(x_t, t | c)`` that
    transports a standard-normal prior to the action distribution along the
    linear probability path ``x_t = (1 - t) · x_0 + t · a``. The target
    velocity for that path is constant: ``a - x_0``.

    - Training: ``compute_loss`` samples ``t ~ U[0, 1]`` and ``x_0 ~ N(0, I)``
      and regresses the velocity (the standard CFM objective).
    - Inference: ``forward`` integrates the ODE ``dx/dt = v_θ`` from ``t = 0``
      to ``t = 1`` with Euler steps, then squashes to [-1, 1] via tanh.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_inference_steps: int = 10,
        time_embed_dim: int = 64,
        solver: str = "euler",
        manifold: str = "euclidean",
        rotation_format: str = "7dof_euler",
    ):
        super().__init__(input_dim, action_dim)
        self.num_inference_steps = num_inference_steps
        self.time_embed_dim = time_embed_dim
        self.solver = solver
        self.manifold = manifold.lower()
        self.rotation_format = rotation_format

        # Conditioning projection from pooled hidden states.
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        # Time embedding (sinusoidal features -> MLP).
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Velocity field: [x_t ; cond ; time] -> velocity.
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of a [B] time tensor -> [B, time_embed_dim]."""
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.time_embed_dim:  # odd time_embed_dim
            emb = F.pad(emb, (0, self.time_embed_dim - emb.shape[-1]))
        return emb

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Predict dx/dt given the current sample, time, and conditioning."""
        t_emb = self.time_mlp(self._time_embedding(t))
        inp = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(inp)

    def solve_ode(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        Integrates the continuous velocity ODE dx/dt = v_theta(x, t, cond) from t=0 to t=1.

        Supported solvers:
        - 'euler': 1st-order explicit Euler
        - 'midpoint': 2nd-order Runge-Kutta Midpoint method (evaluates velocity at half-step)
        - 'heun': 2nd-order Runge-Kutta Heun predictor-corrector
        """
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype
        x = x0.clone()
        dt = 1.0 / steps

        if solver == "euler":
            for step in range(steps):
                t = torch.full((B,), step * dt, device=device, dtype=dtype)
                x = x + dt * self.velocity(x, t, cond)
        elif solver == "midpoint":
            for step in range(steps):
                t_curr = torch.full((B,), step * dt, device=device, dtype=dtype)
                t_mid = torch.full((B,), (step + 0.5) * dt, device=device, dtype=dtype)
                v1 = self.velocity(x, t_curr, cond)
                x_mid = x + 0.5 * dt * v1
                v_mid = self.velocity(x_mid, t_mid, cond)
                x = x + dt * v_mid
        elif solver == "heun":
            for step in range(steps):
                t_curr = torch.full((B,), step * dt, device=device, dtype=dtype)
                t_next = torch.full((B,), min((step + 1) * dt, 1.0), device=device, dtype=dtype)
                v1 = self.velocity(x, t_curr, cond)
                x_pred = x + dt * v1
                v2 = self.velocity(x_pred, t_next, cond)
                x = x + 0.5 * dt * (v1 + v2)
        else:
            raise ValueError(f"Unknown ODE solver: {solver}. Choose 'euler', 'midpoint', or 'heun'.")

        if self.manifold == "se3" and self.rotation_format == "8dof_quat":
            # Project quaternion slice onto unit 3-sphere S^3 to prevent manifold drift
            if self.action_dim % 8 == 0:
                orig_shape = x.shape
                x_reshaped = x.view(B, -1, 8)
                q = x_reshaped[..., 3:7]
                q_proj = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
                x = torch.cat(
                    [x_reshaped[..., :3], q_proj, x_reshaped[..., 7:8]], dim=-1
                ).view(orig_shape)

        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        solver: str | None = None,
        steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Integrate the learned ODE from noise to an action.
        Returns: [B, action_dim] in [-1, 1].
        """
        dtype = self.cond_proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        cond = self.cond_proj(hidden_states)
        B = hidden_states.shape[0]
        device = hidden_states.device

        if initial_noise is not None:
            x0 = initial_noise.to(device=device, dtype=dtype)
        else:
            x0 = torch.randn(B, self.action_dim, device=device, dtype=dtype)

        solver = solver or self.solver
        steps = steps or self.num_inference_steps

        x = self.solve_ode(x0, cond, steps=steps, solver=solver)
        return torch.tanh(x)

    def compute_loss(
        self, hidden_states: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Conditional flow matching loss with Euclidean or SE(3) Riemannian probability paths."""
        dtype = self.cond_proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        cond = self.cond_proj(hidden_states)

        target = targets[..., : self.action_dim].to(device=cond.device, dtype=dtype)
        B = target.shape[0]

        x0 = torch.randn_like(target)
        t = torch.rand(B, device=target.device, dtype=dtype)

        if self.manifold == "se3":
            x_t, target_velocity = se3_flow_interpolate(
                target, x0, t, format=self.rotation_format
            )
        else:
            # Fused SRAM interpolation of OT path and target velocity
            x_t, target_velocity = fused_flow_interpolate(target, x0, t)

        pred_velocity = self.velocity(x_t, t, cond)
        return fused_flow_mse_loss(pred_velocity, target_velocity)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)


def get_action_head(input_dim: int, config: dict) -> BaseActionHead:
    """Create an action head from config."""
    head_type = config.get("head_type", "mlp_continuous")

    if head_type == "mlp_discrete":
        return DiscreteActionHead(
            input_dim=input_dim,
            action_dim=config.get("action_dim", 7),
            hidden_dim=config.get("hidden_dim", 256),
            num_bins=config.get("num_bins", 256),
            use_triton=config.get("use_triton", True),
        )
    elif head_type == "mlp_continuous":
        return ContinuousActionHead(
            input_dim=input_dim,
            action_dim=config.get("action_dim", 7),
            hidden_dim=config.get("hidden_dim", 256),
            use_triton=config.get("use_triton", True),
        )
    elif head_type == "flow_matching":
        return FlowMatchingActionHead(
            input_dim=input_dim,
            action_dim=config.get("action_dim", 7),
            hidden_dim=config.get("hidden_dim", 256),
            num_inference_steps=config.get("num_inference_steps", 10),
            solver=config.get("solver", "euler"),
            manifold=config.get("manifold", "euclidean"),
            rotation_format=config.get("rotation_format", "7dof_euler"),
        )
    else:
        raise ValueError(f"Unknown action head type: {head_type}")
