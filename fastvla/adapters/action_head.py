"""Action Head Adapters for FastVLA — Different action representations."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_device


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


class DiscreteActionHead(BaseActionHead):
    """
    MLP action head with discretized action bins (OpenVLA style).
    Each action dimension is discretized into `num_bins` bins.
    Output: [B, action_dim] (argmax of bin predictions)
    """

    def __init__(self, input_dim: int, action_dim: int = 7,
                 hidden_dim: int = 256, num_bins: int = 256,
                 use_triton: bool = True):
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, input_dim] (pooled LLM output)
        Returns:
            actions: [B, action_dim] (discrete actions in [0, 1])
        """
        # Ensure input dtype matches layer weights dtype for mixed precision compatibility
        hidden_states = hidden_states.to(self.fc1.weight.dtype)
        h = F.relu(self.fc1(hidden_states))
        logits = self.fc2(h)  # [B, action_dim * num_bins]
        logits = logits.view(-1, self.action_dim, self.num_bins)

        # Argmax over bins
        actions = logits.argmax(dim=-1).float() / (self.num_bins - 1)  # Normalize to [0, 1]
        return actions

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE loss between predicted and target actions."""
        return F.mse_loss(predictions, targets)


class ContinuousActionHead(BaseActionHead):
    """
    MLP action head for continuous action output.
    Outputs continuous values in [-1, 1] via Tanh.
    """

    def __init__(self, input_dim: int, action_dim: int = 7,
                 hidden_dim: int = 256, use_triton: bool = True):
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
    Flow matching action head (π₀ style).
    Uses conditional flow matching for continuous action generation.
    Simplified implementation for demonstration.
    """

    def __init__(self, input_dim: int, action_dim: int = 7,
                 hidden_dim: int = 256):
        super().__init__(input_dim, action_dim)
        # Simplified: MLP that predicts action directly
        # Full implementation would use flow matching ODE solver
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure input dtype matches layer weights dtype for mixed precision compatibility
        hidden_states = hidden_states.to(self.fc1.weight.dtype)
        h = F.relu(self.fc1(hidden_states))
        return torch.tanh(self.fc2(h))

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
        )
    else:
        raise ValueError(f"Unknown action head type: {head_type}")
