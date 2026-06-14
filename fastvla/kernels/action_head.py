import torch
import torch.nn as nn
from .action import action_decode_forward


def _match_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast only when needed (zero-copy if dtypes already match)."""
    return t if t.dtype == dtype else t.to(dtype)


class ActionDecodeFunction(torch.autograd.Function):
    """
    Autograd-aware action decoding for the fused 2-layer MLP
    ``tanh(ReLU(x @ W1 + b1) @ W2 + b2)``.

    The intermediate activation ``h1`` and the output are saved during the
    forward pass so the backward pass does NOT recompute the forward chain.
    For a small action head these activations are negligible in memory, so
    storing them is strictly cheaper than the previous recompute-everything
    approach.
    """

    @staticmethod
    def forward(ctx, hidden, weight1, bias1, weight2, bias2):
        dtype = hidden.dtype
        weight1 = _match_dtype(weight1, dtype)
        bias1 = _match_dtype(bias1, dtype)
        weight2 = _match_dtype(weight2, dtype)
        bias2 = _match_dtype(bias2, dtype)

        h1 = torch.relu(hidden @ weight1 + bias1)
        out = torch.tanh(h1 @ weight2 + bias2)

        ctx.save_for_backward(hidden, weight1, weight2, h1, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight1, weight2, h1, out = ctx.saved_tensors
        grad_output = _match_dtype(grad_output, hidden.dtype)

        # d/dx tanh = 1 - tanh^2  (reuse cached output, no recompute)
        d_out = grad_output * (1.0 - out * out)

        grad_weight2 = h1.t() @ d_out
        grad_bias2 = d_out.sum(dim=0)
        d_h1 = d_out @ weight2.t()

        # ReLU gradient via cached h1
        d_h1_relu = d_h1 * (h1 > 0).to(d_h1.dtype)

        grad_weight1 = hidden.t() @ d_h1_relu
        grad_bias1 = d_h1_relu.sum(dim=0)
        grad_hidden = d_h1_relu @ weight1.t()

        return grad_hidden, grad_weight1, grad_bias1, grad_weight2, grad_bias2


class TritonActionHead(nn.Module):
    """
    Memory-efficient Action Head using custom Triton kernels.
    Fuses: Linear -> ReLU -> Linear -> Tanh

    - Inference (no grad): uses the fully-fused Triton forward kernel.
    - Training: uses an autograd Function with a backward that reuses cached
      activations instead of recomputing the forward chain.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.action_dim = output_dim
        self.weight1 = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.empty(hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(hidden_dim, output_dim))
        self.bias2 = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight1, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.bias1)
        nn.init.kaiming_normal_(self.weight2, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        w1 = _match_dtype(self.weight1, dtype)
        b1 = _match_dtype(self.bias1, dtype)
        w2 = _match_dtype(self.weight2, dtype)
        b2 = _match_dtype(self.bias2, dtype)

        # Fully-fused Triton path only when we don't need a graph.
        if x.is_cuda and not torch.is_grad_enabled():
            return action_decode_forward(x, w1, b1, w2, b2)

        if x.is_cuda:
            return ActionDecodeFunction.apply(x, w1, b1, w2, b2)

        # CPU / no-Triton fallback
        h = torch.nn.functional.relu(x @ w1 + b1)
        return torch.tanh(h @ w2 + b2)
