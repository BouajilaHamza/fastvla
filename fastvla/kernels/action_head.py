import torch
import torch.nn as nn
from .action import action_decode_forward, action_decode_backward


class ActionDecodeFunction(torch.autograd.Function):
    """
    Triton-optimized action decoding with gradient checkpointing.
    """

    @staticmethod
    def forward(ctx, hidden, weight1, bias1, weight2, bias2):
        # Ensure all inputs have the same dtype to prevent mixed precision errors
        dtype = hidden.dtype
        if weight1.dtype != dtype:
            weight1 = weight1.to(dtype)
        if bias1.dtype != dtype:
            bias1 = bias1.to(dtype)
        if weight2.dtype != dtype:
            weight2 = weight2.to(dtype)
        if bias2.dtype != dtype:
            bias2 = bias2.to(dtype)
        
        # We only save what's absolutely necessary for recomputation
        ctx.save_for_backward(hidden, weight1, bias1, weight2, bias2)
        return action_decode_forward(hidden, weight1, bias1, weight2, bias2)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight1, bias1, weight2, bias2 = ctx.saved_tensors
        
        # Ensure dtype consistency in backward pass
        dtype = hidden.dtype
        if grad_output.dtype != dtype:
            grad_output = grad_output.to(dtype)
        
        grads = action_decode_backward(
            grad_output, hidden, weight1, bias1, weight2, bias2
        )
        return grads


class TritonActionHead(nn.Module):
    """
    Memory-efficient Action Head using custom Triton kernels.
    Fuses: Linear -> ReLU -> Linear -> Tanh
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
        # Fix dtype mismatch for mixed precision training
        # We ensure weights and biases match the input x's dtype
        dtype = x.dtype
        w1, b1 = self.weight1.to(dtype), self.bias1.to(dtype)
        w2, b2 = self.weight2.to(dtype), self.bias2.to(dtype)

        if x.is_cuda:
            return ActionDecodeFunction.apply(x, w1, b1, w2, b2)
        
        # Fallback for CPU / No-Triton
        h = torch.nn.functional.relu(x @ w1 + b1)
        return torch.tanh(h @ w2 + b2)
