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
        # Ensure input dtype matches weight dtype
        target_dtype = self.weight1.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        
        if x.is_cuda:
            return ActionDecodeFunction.apply(
                x, self.weight1, self.bias1, self.weight2, self.bias2
            )
        # Fallback for CPU (using the same logic for consistency)
        h = torch.nn.functional.relu(x @ self.weight1 + self.bias1)
        return torch.tanh(h @ self.weight2 + self.bias2)
