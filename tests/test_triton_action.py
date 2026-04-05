import torch
import torch.nn as nn
from fastvla.kernels import TritonActionHead


def test_triton_action_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping Triton parity test (no CUDA)")
        return

    B, D, H, A = 4, 128, 256, 7

    # Standard PyTorch Reference
    torch_head = nn.Sequential(
        nn.Linear(D, H), nn.ReLU(), nn.Linear(H, A), nn.Tanh()
    ).to(device)

    # Triton Head
    triton_head = TritonActionHead(D, H, A).to(device)

    # Copy weights for exact parity
    with torch.no_grad():
        triton_head.weight1.copy_(torch_head[0].weight.t())
        triton_head.bias1.copy_(torch_head[0].bias)
        triton_head.weight2.copy_(torch_head[2].weight.t())
        triton_head.bias2.copy_(torch_head[2].bias)

    x = torch.randn(B, D, device=device, requires_grad=True)

    # Forward Pass
    out_torch = torch_head(x)
    out_triton = triton_head(x)

    print(
        f"Forward Max Diff: {torch.max(torch.abs(out_torch - out_triton)).item():.2e}"
    )
    assert torch.allclose(out_torch, out_triton, atol=1e-5), "Forward parity failed!"
    print("✅ Forward parity passed!")

    # Backward Pass
    grad_output = torch.randn_like(out_torch)
    out_torch.backward(grad_output)
    grad_x_torch = x.grad.clone()

    x.grad.zero_()
    out_triton.backward(grad_output)
    grad_x_triton = x.grad.clone()

    print(
        f"Backward Grad_X Max Diff: {torch.max(torch.abs(grad_x_torch - grad_x_triton)).item():.2e}"
    )
    assert torch.allclose(grad_x_torch, grad_x_triton, atol=1e-5), (
        "Backward parity failed!"
    )
    print("✅ Backward parity passed!")


if __name__ == "__main__":
    test_triton_action_parity()
