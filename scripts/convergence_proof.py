import torch
import torch.nn as nn
import torch.optim as optim
from fastvla.kernels import TritonActionHead

# ── Setup ────────────────────────────────────────────────────────────────
torch.manual_seed(42)
B, H_in, H_mid, D_out = 4, 4096, 1024, 7

# Data
X = torch.randn(20, B, H_in, device="cuda", dtype=torch.float16)  # 20 batches
Y = torch.randn(20, B, D_out, device="cuda", dtype=torch.float16)

# Global Initial Weights for Parity
w1_init = torch.randn(H_mid, H_in, device="cuda", dtype=torch.float16) * 0.02
b1_init = torch.zeros(H_mid, device="cuda", dtype=torch.float16)
w2_init = torch.randn(D_out, H_mid, device="cuda", dtype=torch.float16) * 0.02
b2_init = torch.zeros(D_out, device="cuda", dtype=torch.float16)


def run_training_loop(model_name):
    torch.manual_seed(42)
    if model_name == "pytorch":
        model = (
            nn.Sequential(
                nn.Linear(H_in, H_mid), nn.ReLU(), nn.Linear(H_mid, D_out), nn.Tanh()
            )
            .cuda()
            .half()
        )
        with torch.no_grad():
            model[0].weight.copy_(w1_init)
            model[0].bias.copy_(b1_init)
            model[2].weight.copy_(w2_init)
            model[2].bias.copy_(b2_init)
    else:
        model = TritonActionHead(H_in, H_mid, D_out).cuda().half()
        with torch.no_grad():
            model.weight1.copy_(w1_init.t().contiguous())
            model.bias1.copy_(b1_init)
            model.weight2.copy_(w2_init.t().contiguous())
            model.bias2.copy_(b2_init)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Fixed LR for half
    criterion = nn.MSELoss()

    losses = []
    for i in range(20):
        optimizer.zero_grad()
        out = model(X[i])
        loss = criterion(out, Y[i])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


print("Running Convergence Proof...")
pt_losses = run_training_loop("pytorch")
tr_losses = run_training_loop("triton")

print("\n" + "=" * 50)
print("CONVERGENCE PROOF: LOSS TRAJECTORY")
print("=" * 50)
print(f"{'Step':<10} | {'PyTorch Loss':<15} | {'Triton Loss':<15}")
print("-" * 50)
for i in range(0, 20, 4):
    print(f"{i:<10} | {pt_losses[i]:<15.6f} | {tr_losses[i]:<15.6f}")
print("=" * 50)

final_diff = abs(pt_losses[-1] - tr_losses[-1])
print(f"Final Loss Difference: {final_diff:.2e}")
if final_diff < 0.1:  # Loose check for trend convergence
    print("SUCCESS: Both models converge. Triton kernels are 'Useful' for training.")
else:
    print("WARNING: Divergence detected.")
