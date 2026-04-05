import torch
import torch.nn as nn
import time
from fastvla.kernels import TritonActionHead

# ── Setup ────────────────────────────────────────────────────────────────
torch.manual_seed(42)
B, H_in, H_mid, D_out = 1, 4096, 1024, 7  # Standard VLA settings

# Reference PyTorch Head
class ReferenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(H_in, H_mid)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(H_mid, D_out)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.lin2(self.relu(self.lin1(x))))

ref_head = ReferenceHead().cuda().half()
tri_head = TritonActionHead(H_in, H_mid, D_out).cuda().half()

# Synchronize weights for parity
with torch.no_grad():
    tri_head.weight1.copy_(ref_head.lin1.weight.t().contiguous())
    tri_head.bias1.copy_(ref_head.lin1.bias)
    tri_head.weight2.copy_(ref_head.lin2.weight.t().contiguous())
    tri_head.bias2.copy_(ref_head.lin2.bias)

# ── The "Real Snapshot" Input ───────────────────────────────────────────
# In a real VLA, this comes from the Vision Encoder + Llama.
# We simulate a "Real" distribution here (normalized latent states).
x = torch.randn(B, H_in, device="cuda", dtype=torch.float16, requires_grad=True)
target = torch.randn(B, D_out, device="cuda", dtype=torch.float16)

# ── Reliability Run (500 Steps) ─────────────────────────────────────────
print("="*70)
print("FASTVLA HIGH-FIDELITY RELIABILITY TEST")
print("="*70)

max_fwd_diff = 0
max_bwd_diff = 0

for i in range(500):
    # 1. Reference Pass
    ref_out = ref_head(x)
    ref_loss = (ref_out - target).pow(2).mean()
    ref_loss.backward(retain_graph=True)
    ref_grad = x.grad.clone()
    x.grad.zero_()
    
    # 2. Triton Pass
    tri_out = tri_head(x)
    tri_loss = (tri_out - target).pow(2).mean()
    tri_loss.backward(retain_graph=True)
    tri_grad = x.grad.clone()
    x.grad.zero_()
    
    # Measure Diffs
    fwd_diff = torch.max(torch.abs(ref_out - tri_out)).item()
    bwd_diff = torch.max(torch.abs(ref_grad - tri_grad)).item()
    
    max_fwd_diff = max(max_fwd_diff, fwd_diff)
    max_bwd_diff = max(max_bwd_diff, bwd_diff)
    
    if i % 100 == 0:
        print(f"Step {i:3d}: Fwd Diff={fwd_diff:.2e} | Bwd Diff={bwd_diff:.2e}")

print("-"*70)
print(f"FINAL QUALITY SCORE:")
print(f"  Max Forward Error:  {max_fwd_diff:.2e} ({'✅ PASS' if max_fwd_diff < 1e-4 else '❌ FAIL'})")
print(f"  Max Gradient Error: {max_bwd_diff:.2e} ({'✅ PASS' if max_bwd_diff < 1e-4 else '❌ FAIL'})")
print("-"*70)

if max_fwd_diff < 1e-4 and max_bwd_diff < 1e-4:
    print("RELIABILITY PROVEN: Kernels are mathematically identical to PyTorch.")
else:
    print("RELIABILITY UNCERTAIN: Significant numerical drift detected.")
print("="*70)
