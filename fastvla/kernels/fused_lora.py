"""
Fused 4-Bit LoRA Triton GEMM Kernel.

Fuses the LoRA linear projection and residual addition:
    Y = Y_base + alpha * ( (X @ A^T) @ B^T )

In standard PEFT / BitsAndBytes LoRA:
1. h = X @ A^T  [M, r]
2. tmp1 = h @ B^T  [M, N]  <-- Large intermediate tensor allocated in HBM
3. tmp2 = tmp1 * alpha     <-- Another HBM allocation
4. Y = Y_base + tmp2       <-- Third HBM memory round-trip

Across 224 LoRA linear modules in a 32-layer 7B model (q, k, v, o, gate, up, down):
- Standard PEFT creates 672 intermediate HBM allocations and kernel launches per forward step.
- FusedLoraLinear computes alpha * (h @ B^T) + Y_base directly in SRAM via a single fused
  Triton GEMM kernel, completely eliminating intermediate projection buffers in HBM.
- In backward pass, dh = alpha * (dY @ B) is computed into compact [M, r] buffers,
  slashing peak VRAM and accelerating backward passes on commodity GPUs (e.g., Tesla T4).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# Triton Kernels
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _fused_lora_b_add_fwd_kernel(
        h_ptr,          # [M, R]
        b_ptr,          # [N, R] (lora_B weight is [N, R], so B^T is [R, N])
        base_ptr,       # [M, N]
        out_ptr,        # [M, N]
        alpha,          # float32
        M, N, R,        # dimensions
        stride_hm, stride_hr,
        stride_bn, stride_br,
        stride_basem, stride_basen,
        stride_outm, stride_outn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """
        Computes Out = Base + alpha * (H @ B^T) in SRAM.
        H: [M, R], B: [N, R] -> H @ B^T: [M, N]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)

        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_r = offs_r < R

        # Accumulator for H @ B^T tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over R dimension if R > BLOCK_R, else single tile
        for r_start in range(0, R, BLOCK_R):
            curr_r = r_start + offs_r
            mask_curr_r = curr_r < R

            # Load H tile: [BLOCK_M, BLOCK_R] in native dtype for Tensor Cores
            h_ptrs = h_ptr + (offs_m[:, None] * stride_hm + curr_r[None, :] * stride_hr)
            h_tile = tl.load(h_ptrs, mask=mask_m[:, None] & mask_curr_r[None, :], other=0.0)

            # Load B tile: [BLOCK_N, BLOCK_R] in native dtype for Tensor Cores
            b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + curr_r[None, :] * stride_br)
            b_tile = tl.load(b_ptrs, mask=mask_n[:, None] & mask_curr_r[None, :], other=0.0)

            # Hardware Tensor Core Dot Product (FP16/BF16 in -> FP32 acc)
            acc += tl.dot(h_tile, tl.trans(b_tile), out_dtype=tl.float32)

        # Scale by alpha
        lora_val = acc * alpha

        # Load Base output tile: [BLOCK_M, BLOCK_N]
        base_ptrs = base_ptr + (offs_m[:, None] * stride_basem + offs_n[None, :] * stride_basen)
        base_tile = tl.load(base_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        out_val = (base_tile + lora_val).to(out_ptr.dtype.element_ty)

        # Store directly to global output
        out_ptrs = out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
        tl.store(out_ptrs, out_val, mask=mask_m[:, None] & mask_n[None, :])


    @triton.jit
    def _fused_lora_dh_kernel(
        dy_ptr,         # [M, N]
        b_ptr,          # [N, R]
        dh_ptr,         # [M, R]
        alpha,          # float32
        M, N, R,
        stride_dym, stride_dyn,
        stride_bn, stride_br,
        stride_dhm, stride_dhr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """
        Computes dH = alpha * (dY @ B)
        dY: [M, N], B: [N, R] -> dH: [M, R]
        """
        pid_m = tl.program_id(0)
        pid_r = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_n = tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_r = offs_r < R

        acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            curr_n = n_start + offs_n
            mask_curr_n = curr_n < N

            # Load dY tile: [BLOCK_M, BLOCK_N] in native dtype
            dy_ptrs = dy_ptr + (offs_m[:, None] * stride_dym + curr_n[None, :] * stride_dyn)
            dy_tile = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_curr_n[None, :], other=0.0)

            # Load B tile: [BLOCK_N, BLOCK_R] in native dtype
            b_ptrs = b_ptr + (curr_n[:, None] * stride_bn + offs_r[None, :] * stride_br)
            b_tile = tl.load(b_ptrs, mask=mask_curr_n[:, None] & mask_r[None, :], other=0.0)

            acc += tl.dot(dy_tile, b_tile, out_dtype=tl.float32)

        dh_val = (acc * alpha).to(dh_ptr.dtype.element_ty)

        dh_ptrs = dh_ptr + (offs_m[:, None] * stride_dhm + offs_r[None, :] * stride_dhr)
        tl.store(dh_ptrs, dh_val, mask=mask_m[:, None] & mask_r[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────────

class FusedLoraBAddFunction(torch.autograd.Function):
    """
    Fused Triton autograd operator for:
        Y = Y_base + alpha * (H @ B^T)
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor, b_weight: torch.Tensor, base_out: torch.Tensor, alpha: float) -> torch.Tensor:
        orig_shape = base_out.shape
        M = base_out.numel() // base_out.shape[-1]
        N = base_out.shape[-1]
        R = h.shape[-1]

        h_2d = h.view(M, R).contiguous()
        b_2d = b_weight.contiguous()  # [N, R]
        base_2d = base_out.view(M, N).contiguous()

        # Check if Triton can be used
        use_triton = HAS_TRITON and h.is_cuda and b_weight.is_cuda and base_out.is_cuda

        if use_triton:
            out_2d = torch.empty((M, N), device=base_out.device, dtype=base_out.dtype)

            BLOCK_M = 32 if M < 64 else 64
            BLOCK_N = 64
            BLOCK_R = max(16, triton.next_power_of_2(min(R, 64)))

            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _fused_lora_b_add_fwd_kernel[grid](
                h_2d,
                b_2d,
                base_2d,
                out_2d,
                float(alpha),
                M, N, R,
                h_2d.stride(0), h_2d.stride(1),
                b_2d.stride(0), b_2d.stride(1),
                base_2d.stride(0), base_2d.stride(1),
                out_2d.stride(0), out_2d.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_R=BLOCK_R,
            )
            out = out_2d.view(orig_shape)
        else:
            # PyTorch fallback: out = base + alpha * (h @ b^T)
            lora_out = torch.matmul(h_2d, b_2d.t()) * alpha
            out = (base_2d + lora_out).view(orig_shape)

        ctx.save_for_backward(h_2d, b_2d)
        ctx.alpha = alpha
        ctx.orig_base_shape = orig_shape
        ctx.use_triton = use_triton
        ctx.M = M
        ctx.N = N
        ctx.R = R

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h_2d, b_2d = ctx.saved_tensors
        alpha = ctx.alpha
        M, N, R = ctx.M, ctx.N, ctx.R

        grad_2d = grad_output.view(M, N).contiguous()

        grad_h = None
        grad_b = None
        grad_base = grad_output  # d(Y)/d(Base) = I

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.use_triton and ctx.needs_input_grad[0]:
                dh_2d = torch.empty((M, R), device=grad_output.device, dtype=h_2d.dtype)
                BLOCK_M = 32 if M < 64 else 64
                BLOCK_N = 64
                BLOCK_R = max(16, triton.next_power_of_2(min(R, 64)))

                grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(R, BLOCK_R))
                _fused_lora_dh_kernel[grid](
                    grad_2d,
                    b_2d,
                    dh_2d,
                    float(alpha),
                    M, N, R,
                    grad_2d.stride(0), grad_2d.stride(1),
                    b_2d.stride(0), b_2d.stride(1),
                    dh_2d.stride(0), dh_2d.stride(1),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_R=BLOCK_R,
                )
                grad_h = dh_2d.view_as(h_2d)
            else:
                # dh = alpha * (grad_output @ B)
                grad_h = (torch.matmul(grad_2d, b_2d) * alpha).view_as(h_2d)

            if ctx.needs_input_grad[1]:
                # dB = alpha * (grad_output^T @ H) -> [N, R]
                grad_b = torch.matmul(grad_2d.t(), h_2d) * alpha

        return grad_h, grad_b, grad_base, None


def fused_lora_b_add(h: torch.Tensor, b_weight: torch.Tensor, base_out: torch.Tensor, alpha: float) -> torch.Tensor:
    """Convenience wrapper for FusedLoraBAddFunction."""
    return FusedLoraBAddFunction.apply(h, b_weight, base_out, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Fused LoRA Module & PEFT Monkey-Patching
# ─────────────────────────────────────────────────────────────────────────────

class FusedLoraLinear(nn.Module):
    """
    Fused replacement for PEFT LoRA linear layers (Linear4bit / Linear).
    Wraps the underlying base layer and LoRA adapter weights with the fused Triton GEMM.
    """

    def __init__(self, original_peft_layer: nn.Module):
        super().__init__()
        self.base_layer = original_peft_layer.base_layer
        self.lora_A = original_peft_layer.lora_A
        self.lora_B = original_peft_layer.lora_B
        self.lora_dropout = getattr(original_peft_layer, "lora_dropout", nn.ModuleDict())
        self.scaling = getattr(original_peft_layer, "scaling", {})
        self.active_adapter = getattr(original_peft_layer, "active_adapter", "default")
        self.active_adapters = getattr(original_peft_layer, "active_adapters", ["default"])
        self.in_features = original_peft_layer.in_features
        self.out_features = original_peft_layer.out_features
        self.r = getattr(original_peft_layer, "r", {})

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Base quantized forward pass (e.g., 4-bit bnb or linear)
        base_result = self.base_layer(x, *args, **kwargs)

        # 2. Fused LoRA branches
        for adapter_name in self.active_adapters:
            if adapter_name not in self.lora_A:
                continue

            lora_A = self.lora_A[adapter_name]
            lora_B = self.lora_B[adapter_name]
            dropout = self.lora_dropout[adapter_name] if adapter_name in self.lora_dropout else nn.Identity()
            scaling = self.scaling[adapter_name] if adapter_name in self.scaling else 1.0

            # Cast x to lora dtype if needed
            x_in = dropout(x).to(lora_A.weight.dtype)

            # Intermediate compact projection: h = x @ A^T [B, S, r]
            h = lora_A(x_in)

            # Fused Triton GEMM: base_result = base_result + scaling * (h @ B^T)
            b_weight = lora_B.weight if hasattr(lora_B, "weight") else lora_B[0].weight
            base_result = fused_lora_b_add(h, b_weight, base_result, scaling)

        return base_result


def apply_fused_lora(model: nn.Module) -> int:
    """
    Recursively scans the model and replaces all PEFT LoRA linear layers
    (e.g., q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj across 32 layers)
    with FusedLoraLinear to eliminate intermediate HBM memory traffic.

    Args:
        model: PyTorch model with PEFT LoRA adapters.

    Returns:
        Number of modules successfully converted to FusedLoraLinear.
    """
    num_fused = 0

    for name, module in list(model.named_modules()):
        # Check for PEFT LoRA layers
        has_lora_branches = hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer")
        if has_lora_branches and not isinstance(module, FusedLoraLinear):
            # Find parent module to replace child
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            fused_layer = FusedLoraLinear(module)
            setattr(parent, parts[-1], fused_layer)
            num_fused += 1

    return num_fused
