import torch
import triton
import triton.language as tl


@triton.jit
def _multi_cam_pack_forward_kernel(
    # Pointers to input tensors
    cams_ptr,
    # Output pointer
    output_ptr,
    # Tensor dimensions
    B,
    C,
    H,
    W,
    D,
    # Strides for cams [B, num_cams, C, H, W]
    stride_b,
    stride_cam,
    stride_c,
    stride_h,
    stride_w,
    # Strides for output [B, C*num_cams, H, W]
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    # Number of cameras
    num_cams: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for packing multiple camera views into a single tensor.

    Args:
        cams_ptr: [B, num_cams, C, H, W] Input camera views
        output_ptr: [B, C*num_cams, H, W] Output packed tensor
    """
    # Parallelize over batch, channels, height, and width
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Calculate camera and channel indices
    cam_idx = pid_c // C
    chan_idx = pid_c % C

    # Check bounds
    if (pid_b >= B) or (cam_idx >= num_cams) or (pid_h >= H) or (pid_w >= W):
        return

    # Calculate input offset
    input_offset = (
        pid_b * stride_b
        + cam_idx * stride_cam
        + chan_idx * stride_c
        + pid_h * stride_h
        + pid_w * stride_w
    )

    # Calculate output offset
    output_c = cam_idx * C + chan_idx
    output_offset = (
        pid_b * stride_ob + output_c * stride_oc + pid_h * stride_oh + pid_w * stride_ow
    )

    # Load and store
    val = tl.load(cams_ptr + input_offset)
    tl.store(output_ptr + output_offset, val)


@triton.jit
def _multi_cam_pack_backward_kernel(
    grad_output_ptr,
    grad_cams_ptr,
    # Tensor dimensions and strides...
    B,
    C,
    H,
    W,
    D,
    stride_grad_out_b,
    stride_grad_out_c,
    stride_grad_out_h,
    stride_grad_out_w,
    stride_grad_cam_b,
    stride_grad_cam_cam,
    stride_grad_cam_c,
    stride_grad_cam_h,
    stride_grad_cam_w,
    num_cams: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for multi-camera packing."""
    # Similar to forward but in reverse
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    cam_idx = pid_c // C
    chan_idx = pid_c % C

    if (pid_b >= B) or (cam_idx >= num_cams) or (pid_h >= H) or (pid_w >= W):
        return

    # Calculate output gradient offset
    output_c = cam_idx * C + chan_idx
    grad_out_offset = (
        pid_b * stride_grad_out_b
        + output_c * stride_grad_out_c
        + pid_h * stride_grad_out_h
        + pid_w * stride_grad_out_w
    )

    # Calculate input gradient offset
    grad_cam_offset = (
        pid_b * stride_grad_cam_b
        + cam_idx * stride_grad_cam_cam
        + chan_idx * stride_grad_cam_c
        + pid_h * stride_grad_cam_h
        + pid_w * stride_grad_cam_w
    )

    # Load and store gradients
    grad = tl.load(grad_output_ptr + grad_out_offset)
    tl.store(grad_cams_ptr + grad_cam_offset, grad)


# Python wrappers
def multi_cam_pack_forward(cams: torch.Tensor) -> torch.Tensor:
    """
    Pack multiple camera views into a single tensor.

    Args:
        cams: [B, num_cams, C, H, W] Input camera views

    Returns:
        [B, C*num_cams, H, W] Packed tensor
    """
    B, num_cams, C, H, W = cams.shape

    # Allocate output
    output = torch.empty((B, num_cams * C, H, W), device=cams.device, dtype=cams.dtype)

    # Launch kernel
    def grid(_):
        return (B, num_cams * C, H, W)

    _multi_cam_pack_forward_kernel[grid](
        cams,
        output,
        B,
        C,
        H,
        W,
        3,  # D=3 for RGB
        cams.stride(0),
        cams.stride(1),
        cams.stride(2),
        cams.stride(3),
        cams.stride(4),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        num_cams=num_cams,
        BLOCK_SIZE=32,  # Tune based on hardware
    )
    return output


def multi_cam_pack_backward(grad_output: torch.Tensor, num_cams: int) -> torch.Tensor:
    """Backward pass for multi-camera packing."""
    B, C_total, H, W = grad_output.shape
    C = C_total // num_cams

    # Allocate gradient tensor
    grad_cams = torch.zeros(
        (B, num_cams, C, H, W), device=grad_output.device, dtype=grad_output.dtype
    )

    # Launch kernel
    def grid(_):
        return (B, num_cams * C, H, W)

    _multi_cam_pack_backward_kernel[grid](
        grad_output,
        grad_cams,
        B,
        C,
        H,
        W,
        3,  # D=3 for RGB
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_cams.stride(0),
        grad_cams.stride(1),
        grad_cams.stride(2),
        grad_cams.stride(3),
        grad_cams.stride(4),
        num_cams=num_cams,
        BLOCK_SIZE=32,  # Match forward pass
    )
    return grad_cams
