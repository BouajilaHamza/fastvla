---
language:
- ar
- en
tags:
- robotics
- vla
- vision-language-action
- flow-matching
- spatial-compression
- physical-ai
- 4bit-quantization
license: apache-2.0
datasets:
- hamzabouajila/ar-pusht-image
- hamzabouajila/ar-libero-10-image
pipeline_tag: robotics
---

# FastVLA: High-Throughput Arabic-Instructable Vision-Language-Action Policy

FastVLA is a compute-efficient, 4-bit NF4 quantized Vision-Language-Action (VLA) policy engineered to train and infer continuous robotic manipulation trajectories on consumer and cloud edge GPUs (NVIDIA Tesla T4, RTX 3090/4090, and L4).

## Model Architecture

- **Vision Backbone:** Pretrained `vit_so400m_patch14_siglip_224` (1152-dim vision features).
- **Spatial Token Compression:** 2x2 [`SpatialPatchMerger`](https://github.com/BouajilaHamza/fastvla) reducing 256 patches to 64 tokens per camera view (-75% token reduction).
- **Language Backbone:** Qwen2.5-3B (or Qwen2-1.5B) quantized to 4-bit NF4 with trainable Low-Rank Adapters (LoRA $r=16, \alpha=32$).
- **Continuous Action Head:** SE(3) Flow Matching Action Head predicting 16-step continuous action chunks via high-order Midpoint ODE solvers in 2 steps.

## Empirical Performance (Tesla T4 Benchmark)

| Metric | Standard 4-Bit Baseline | FastVLA | Improvement |
| :--- | :---: | :---: | :---: |
| **Vision Tokens / View** | 256 tokens | **64 tokens** | **-75%** |
| **Training Step Time** | 559.0 ms | **312.3 ms** | **-44% latency** |
| **Training Throughput** | 1.79 steps/s | **3.20 steps/s** | **+79% faster** |
| **Inference Control Rate** | ~4.5 Hz | **9.0–9.2 Hz** | **2.0x faster** |
| **Peak GPU VRAM** | 6.02 GB | **6.13 GB** | Parity |

## Evaluation Scope & Limitations

### What This Model Does Well
1. **Continuous Trajectory Regression:** Predicts multi-step continuous action chunks (Push-T 2D planar and Franka 7-DoF contact coordinates).
2. **Linguistic Robustness:** Resilient to dialectal Arabic variations (Levantine, Egyptian, Gulf synonyms) and common orthographic typing noise with less than 10.1% trajectory drift.
3. **Zero-Shot Semantic Transfer:** When probed on unseen object and spatial relationship tasks from `LIBERO-Spatial` and `LIBERO-Object`, the model exhibits only +3.4% error gap relative to in-distribution training data.

### Limitations & Ethical Considerations
- **Offline Trajectory vs Closed-Loop Grasping:** Offline trajectory L2 error does not guarantee closed-loop 500-step task completion in complex multi-contact physics. Full closed-loop simulator rollouts are required for validation.
- **Physical Safety:** Never deploy continuous neural policies on physical robotic arms without low-level joint velocity clamps, emergency stop hardware interrupts, and workspace safety bounding boxes.

## Quick Usage

```python
import torch
from fastvla import FastVLAModel

# Initialize in 4-bit NF4 with Flow Matching
model = FastVLAModel.from_pretrained("fastvla-3b-arabic", load_in_4bit=True)
model.eval()

# Forward pass: 1 camera view [B, 1, 3, 224, 224]
pixel_values = torch.randn(1, 1, 3, 224, 224, device="cuda", dtype=torch.float16)
input_ids = torch.tensor([[101, 2054, 102]], device="cuda")

with torch.no_grad():
    actions, _ = model(pixel_values=pixel_values, input_ids=input_ids)
    print("Predicted 7-DoF Action Chunk Shape:", actions.shape)
```
