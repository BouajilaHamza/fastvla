# FastVLA: Democratizing VLA Fine-Tuning

FastVLA makes Vision-Language-Action (VLA) model fine-tuning accessible on **single consumer GPUs** — specifically, a free Tesla T4 (15GB VRAM).

> **Goal:** Do for VLA fine-tuning what Unsloth did for LLM fine-tuning.

## 📈 Real Benchmark Results (Tesla T4, 15GB)

### OpenVLA-7B: Before vs After Optimization

| Metric | FP16 (Naive) | 4-bit QLoRA (FastVLA) | Improvement |
|--------|-------------|----------------------|-------------|
| **VRAM (load)** | 13.20 GB | **4.32 GB** | **67% less** |
| **VRAM (training peak)** | OOM risk | **10.35 GB** | **Fits on T4** |
| **Trainable params** | 7,541,237,184 | **39,976,960** | **188x fewer** |
| **Trainable %** | 100% | **1.02%** | |
| **Forward pass** | N/A | **1,891 ms** | |
| **Backward pass** | N/A | **522 ms** | |
| **Steps/sec** | N/A | **0.89** | |
| **Loss (50 steps)** | N/A | **20.0 → 10.35** | ✅ Converging |

### Key Achievement

> **OpenVLA-7B fits on a Tesla T4 (15GB) with 4-bit QLoRA.**
> Without optimization: 13.20 GB load → only 2.4 GB free → can't train.
> With 4-bit QLoRA: 4.32 GB load → 11.3 GB free → fine-tuning works.

## 🚀 Features

- **4-bit QLoRA**: Fine-tune a 7B VLA on 15GB VRAM (free Colab/T4)
- **LoRA Adapters**: Only 1.02% trainable params (40M vs 7.5B)
- **Triton Action Head**: Fused MLP kernel (Linear→ReLU→Linear→Tanh) with gradient checkpointing
- **Flexible Architecture**: Any HuggingFace vision encoder + any LLM
- **CPU/GPU Auto-Select**: Runs on CPU for testing, GPU for training
- **Complete Training Loop**: Checkpointing, evaluation, logging
- **Dummy Mode**: Tiny random-weight model for fast validation (no downloads)

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/FastVLA.git
cd FastVLA
pip install -r requirements.txt
```

## 📖 Quick Start

### Fine-Tune OpenVLA-7B on T4 (Recommended)

```python
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model (fits on T4: 4.32 GB)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

# Apply LoRA (only 40M trainable params)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)

# Training step
pixel_values = torch.randn(1, 6, 224, 224, device="cuda", dtype=torch.float32)
input_ids = torch.randint(0, 32000, (1, 64), device="cuda")
labels = input_ids.clone()

outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
outputs.loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### Use FastVLA's High-Level API

```python
from fastvla import FastVLAModel

# Load any VLA with 4-bit QLoRA
model = FastVLAModel.from_pretrained(
    vision_encoder_name="google/vit-base-patch16-224",
    llm_name="meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    use_peft=True,
    gradient_checkpointing=True,
)

# Or use dummy mode for fast testing (no downloads)
model = FastVLAModel.from_pretrained(dummy=True)
```

## 🧪 Testing

```bash
# Full test suite
pytest tests/ -v

# Real OpenVLA benchmark (requires GPU)
python benchmark_real_vla.py

# Fine-tune OpenVLA-7B
python finetune_openvla.py --steps 100 --batch 1

# GPU kernel benchmarks
python benchmark_gpu.py

# Quick CPU validation (dummy model)
python mvp_test.py
```

## 🏗️ Project Structure

```
fastvla/
├── config.py                    # Model configuration (PretrainedConfig)
├── model.py                     # FastVLAModel — flexible, dummy mode, CPU/GPU
├── optimization.py              # 4-bit quantization, 8-bit optimizer, PEFT
├── training.py                  # FastVLATrainer with checkpointing
├── benchmarking.py              # Performance profiling
├── utils.py                     # get_device() auto-select utility
├── kernels/
│   ├── __init__.py              # Auto-dispatch: Triton (GPU) or PyTorch (CPU)
│   ├── fusion.py                # Vision-language fusion (Triton + backward)
│   ├── action.py                # Fused MLP action head (Triton + backward)
│   ├── action_head.py           # TritonActionHead module with grad checkpointing
│   ├── multicam.py              # Multi-camera packing
│   └── cpu_fallbacks.py         # Pure PyTorch fallbacks for CPU
└── data/
    ├── collator.py              # Multi-modal data collator
    └── datasets.py              # Robotics dataset loaders

benchmark_real_vla.py            # OpenVLA-7B on T4 benchmark
finetune_openvla.py              # Fine-tuning script with CLI args
benchmark_gpu.py                 # GPU kernel benchmarks
mvp_test.py                      # Quick CPU validation (dummy model)
```

## 🔬 Technical Details

### How We Fit OpenVLA-7B on 15GB

| Technique | What It Does | VRAM Saved |
|-----------|-------------|------------|
| **4-bit NF4 quantization** | LLaMA-2-7B: 14GB → 4GB | ~10 GB |
| **LoRA adapters** | Train 40M params instead of 7.5B | ~8 GB (optimizer state) |
| **Gradient checkpointing** | Recompute activations vs store | ~2-3 GB |
| **Frozen vision encoder** | No gradients for DINOv2 + SigLIP | ~1 GB |
| **Efficient batching** | Batch size 1, seq len 64 | Minimal |

**Total**: 13.20 GB → **4.32 GB load** / **10.35 GB training peak**

### Triton Action Head

The action head fuses `Linear → ReLU → Linear → Tanh` into a single Triton kernel:
- **Forward**: Fused MLP with tiling over hidden dimension (fits T4's 64KB SRAM)
- **Backward**: Gradient checkpointing — recomputes intermediate activations on-the-fly
- **Numerical parity**: 5.66e-07 max diff vs standard PyTorch
- **Auto-dispatch**: Uses Triton on GPU, falls back to PyTorch on CPU

## 📊 Benchmark Details

### Training Throughput (OpenVLA-7B, T4)

```
100 steps @ batch_size=1, seq_len=64, image=224×224
Total time:     122.1s (2.0 min)
Avg step time:  1,221ms
Steps/sec:      0.82
Peak VRAM:      10.35 GB (5.28 GB free)
Loss:           17.57 → 10.33 (Δ = -7.23, converging)

Step time breakdown:
  Step 1:   4,551ms (first step overhead)
  Step 10:  1,080ms (stabilized)
  Step 50:  1,659ms
  Step 100: 1,046ms (stable ~1s/step)
```

> **Note:** Step time is dominated by the vision encoder (DINOv2-L + SigLIP-SO400M fused backbone processing 224×224 images). The 4-bit QLoRA LLM adds minimal overhead. Throughput improvements expected with Unsloth patches and smaller image sizes.

## 🤝 Contributing

Contributions welcome! Priority areas:
- Unsloth integration for LLM speedup
- Real robotics dataset fine-tuning (LIBERO, Franka Kitchen)
- Flash attention for vision encoder
- Multi-GPU support

## 📜 License

MIT License — see [LICENSE](LICENSE)
