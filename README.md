# 🚀 FastVLA: Ultra-Efficient VLA Fine-Tuning

FastVLA is a high-performance library designed to bring **Vision-Language-Action (VLA)** fine-tuning to commodity hardware. By leveraging **Triton-accelerated kernels**, **4-bit QLoRA**, and **Unsloth-inspired memory optimizations**, FastVLA enables training 7B+ parameter models on a single Tesla T4 (15GB VRAM).

> [!IMPORTANT]
> **Goal:** Democratize VLA training. If you have a free Google Colab or a T4 instance, you can now fine-tune state-of-the-art robotics models.

---

## 📈 Real-World Benchmark: PushT (Tesla T4)

We've verified FastVLA by fine-tuning **OpenVLA-7B** on the `lerobot/pusht_image` dataset (Real Robotics Data).

| Metric | Results on Tesla T4 (15GB) | Status |
| :--- | :--- | :--- |
| **VRAM Usage** | **5.38 GB** (Training Peak) | 🚀 Ultra-Light |
| **Throughput** | **1.42s / step** (~0.7 steps/sec) | ⚡ Fast |
| **Model Size** | 7.3 Billion Parameters (4-bit) | 🧠 Full Scale |
| **Learning Signal** | 19.6 → 1.15 Loss (in 400 steps) | ✅ Verified |

> **Key Takeaway:** Using 4-bit QLoRA and our custom Triton Action Head, you can fine-tune a 7B VLA while leaving **~10GB of VRAM free** for other processes.

---

## ✨ Features

- **4-bit QLoRA**: Reduces 7B model memory from 28GB to 4.3GB with near-zero quality loss.
- **Triton Action Head**: Fused `Linear → ReLU → Linear → Tanh` kernel with **Gradient Checkpointing** to save activation memory.
- **Corrected VLA Objective**: Implements proper **discretized action prediction** (256 bins) matching the original OpenVLA pre-training.
- **Unsloth Integration**: Infrastructure ready for Unsloth's ultra-fast LLM patches.
- **Robotics-First**: Built-in support for `PushT` and `LIBERO` datasets via HuggingFace `datasets`.

---

## 🛠️ Installation

FastVLA uses `uv` for lightning-fast dependency management.

```bash
# Clone the repo
git clone https://github.com/BouajilaHamza/FastVLA.git
cd FastVLA

# Install dependencies using uv
uv sync
```

---

## 📖 Quick Start

### Fine-Tune on PushT (Real Robotics)
Run our optimized PushT script. It automatically handles image normalization and action discretization.

```bash
uv run scripts/finetune_pusht.py --steps 2000 --batch 1 --lr 1e-4
```

### High-Level API
```python
from fastvla import FastVLAModel

# Load any VLA with 4-bit QLoRA optimization
model = FastVLAModel.from_pretrained(
    vision_encoder_name="google/vit-base-patch16-224",
    llm_name="meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    use_peft=True,
    gradient_checkpointing=True,
)

# Inference (predict continuous actions)
action = model.generate(images=image_tensor, input_ids=text_ids)
```

---

## 🏗️ Project Structure

- `fastvla/`: Core library containing the model architecture and Triton kernels.
  - `kernels/`: Fused Triton kernels for Action Heads and Fusion.
- `scripts/`: Production-ready fine-tuning and benchmarking scripts.
  - `finetune_pusht.py`: The recommended script for PushT fine-tuning.
  - `finetune_libero.py`: Configuration for the LIBERO simulation benchmark.
- `results/`: Standardized output for training logs and benchmark JSONs.
- `tests/`: Comprehensive test suite for numerical parity and kernel stability.

---

## 🧪 Testing & Validation

We enforce strict **Numerical Parity** between our Triton kernels and PyTorch benchmarks.

```bash
# Run all tests
uv run pytest tests/ -v

# Run GPU kernel benchmarks
uv run python scripts/benchmark_gpu.py
```

---

## 🤝 Contributing & Roadmap
- [ ] **Unsloth v2 Integration**: Direct patching for vision encoders.
- [ ] **FlashAttention-3**: Support for latest Hopper/Ada kernels.
- [ ] **Multi-Camera Fusing**: Optimized packing for 3+ camera setups.

---

## 📜 License
MIT License. Created by the FastVLA Research Team.
