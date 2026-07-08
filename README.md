<div align="center">

<svg width="120" height="120" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M20 20L50 80L80 20H65L50 55L35 20H20Z" fill="#CCFF00"/>
    <rect x="20" y="85" width="60" height="4" fill="#CCFF00"/>
    <path d="M85 45L95 50L85 55V45Z" fill="#CCFF00"/>
</svg>

# `FASTVLA`

## The fast, memory-efficient fine-tuning and adaptation layer for Vision-Language-Action models.

### Fine-tune any 7B robot policy — any language, any task — for under $1. On budget GPUs.

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![Unsloth](https://img.shields.io/badge/Unsloth-7B61FF?style=for-the-badge&logo=unsloth&logoColor=white)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/PEFT-000000?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/huggingface/peft)
[![Modal](https://img.shields.io/badge/Modal-000000?style=for-the-badge&logoColor=white)](https://modal.com)
[![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

[**Arabic Datasets**](docs/datasets/ARABIC_DATASETS.md) | [**RL Technical Report**](docs/reports/RL_TECHNICAL_REPORT.md) | [**Model on HF Hub**](https://huggingface.co/hamzabouajila/fastvla-arabic-hero)

</div>

---

## 📢 Transparency Update (April 2026)

We have recently identified and resolved several architectural bugs and measurement discrepancies in our initial benchmarks. We are committed to transparency and reproducibility.
- **Metric Correction:** Previous speedup claims (7.6x) were based on flawed measurement methodologies. We have adjusted our reporting to a more realistic **~2x throughput improvement** vs standard 4-bit QLoRA.
- **Bug Fixes:** Resolved issues with double-model loading and incorrect quantization flags.
- **Active Issues:** See [Issue #1](https://github.com/BouajilaHamza/fastvla/issues/1) and [Issue #2](https://github.com/BouajilaHamza/fastvla/issues/2) for full details.

---

## What is FastVLA?

**FastVLA is the optimization and adaptation layer for VLA models.**

Fine-tuning Vision-Language-Action models (like OpenVLA) for new tasks, languages, or domains traditionally requires massive compute clusters. FastVLA removes these barriers by combining **Unsloth 4-bit kernels**, **custom Triton action heads**, and **memory-efficient RL (PPO/GRPO)**.

FastVLA positions itself as a **training and export bridge**:
1. **Train/Fine-tune** on budget GPUs (L4/T4) for under $1.
2. **Optimize** for specific task domains and non-English instructions (e.g., Arabic).
3. **Export** models in high-performance formats (e.g., VLASH-compatible) for ultra-fast runtime deployment.

---

## Key Features

- **Multi-lingual Support:** Native support for Arabic and other non-English instruction sets.
- **Extreme Efficiency:** Fine-tune 7B parameter models on a single GPU (L4/T4) for under $1.
- **Custom Kernels:** Leverages Triton for optimized multi-modal fusion and action decoding.
- **High Capacity:** Supports the full 32-layer LLM backbones without truncation for maximum reasoning power.
- **Deployment Ready:** Designed to export models for real-time robotic control loops.

---

## Repository Structure

The repository is organized for clarity and ease of use:

- `fastvla/`: Core library containing model definitions, kernels, and RL trainers.
- `scripts/`: Organized tools for deployment and training:
    - `scripts/training/`: Core BC and RL training scripts.
    - `scripts/modal/`: Modal.com deployment and simulation scripts.
    - `scripts/dataset/`: Arabic localization and dataset translation tools.
    - `scripts/evaluation/`: Benchmarking and success-rate evaluation tools.
- `docs/`: In-depth documentation for [Arabic Datasets](docs/datasets/ARABIC_DATASETS.md) and [Technical Reports](docs/reports/RL_TECHNICAL_REPORT.md).
- `tests/`: Comprehensive test suite for kernels, data, and model stability.

---

## Performance (NVIDIA L4)

*Metrics are tentative and pending a full reproducible correction run.*

| Metric | standard 4-bit QLoRA | FastVLA (Optimized) | Improvement |
| :--- | :--- | :--- | :--- |
| Training Throughput | ~8 samples/sec | **~16 samples/sec** | **~2.0x faster** |
| Peak VRAM (Batch 16) | ~12.5 GB | **~6.8 GB** | **~45% reduction** |
| Inference Latency | ~1.4s | **~0.7s** | **Improved** |

> Note: Real-time capable 5Hz+ control is achievable when utilizing action chunking and exported runtime engines.

---

## Installation

```bash
git clone https://github.com/BouajilaHamza/fastvla.git
cd fastvla
uv sync
```

---

## Quickstart

### Fine-tune on PushT (Arabic)

```bash
# 1. BC pretraining on Modal
modal run scripts/training/train_scratch_relative.py --bc-epochs 10

# 2. RL refinement with GRPO
modal run scripts/modal/modal_rl_grpo.py --epochs 100
```

For more details on our Arabic-native datasets and localization process, see [ARABIC_DATASETS.md](docs/datasets/ARABIC_DATASETS.md).

---

## License & Citation

Apache-2.0 License.

```bibtex
@software{fastvla2026,
  author  = {Bouajila, Hamza},
  title   = {FastVLA: Efficient Fine-Tuning for Vision-Language-Action Models},
  url     = {https://github.com/BouajilaHamza/fastvla},
  year    = {2026}
}
```
