<div align="center">

<svg width="120" height="120" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M20 20L50 80L80 20H65L50 55L35 20H20Z" fill="#CCFF00"/>
    <rect x="20" y="85" width="60" height="4" fill="#CCFF00"/>
    <path d="M85 45L95 50L85 55V45Z" fill="#CCFF00"/>
</svg>

# `FASTVLA`

## The fast, memory-efficient fine-tuning library for Vision-Language-Action models.

### Fine-tune any 7B robot policy — any language, any task — for under $1. On one GPU.

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![Unsloth](https://img.shields.io/badge/Unsloth-7B61FF?style=for-the-badge&logo=unsloth&logoColor=white)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/PEFT-000000?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/huggingface/peft)
[![Modal](https://img.shields.io/badge/Modal-000000?style=for-the-badge&logoColor=white)](https://modal.com)
[![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

[**Arabic Datasets**](docs/datasets/ARABIC_DATASETS.md) | [**RL Technical Report**](docs/reports/RL_TECHNICAL_REPORT.md) | [**Model on HF Hub**](https://huggingface.co/hamzabouajila/fastvla-arabic-precision)

</div>

---

## What is FastVLA?

**FastVLA is to VLA fine-tuning what Unsloth is to LLM fine-tuning.**

Vision-Language-Action models like OpenVLA map camera observations and language instructions to robot actions. Fine-tuning them for new tasks, new languages, or new domains currently requires A100/H100 clusters and weeks of engineering.

FastVLA removes those constraints. By combining Unsloth 4-bit kernels, custom Triton action heads, and temporal action chunking, the entire pipeline — BC pretraining + PPO reinforcement learning — fits on a single NVIDIA L4 for under $1.

---

## Key Features

- **Multi-lingual Support:** Native support for Arabic and other non-English instruction sets.
- **Extreme Efficiency:** Fine-tune 7B parameter models on a single GPU (L4/T4) for under $1.
- **Real-time Performance:** Up to 7.6x faster inference than standard 4-bit baselines.
- **Robust RL Integration:** Stable PPO and GRPO implementations for robotic manipulation.
- **Cloud Native:** Ready-to-use Modal deployment scripts for instant scaling.

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

## Results at a Glance

### NVIDIA L4 Performance

| Metric | OpenVLA-7B Baseline | FastVLA | Improvement |
| :--- | :--- | :--- | :--- |
| Inference Latency | 1420 ms | **186 ms** | **7.6x faster** |
| Peak VRAM | 16.5 GB | **4.45 GB** | **73% less** |
| Control Frequency | 0.7 Hz | **5.4 Hz** | Real-time capable |

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

## Technical Performance

For a deep dive into our Reinforcement Learning results, policy consolidation behavior, and system stability during the PushT benchmarks, read the [RL Technical Report](docs/reports/RL_TECHNICAL_REPORT.md).

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
