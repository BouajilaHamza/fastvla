BITSANDBYTES
bitsandbytes enables accessible large language models via k-bit quantization for PyTorch.

MAIN FEATURES:
* 8-bit optimizers: Block-wise quantization for 32-bit performance at lower memory.
* LLM.int8(): 8-bit quantization for inference without performance degradation.
* QLoRA: 4-bit quantization for efficient training.

SYSTEM REQUIREMENTS
* Python 3.10+
* PyTorch 2.3+
* Supports Linux (x86-64, aarch64), Windows, and macOS (M1+).
* Accelerator support for NVIDIA (CUDA), AMD (ROCm), Intel (XPU/Gaudi), and Apple (Metal/MPS).

LICENSE
bitsandbytes is MIT licensed.

HOW TO CITE US
QLORA:
@article{dettmers2023qlora,
  title={Qlora: Efficient finetuning of quantized llms},
  author={Dettmers, Tim et al.},
  year={2023}
}

LLM.INT8():
@article{dettmers2022llmint8,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim et al.},
  year={2022}
}
