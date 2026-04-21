ENGLISH | 简体中文 | 繁體中文 | 한국어 | ESPAÑOL | 日本語 | हिन्दी | РУССКИЙ | PORTUGUÊS | తెలుగు | FRANÇAIS | DEUTSCH | ITALIANO | TIẾNG VIỆT | العربية | اردو | বাংলা |

STATE-OF-THE-ART PRETRAINED MODELS FOR INFERENCE AND TRAINING

Transformers acts as the model-definition framework for state-of-the-art machine learning with text, computer vision, audio, video, and multimodal models, for both inference and training.

It centralizes the model definition so that this definition is agreed upon across the ecosystem. transformers is the pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference engines (vLLM, SGLang, TGI, ...), and adjacent modeling libraries (llama.cpp, mlx, ...) which leverage the model definition from transformers.

We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be simple, customizable, and efficient.

There are over 1M+ Transformers model checkpoints on the Hugging Face Hub you can use.

INSTALLATION
Transformers works with Python 3.10+, and PyTorch 2.4+.

# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate

# pip
pip install "transformers[torch]"
# uv
uv pip install "transformers[torch]"

QUICKSTART
Get started with Transformers right away with the Pipeline API.

from transformers import pipeline
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")

WHY SHOULD I USE TRANSFORMERS?
1. Easy-to-use state-of-the-art models.
2. Lower compute costs, smaller carbon footprint.
3. Choose the right framework for every part of a model's lifetime.
4. Easily customize a model or an example to your needs.

WHY SHOULDN'T I USE TRANSFORMERS?
* Not a modular toolbox of building blocks; code is intentionally not refactored for researcher iteration.
* Training API is optimized for Transformers models.
* Example scripts are only examples and may need adaptation.

CITATION
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf et al.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
