UNSLOTH STUDIO LETS YOU RUN AND TRAIN MODELS LOCALLY.

Features • Quickstart • Notebooks • Documentation

⚡ GET STARTED

MACOS, LINUX, WSL:
curl -fsSL https://unsloth.ai/install.sh | sh

WINDOWS:
irm https://unsloth.ai/install.ps1 | iex

COMMUNITY:
* Discord
* 𝕏 (Twitter)
* Reddit

⭐ FEATURES
Unsloth Studio (Beta) lets you run and train text, audio, embedding, vision models on Windows, Linux and macOS.

INFERENCE
* Search + download + run models including GGUF, LoRA adapters, safetensors
* Export models: Save or export models to GGUF, 16-bit safetensors and other formats.
* Tool calling: Support for self-healing tool calling and web search
* Code execution: lets LLMs test code in Claude artifacts and sandbox environments
* Auto-tune inference parameters and customize chat templates.
* We work directly with teams behind gpt-oss, Qwen3, Llama 4, Mistral, Gemma 1-3, and Phi-4, where we’ve fixed bugs that improve model accuracy.
* Upload images, audio, PDFs, code, DOCX and more file types to chat with.

TRAINING
* Train and RL 500+ models up to 2x faster with up to 70% less VRAM, with no accuracy loss.
* Custom Triton and mathematical kernels. See some collabs we did with PyTorch and Hugging Face.
* Data Recipes: Auto-create datasets from PDF, CSV, DOCX etc. Edit data in a visual-node workflow.
* Reinforcement Learning (RL): The most efficient RL library, using 80% less VRAM for GRPO, FP8 etc.
* Supports full fine-tuning, RL, pretraining, 4-bit, 16-bit and, FP8 training.
* Observability: Monitor training live, track loss and GPU usage and customize graphs.
* Multi-GPU training is supported, with major improvements coming soon.

📥 INSTALL
Unsloth can be used in two ways: through Unsloth Studio, the web UI, or through Unsloth Core, the code-based version. Each has different requirements.

UNSLOTH STUDIO (WEB UI)
Unsloth Studio (Beta) works on Windows, Linux, WSL and macOS.
* CPU: Supported for Chat and Data Recipes currently
* NVIDIA: Training works on RTX 30/40/50, Blackwell, DGX Spark, Station and more
* macOS: Currently supports chat and Data Recipes. MLX training is coming very soon
* AMD: Chat + Data works. Train with Unsloth Core. Studio support is out soon.
* Coming soon: Training support for Apple MLX, AMD, and Intel.
* Multi-GPU: Available now, with a major upgrade on the way

MACOS, LINUX, WSL:
curl -fsSL https://unsloth.ai/install.sh | sh

WINDOWS:
irm https://unsloth.ai/install.ps1 | iex

LAUNCH
unsloth studio -H 0.0.0.0 -p 8888

UPDATE
To update, use the same install commands as above. Or run (does not work on Windows):
unsloth studio update

DOCKER
Use our Docker image unsloth/unsloth container. Run:
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 8000:8000 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth

UNSLOTH CORE (CODE-BASED)
LINUX, WSL:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto

WINDOWS:
winget install -e --id Python.Python.3.13
winget install --id=astral-sh.uv -e
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto

📒 FREE NOTEBOOKS
Train for free with our notebooks. You can use our new free Unsloth Studio notebook to run and train models for free in a web UI.

🦥 UNSLOTH NEWS
* Qwen3.6: Qwen3.6-35B-A3B can now be trained and run in Unsloth Studio.
* Gemma 4: Run and train Google’s new models directly in Unsloth.
* Introducing Unsloth Studio: our new web UI for running and training LLMs.
* Qwen3.5 - 0.8B, 2B, 4B, 9B, 27B, 35-A3B, 112B-A10B are now supported.
* Train MoE LLMs 12x faster with 35% less VRAM - DeepSeek, GLM, Qwen and gpt-oss.
* Embedding models: Unsloth now supports ~1.8-3.3x faster embedding fine-tuning.
* New 7x longer context RL vs. all other setups, via our new batching algorithms.
* New RoPE & MLP Triton Kernels & Padding Free + Packing: 3x faster training & 30% less VRAM.
* 500K Context: Training a 20B model with >500K context is now possible on an 80GB GPU.
* FP8 & Vision RL: You can now do FP8 & VLM GRPO on consumer GPUs.
* gpt-oss by OpenAI: Read our RL blog, Flex Attention blog and Guide.

📥 ADVANCED INSTALLATION
(Detailed instructions for Developer installs, Nightly builds, and Uninstallation are available in the source content).

CITATION
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}

LICENSE
Unsloth uses a dual-licensing model of Apache 2.0 and AGPL-3.0.
