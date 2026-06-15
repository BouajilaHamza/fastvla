# Accessibility Roadmap — Memory & Speed

Concrete checklist of every lever between "today's FastVLA" and "VLA fine-tuning is as accessible as Unsloth-LLM fine-tuning". Items grouped by the gap they close (memory floor, training speed, evaluation honesty, library polish).

Status legend: ☑ done · 🟡 partial / off-by-default · ☐ planned · ⏸ blocked

---

## A. Memory floor — fit bigger VLAs on smaller GPUs

| # | Item | Status | Expected gain | Notes / location |
|---|---|---|---|---|
| A1 | bitsandbytes 4-bit base weights | ☑ | −60–70 % vs fp16 | `fastvla/model.py` load path |
| A2 | PEFT LoRA on LLM trunk only | ☑ | −95 % trainable params | `fastvla/model.py` `_apply_peft` |
| A3 | PagedAdamW8bit (paged optimiser state to CPU) | ☑ | Prevents OOM spikes; saves ~25 % optimiser mem | `fastvla/optimization.py` |
| A4 | Gradient / activation checkpointing **wired into trainer** | ☑ | −30–40 % activation mem | `fastvla/training.py` (fixed in PR #4 — was declared but never called) |
| A5 | Freeze vision encoder by default | ☑ | Removes ~300 M params from optimiser state | `config.freeze_vision_encoder=True` |
| A6 | Skip LM head in forward (`_encode_sequence`) | ☑ | Kills `[B, T, ~128k]` logits tensor every step | `fastvla/model.py` |
| A7 | Masked-mean pooling (no padded-token dilution) | ☑ | Correctness, not memory | `fastvla/model.py` |
| A8 | Fused Triton action head with cached forward | ☑ | −1 backward chain recompute | `fastvla/kernels/action.py` |
| A9 | **Liger-style fused RMSNorm + RoPE + SwiGLU** | ☐ | Unsloth claims −20–30 % activation mem on LLM trunk | New `fastvla/kernels/liger_*.py` |
| A10 | **Fused cross-entropy for discrete action head** | 🟡 | Avoids materialising full bin distribution tensor | `fastvla/adapters/action_head.py` (CE in PR #4; not fused) |
| A11 | **Activation offload to CPU (selective layers)** | ☐ | Enables 7B + B=4 on T4 (15 GB) | New `fastvla.distributed.offload` |
| A12 | **Selective layer-N freeze (last-k attention layers only)** | ☐ | −50 % LoRA mem when fine-tuning a single skill | Config flag |
| A13 | **Cached vision features** (precompute SigLIP embeddings per dataset, store on disk) | ☐ | Removes vision encoder from train loop entirely | New `scripts/dataset/precache_vision.py` |
| A14 | **Shared vision trunk for multi-camera** (instead of per-cam adapter copies) | 🟡 | Cuts multi-cam mem by N | `fastvla/kernels/multicam.py` exists; not yet weight-shared |
| A15 | **QLoRA on vision encoder too** (not just LLM) | ☐ | Important once vision encoder is unfrozen for hard tasks | Config + adapter loader |
| A16 | **8-bit master weights for optimiser** (already PagedAdamW**8bit**, generalize) | ☑ | Done | `fastvla/optimization.py` |
| A17 | **FlashAttention-3 path for Hopper** | ☐ | Saves activation mem on H100 attention | `fastvla/model.py` `attn_implementation` |
| A18 | **Sequence packing for short instructions** (Arabic tokens shorter than English) | ☐ | Effective batch ↑ at zero mem cost | `fastvla/data/collator.py` |

**Target**: OpenVLA-7B fine-tune fits in **6 GB peak** (already ~5.5 GB on T4) → push to **4 GB peak** with A9 + A10 + A11 + A13.

---

## B. Training speed — more iterations per second per GPU dollar

| # | Item | Status | Expected gain | Notes / location |
|---|---|---|---|---|
| B1 | DataLoader `num_workers > 0`, `pin_memory`, `persistent_workers` | ☑ | Fixed in PR #4 — was starving GPU at `num_workers=0` | `fastvla/training.py` |
| B2 | Antialiased image resize in collator | ☑ | Correctness; small speed impact | `fastvla/data/collator.py` |
| B3 | Turing-aware attention (`sdpa` on T4, FA2 on Ada / Ampere) | ☑ | T4: 1.5–2× attention path | `fastvla/model.py` |
| B4 | Fused Triton cross-attention vision↔language | 🟡 | Falls back to PyTorch SDPA past D > 256 (T4/L4 shared-mem limit) | `fastvla/kernels/fusion.py` (fixed this session) |
| B5 | **`torch.compile` on by default** for single-GPU Ada / Hopper | ☑ | +20–40 % | `_auto_torch_compile()` in `fastvla/config.py` flips the default by checking `torch.cuda.get_device_capability() >= (8, 9)`. Test: `tests/test_auto_compile.py` |
| B6 | **CUDA graphs for inference path** | ☐ | −10–30 % inference latency at small batch | `fastvla/inference.py` (to-be-created) |
| B7 | **Parallel-decoded action chunks** (OFT recipe — chunk_size ≥ 8) | 🟡 | 5–10× control rate without extra compute per chunk | `config.chunk_size` exists; parallel decode loop not yet implemented |
| B8 | **FAST tokenizer for action sequences** (Pi0-FAST) | ☐ | 5× autoregressive inference | New `fastvla/adapters/action_tokenizer.py` |
| B9 | **Speculative decoding** for action tokens | ☐ | 2× action gen at no quality cost | Long-term |
| B10 | **vLLM-style paged KV cache for LLM trunk** | ☐ | Large batch inference | Long-term |
| B11 | **Async inference loop** (overlap compute + IO, like SmolVLA's 30 % async win) | ☐ | 30 % task-level latency | `fastvla/inference.py` |
| B12 | **NVIDIA cuDNN v9 conv + SDPA path for vision encoder** | ☑ | nvcr base image; check kernel selection | Image: `nvcr.io/nvidia/pytorch:24.01-py3` |
| B13 | **bf16 mixed precision** for non-quantised paths | ☑ | Default on Ada | `fastvla/model.py` dtype handling |
| B14 | **Gradient accumulation tuned per GPU** (smaller bs × more accum) | 🟡 | Trades mem for steady throughput | Manual today, should auto-pick |
| B15 | **DDP / FSDP clean module** (for 2× T4 / 4× L4 setups) | 🟡 | Linear scaling on cheap multi-GPU | Currently coupled to Modal; needs `fastvla.distributed` |
| B16 | **Tensor-parallel for vision encoder when N=2** | ☐ | Halves vision encoder step time on 2× L4 | Long-term |

**Target**: **30+ it/s on L4** for OpenVLA-7B with full fused backbone (currently 16.2 it/s with SigLIP fallback; honest projection with full backbone + B5 + B7 is ~25–35 it/s).

---

## C. Evaluation honesty — make every published number reproducible

| # | Item | Status | Notes |
|---|---|---|---|
| C1 | Modal smoke benchmark (no HF download) | ☑ | `examples/modal_smoke_benchmark.py` |
| C2 | Modal production benchmark (real HF weights, OpenVLA + SmolVLA) | ☑ | `examples/modal_production_benchmark.py` |
| C3 | **Vanilla QLoRA baseline benchmark in repo** | ☑ | `examples/modal_baseline_benchmark.py` runs; bf16 row measured (138 ms / 7.25 Hz / 14.1 GB on L4). 4-bit still cited — plain HF + bnb fails to load OpenVLA |
| C4 | **OpenVLA full-fused vision tower load fix** | ☑ | `fastvla/adapters/vision.py::OpenVLAFusedVisionAdapter.from_pretrained` now tries `AutoModelForVision2Seq` → dynamic class load → `AutoModel` → SigLIP fallback. Test: `tests/test_openvla_loader.py` |
| C5 | **LIBERO eval harness** | ☐ | Standard VLA benchmark — required to compare against OFT's 97.1 % |
| C6 | **PushT eval with success-rate logging** | 🟡 | Training scripts exist; eval not yet a CI artefact |
| C7 | **Batch-size sweep** (B ∈ {1, 4, 8, 16}) | ☐ | Needed to show paged AdamW + 4-bit gains at scale |
| C8 | **Real-robot demo (SO-100 / koch)** | ☐ | Closes the credibility gap with LeRobot / OpenVLA-OFT |
| C9 | **CI runs smoke benchmark on every PR** | ☐ | Would have caught `model.py:208` + `kernels/fusion.py` bugs before merge |
| C10 | **HF Hub model card with reproducible recipe** | 🟡 | One model exists (`hamzabouajila/fastvla-arabic-precision`); needs eval numbers in card |

---

## D. Library polish — meet users where they are

| # | Item | Status | Notes |
|---|---|---|---|
| D1 | `uv` + `pyproject.toml` install path | ☑ | `uv sync` works |
| D2 | PyPI release | ☐ | Currently `pip install git+…`; ship `pip install fastvla` |
| D3 | Quickstart notebook (Colab L4) | ☐ | Lower-friction "try it" path than Modal |
| D4 | mkdocs site with API reference | ☐ | Currently `docs/*.md` only |
| D5 | Public W&B reports linked from README | 🟡 | W&B projects exist; report views not yet published |
| D6 | Docstrings + type hints on all public API | 🟡 | Partial |
| D7 | LICENSE + CONTRIBUTING (done) + CODE_OF_CONDUCT | 🟡 | Code of conduct missing |
| D8 | Cookiecutter for "add a new VLA model" | ☐ | Lowers onboarding cost for contributors |
| D9 | Per-model preset configs in `fastvla.registry` | 🟡 | `openvla-7b`, `smolvla`, `pi0-base`, `olmovla` present; some IDs broken (`allenai/olmovla-7b-hf` 404s) |

---

## Sequencing — what to do next, in priority order

Sprint 1 ☑ (commits `cbb8af9` and follow-ups) closed C3 (partial — bf16 measured), C4 (loader fixed + test), B5 (`torch.compile` auto-default).

1. **B7 — implement parallel-decoded action chunks** (OFT recipe). Single biggest control-rate win and brings inference back into the discussion.
2. **A9 + A10 — Liger-style fused RMSNorm/RoPE/SwiGLU + fused CE**. Catches FastVLA up to Unsloth's memory floor on the LLM trunk.
3. **C5 — LIBERO eval harness**. The standard benchmark; required to publish task success rates.
4. **A13 — cached vision features** + **B11 — async inference**. Both make the "fine-tune in 30 min on an L4" demo dramatically faster.
5. **D2 + D3 + D4 — PyPI release, Colab notebook, mkdocs**. Library polish that 10×s adoption.
6. **C3 follow-up — measure the vanilla 4-bit QLoRA row**. Requires either patching upstream OpenVLA's `PrismaticForConditionalGeneration.__init__` to skip `.to()` on bnb submodules, or swapping the baseline to a different VLA whose vanilla 4-bit load path is intact.

---

## Sources

- Unsloth memory / speed lever inventory: [Red Hat developer post](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning)
- OpenVLA-OFT parallel decoding + action chunking: [openvla-oft.github.io](https://openvla-oft.github.io/) / [arXiv 2502.19645](https://arxiv.org/pdf/2502.19645)
- Pi0-FAST action tokenizer: [LeRobot v0.5.0 release notes](https://awesomeagents.ai/news/lerobot-v050-humanoid-open-source/)
- SmolVLA async inference 30 % gain: [HF blog](https://huggingface.co/blog/smolvla)
- Liger kernels (RMSNorm / RoPE / SwiGLU fused): [LinkedIn/liger-kernel](https://github.com/linkedin/Liger-Kernel)
