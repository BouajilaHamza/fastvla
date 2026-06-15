# FastVLA Training-Speed Deep Dive

All numbers below are reproducible by running:

```bash
modal run --detach examples/modal_smoke_benchmark.py        # dummy backbone, no HF download
modal run --detach examples/modal_production_benchmark.py   # real OpenVLA-7B + SmolVLA on L4 + T4
```

Raw artifacts: `benchmark_results.json`, `production_benchmark_results.json`, W&B projects `fastvla-smoke-benchmark` + `fastvla-production-benchmark`.

---

## 1. Measured throughput (this repo, Modal L4 + T4)

Single-image inference, B = 1, T = 32. 50 inference iterations / 25 training steps (smoke) and 30 / 10 (production).

### SmolVLA (SigLIP-so400m + SmolLM2-135M, ~270 M params)

| GPU | Inference avg | p95 | Control Hz | Train step avg | Train it/s | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | 20.13 ms | 20.75 ms | 49.68 Hz | 41.07 ms | **24.4 it/s** | 1.71 GB | 3.28 GB |
| T4 | 71.60 ms | 73.91 ms | 13.97 Hz | 144.32 ms | 6.93 it/s | 1.71 GB | 3.29 GB |

### OpenVLA-7B — SigLIP fallback path (see caveat below)

| GPU | Inference avg | p95 | Control Hz | Train step avg | Train it/s | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | 50.56 ms | 52.63 ms | 19.78 Hz | 61.84 ms | **16.2 it/s** | 4.36 GB | 5.22 GB |
| T4 | 213.83 ms | 215.77 ms | 4.68 Hz | 229.95 ms | 4.35 it/s | 5.45 GB | 5.64 GB |

> **Caveat — OpenVLA-7B vision tower:** `transformers.AutoModel` does not currently recognise `OpenVLAConfig` even with `trust_remote_code=True`, so the loader at `fastvla/adapters/vision.py:127-145` falls back to **SigLIP-so400m-384 only**, bypassing OpenVLA's fused DINOv2 + SigLIP backbone. The LLM trunk is still the full Llama-2-7B in 4-bit. Numbers above therefore approximate a "SigLIP + Llama-2-7B + FastVLA action head" deployment, not paper-spec OpenVLA. Restoring the fused backbone is tracked in [issue #2](https://github.com/BouajilaHamza/fastvla/issues/2); after that lands, expect a 30–50 % training slowdown vs the numbers above.

---

## 2. Reference points from the literature

| System | GPU | Throughput | Source |
|---|---|---:|---|
| **Vanilla bf16 OpenVLA-7B inference** | **L4** | **7.25 Hz** (138 ms) — train OOMs at 22 GB | measured: `baseline_benchmark_results.json`, `examples/modal_baseline_benchmark.py` |
| Vanilla 4-bit QLoRA OpenVLA-7B, post-bug baseline | L4 | ~3.0 it/s | [fastvla issue #1](https://github.com/BouajilaHamza/fastvla/issues/1) (4-bit load via plain HF currently crashes — OpenVLA's `PrismaticForConditionalGeneration.__init__` calls `.to()` on bnb submodules, which raises). |
| OpenVLA paper, LoRA fine-tune | 1×A100 80 GB | ~9–14 it/s (estimated, 100 k steps / 10–15 hrs) | [arXiv 2406.09246](https://arxiv.org/html/2406.09246v3) |
| OpenVLA paper, full fine-tune | 8×A100 80 GB | 5–15 hrs / task @ bs 64 | [arXiv 2406.09246](https://arxiv.org/html/2406.09246v3) |
| OpenVLA-OFT distributed | 8×A100/H100 80 GB | ~3–5 it/s per-GPU (50–150 k steps / 1–2 days) | [arXiv 2502.19645](https://arxiv.org/pdf/2502.19645) |
| SmolVLA pretrain | 1×A100 80 GB | ~4 hrs / 20 k steps (≈ 1.4 it/s) | [HF blog](https://huggingface.co/blog/smolvla) |
| Unsloth QLoRA, plain Llama-2-7B | 1×A100 | ~10–15 it/s | [Red Hat](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning) |
| Unsloth QLoRA, plain Llama-2-7B | 1×L4 | ~5–8 it/s | community |

---

## 3. Ratios

FastVLA L4 vs each reference:

| Comparison | Ratio | Honest read |
|---|---:|---|
| **Inference vs vanilla bf16 OpenVLA on L4** (measured) | **2.73×** faster (138 ms → 50.6 ms), **−69 % VRAM** (14.1 GB → 4.4 GB) | Inflated by the SigLIP-only fallback in FastVLA's loader; honest projection with full fused vision: ~1.5–2×. Still meaningful because the bf16 path **cannot train at all on L4** (OOMs at 22 GB), while FastVLA trains at 16.2 it/s. |
| **vs vanilla 4-bit QLoRA on L4** | **5.4×** (cited) | Inflated by the SigLIP-only fallback. With full fused vision: ~2–3×, which matches the 3–4× honest claim in [issue #1](https://github.com/BouajilaHamza/fastvla/issues/1). 4-bit baseline not yet locally measured — see source-table note. |
| **vs OpenVLA paper LoRA on A100** (~9 it/s low end) | **1.8×** | But on a GPU that costs ~1/30th as much per hour. Parity on cheaper hardware is the actual win. |
| **vs OFT distributed, per-GPU** | **3–5×** | Apples vs oranges: OFT uses much larger global batch + parallel-decoded chunks. |
| **vs Unsloth on plain L4 (no vision, no action head)** | **2–3×** | Spurious — FastVLA's "skip LM head" trick avoids materialising the `[B, T, ~128 k]` logits tensor that Unsloth's LLM path *has* to compute. VLA only needs the final hidden state for the action head. |

---

## 4. Where the speedup comes from (concretely)

All wired into `fastvla/model.py` + `fastvla/training.py` after PR #4:

1. **Skip LM head in forward** (`_encode_sequence`): calls the transformer trunk directly instead of `AutoModelForCausalLM`. Avoids `[B, T, ~128k]` logits tensor + the projection matmul every step.
2. **Gradient / activation checkpointing actually enabled** — the trainer used to declare it but never call `model.gradient_checkpointing_enable()`. ~30–40 % activation memory saving.
3. **PagedAdamW8bit** preferred over plain `AdamW8bit` — optimizer state pages to CPU under pressure, prevents OOM spikes on T4.
4. **DataLoader workers + pin\_memory + persistent\_workers** — default `num_workers=0` serialised image decode on the main thread and starved the GPU.
5. **Turing-aware attention** — `sdpa` on T4 (sm\_75, no FA2), `flash_attention_2` on Ada (sm\_89) / Ampere (sm\_80).
6. **Fused Triton action head** — single kernel for the 2-layer MLP forward; autograd backward caches `h1` instead of recomputing the forward chain.
7. **Masked-mean pooling** over the last hidden state — ignores padding tokens that the previous plain-mean was diluting the representation with.
8. **Discrete-action straight-through estimator** — eliminates the train/inference distribution shift (paper-style soft argmax at train vs hard argmax at deploy).

---

## 5. Known gaps (preventing higher scores)

1. **Vanilla 4-bit QLoRA baseline still cited.** `examples/modal_baseline_benchmark.py` now runs end-to-end and the bf16 row (7.25 Hz, 14.1 GB on L4) is locally verified. The 4-bit row remains cited because plain HF + bnb fails to load OpenVLA: `PrismaticForConditionalGeneration.__init__` calls `.to()` on bnb-quantised submodules. Workaround would need patching the upstream class. Tracked as the follow-up to this measurement.
2. **SigLIP fallback masks the real OpenVLA-7B picture.** Custom `auto_map` handling in `fastvla/adapters/vision.py` would restore the fused DINOv2 + SigLIP backbone. Expected impact: −30–50 % it/s on OpenVLA-7B, but apples-to-apples with the paper.
3. **B = 1 batches.** bnb 4-bit + paged AdamW shine most at larger batch sizes. Need a sweep over B ∈ {1, 4, 8, 16} per GPU.
4. **`torch.compile` is off by default.** `config.use_torch_compile=True` is wired but disabled. Typically +20–40 % on Ada.
5. **No `FlashAttention-3` path.** FA2 enabled on sm\_80+; FA3 on Hopper would close part of the H100 gap.
6. **Library bugs still surface during full runs.** Two were fixed in the session that produced these numbers (`fastvla/model.py:208` — vision-adapter `.config` AttributeError; `fastvla/kernels/fusion.py` — shared-memory exceed on L4/T4). Both are in main now; a CI job running the full smoke benchmark would catch regressions like these before release.

---

## 6. Sources

- OpenVLA (paper, Fig. 5): [arXiv 2406.09246](https://arxiv.org/html/2406.09246v3)
- OpenVLA-OFT: [openvla-oft.github.io](https://openvla-oft.github.io/), [arXiv 2502.19645](https://arxiv.org/pdf/2502.19645)
- SmolVLA: [HF blog](https://huggingface.co/blog/smolvla), [HF docs](https://huggingface.co/docs/lerobot/en/smolvla)
- Unsloth QLoRA speedup: [Red Hat developer post](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning)
- Modal L4 pricing: [modal.com/blog/nvidia-l4-price-article](https://modal.com/blog/nvidia-l4-price-article)
- FastVLA transparency: [issue #1](https://github.com/BouajilaHamza/fastvla/issues/1), [issue #2](https://github.com/BouajilaHamza/fastvla/issues/2), [PR #4](https://github.com/BouajilaHamza/fastvla/issues/4)
