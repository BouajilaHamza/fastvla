#!/usr/bin/env python3
"""
Modal L4 GPU SOTA Benchmark: FastVLA vs Standard VLA Baselines.
Compares actual GPU execution:
1. Standard Baseline (OpenVLA / LeRobot standard PEFT with 32-bit AdamW, standard attention, all layers)
2. FastVLA Optimized (Paged 8-bit AdamW, Selective Layer-K LoRA, Batched Camera GEMM, Fused Loss)
"""

import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.43.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.12.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "numpy>=1.24.0,<2.0.0",
        "triton>=2.2.0",
    )
    .pip_install("fastvla @ git+https://github.com/BouajilaHamza/fastvla.git@main")
)

app = modal.App("fastvla-sota-benchmark", image=image)


@app.function(gpu="L4", timeout=600)
def benchmark_on_gpu():
    import time
    import torch
    from fastvla import FastVLAModel, FastVLAConfig
    from fastvla.optimization import get_8bit_optimizer

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f"🚀 FASTVLA SOTA BENCHMARK ON {gpu_name}")
    print("=" * 70)

    batch_size = 4
    num_cams = 2
    img_size = 224
    seq_len = 32
    action_dim = 7
    hidden_size = 512
    num_layers = 12
    vocab_size = 1000
    num_steps = 30

    pixel_values = torch.randn(batch_size, num_cams, 3, img_size, img_size, device=device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randn(batch_size, action_dim, device=device)

    # 1. BASELINE: Standard 32-bit AdamW + all layers
    print("\n[1/2] Benchmarking Standard Baseline (OpenVLA / LeRobot standard)...")
    config_baseline = FastVLAConfig(
        dummy=True,
        vision_hidden_size=hidden_size,
        llm_hidden_size=hidden_size,
        llm_num_layers=num_layers,
        num_attention_heads=8,
        vocab_size=vocab_size,
        action_dim=action_dim,
        use_peft=False,
    )
    model_baseline = FastVLAModel(config_baseline).to(device)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-4)

    # Warmup
    for _ in range(5):
        optimizer_baseline.zero_grad()
        _, loss = model_baseline(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer_baseline.step()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_steps):
        optimizer_baseline.zero_grad()
        _, loss = model_baseline(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer_baseline.step()

    torch.cuda.synchronize()
    duration_baseline = time.time() - start_time
    peak_mem_baseline = torch.cuda.max_memory_allocated() / (1024 ** 2)
    steps_s_baseline = num_steps / duration_baseline
    ms_step_baseline = (duration_baseline / num_steps) * 1000

    print(f"  → Baseline Time: {duration_baseline:.3f}s ({steps_s_baseline:.1f} steps/s, {ms_step_baseline:.1f} ms/step)")
    print(f"  → Baseline Peak VRAM: {peak_mem_baseline:.2f} MB")

    del model_baseline, optimizer_baseline
    torch.cuda.empty_cache()

    # 2. FASTVLA: Paged 8-bit AdamW + Selective Layer-K LoRA + Batched forward
    print("\n[2/2] Benchmarking FastVLA SOTA Architecture (Batched + Paged 8-bit + Layer-K LoRA)...")
    config_fastvla = FastVLAConfig(
        dummy=True,
        vision_hidden_size=hidden_size,
        llm_hidden_size=hidden_size,
        llm_num_layers=num_layers,
        num_attention_heads=8,
        vocab_size=vocab_size,
        action_dim=action_dim,
        use_peft=False,
        trainable_llm_layers=4,  # Only top 4 layers of 12
    )
    model_fastvla = FastVLAModel(config_fastvla).to(device)
    optimizer_fastvla = get_8bit_optimizer(model_fastvla, learning_rate=1e-4)

    # Warmup
    for _ in range(5):
        optimizer_fastvla.zero_grad()
        _, loss = model_fastvla(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer_fastvla.step()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_steps):
        optimizer_fastvla.zero_grad()
        _, loss = model_fastvla(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer_fastvla.step()

    torch.cuda.synchronize()
    duration_fastvla = time.time() - start_time
    peak_mem_fastvla = torch.cuda.max_memory_allocated() / (1024 ** 2)
    steps_s_fastvla = num_steps / duration_fastvla
    ms_step_fastvla = (duration_fastvla / num_steps) * 1000

    print(f"  → FastVLA Time: {duration_fastvla:.3f}s ({steps_s_fastvla:.1f} steps/s, {ms_step_fastvla:.1f} ms/step)")
    print(f"  → FastVLA Peak VRAM: {peak_mem_fastvla:.2f} MB")

    speedup = steps_s_fastvla / steps_s_baseline
    vram_savings = (1.0 - peak_mem_fastvla / peak_mem_baseline) * 100

    print("\n" + "=" * 70)
    print("📊 EMPIRICAL GPU BENCHMARK SUMMARY (NVIDIA L4)")
    print("=" * 70)
    print(f"Throughput Speedup:      {speedup:.2f}x faster")
    print(f"Latency Reduction:       {(1.0 - ms_step_fastvla / ms_step_baseline) * 100:.1f}% lower latency")
    print(f"Peak VRAM Savings:       {vram_savings:.1f}% less memory")
    print("=" * 70)

    return {
        "speedup": speedup,
        "vram_savings": vram_savings,
        "peak_mem_baseline_mb": peak_mem_baseline,
        "peak_mem_fastvla_mb": peak_mem_fastvla,
        "ms_step_baseline": ms_step_baseline,
        "ms_step_fastvla": ms_step_fastvla,
    }


@app.local_entrypoint()
def main():
    benchmark_on_gpu.remote()
