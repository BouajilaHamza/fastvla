"""
FastVLA Smoke-Test + Benchmark — Modal L4 & T4
===============================================
Runs a short real-use-case experiment on both GPU SKUs:
  - OpenVLA-style dummy model (avoids gated HF download in CI)
  - PushT / LeRobot dataset
  - Measures: inference latency, training step time, peak VRAM, control Hz
  - Logs everything to W&B
  - Tears down the Modal app on exit

Usage (from repo root, keys as env vars):
    WANDB_API_KEY=... HF_API_KEY=... \\
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \\
    modal run examples/modal_smoke_benchmark.py
"""

import os
import json
import time
import modal

# ── Image ──────────────────────────────────────────────────────────────────
_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "transformers>=4.43.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.12.0",
        "datasets>=2.20.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "numpy>=1.24.0,<2.0.0",
        "pillow>=10.0.0",
        "triton>=2.3.0",
        "huggingface_hub>=0.26.0",
        "wandb>=0.18.0",
        "psutil",
    )
    .pip_install("unsloth @ git+https://github.com/unslothai/unsloth.git")
    .pip_install("fastvla @ git+https://github.com/BouajilaHamza/fastvla.git@main")
)

app = modal.App("fastvla-smoke-benchmark", image=_image)

# ── Secrets ─────────────────────────────────────────────────────────────────
_secrets = [
    modal.Secret.from_dict({
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HF_TOKEN":      os.environ.get("HF_API_KEY", ""),
        "HF_API_KEY":    os.environ.get("HF_API_KEY", ""),
    })
]


# ── Core benchmark function (GPU-agnostic) ───────────────────────────────────
def _run_benchmark(gpu_label: str) -> dict:
    import torch
    import wandb
    from fastvla import FastVLAModel, FastVLATrainer, get_dataset
    from fastvla.benchmarking import PerformanceProfiler

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_total  = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available() else 0
    )

    print(f"\n{'='*60}")
    print(f"  FastVLA Smoke Benchmark — {gpu_label}")
    print(f"  Device  : {device_name}")
    print(f"  VRAM    : {vram_total:.1f} GB")
    print(f"{'='*60}\n")

    wandb.init(
        project="fastvla-smoke-benchmark",
        name=f"smoke-{gpu_label}-{int(time.time())}",
        config={"gpu": gpu_label, "device": device_name, "vram_gb": vram_total},
        reinit=True,
    )

    results = {
        "gpu_label": gpu_label,
        "device_name": device_name,
        "vram_total_gb": vram_total,
    }

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("► Loading OpenVLA-style dummy model ...")
    t0 = time.time()
    model = FastVLAModel.from_pretrained(
        dummy=True,
        action_dim=7,
        chunk_size=1,
        gradient_checkpointing=True,
        freeze_vision_encoder=True,
        pooling_strategy="masked_mean",
    )
    results["model_load_s"] = round(time.time() - t0, 2)
    print(f"  Loaded in {results['model_load_s']}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ── 2. Inference latency ─────────────────────────────────────────────────
    print("► Inference latency (50 iters, batch=1, seq=32) ...")
    B, T = 1, 32
    batch_inf = {
        "pixel_values":   torch.randn(B, 1, 3, 224, 224, device=device),
        "input_ids":      torch.randint(0, 1000, (B, T), device=device),
        "attention_mask": torch.ones(B, T, device=device),
    }
    WARMUP, ITERS = 5, 50
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(**batch_inf)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            model(**batch_inf)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    inf_avg_ms = sum(latencies) / len(latencies)
    results.update({
        "inference_avg_ms": round(inf_avg_ms, 2),
        "inference_min_ms": round(min(latencies), 2),
        "inference_p95_ms": round(sorted(latencies)[int(0.95 * ITERS)], 2),
        "control_hz":       round(1000.0 / inf_avg_ms, 2),
        "vram_after_inf_gb": round(
            torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0, 3
        ),
    })
    print(f"  Avg {inf_avg_ms:.1f} ms  |  {results['control_hz']} Hz  |  VRAM {results['vram_after_inf_gb']} GB")

    # ── 3. Training throughput ───────────────────────────────────────────────
    print("► Training step benchmark (25 steps, batch=2) ...")
    B_tr = 2
    batch_tr = {
        "pixel_values":   torch.randn(B_tr, 1, 3, 224, 224, device=device),
        "input_ids":      torch.randint(0, 1000, (B_tr, T), device=device),
        "attention_mask": torch.ones(B_tr, T, device=device),
        "labels":         torch.randn(B_tr, 7, device=device),
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model.train()
    step_times = []
    for step in range(25):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        _, loss = model(**batch_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= 5:
            step_times.append(elapsed)
        if step % 5 == 0:
            print(f"    step {step:3d}  loss={loss.item():.4f}  {elapsed*1000:.0f} ms")

    train_avg_ms  = sum(step_times) / len(step_times) * 1000
    vram_peak     = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    vram_reserved = torch.cuda.max_memory_reserved()  / 1024**3 if device == "cuda" else 0
    results.update({
        "train_step_avg_ms":  round(train_avg_ms, 2),
        "train_its":          round(1000.0 / train_avg_ms, 2),
        "peak_vram_alloc_gb": round(vram_peak, 3),
        "peak_vram_res_gb":   round(vram_reserved, 3),
    })
    print(f"  {train_avg_ms:.0f} ms/step  |  {results['train_its']} it/s  |  peak VRAM {vram_peak:.2f} GB")

    # ── 4. Dataset smoke-test ────────────────────────────────────────────────
    print("► Dataset smoke-test (lerobot/pusht_image) ...")
    try:
        ds = get_dataset("lerobot/pusht_image", chunk_size=1)
        sample = ds[0]
        ds_ok = "images" in sample or any("image" in k for k in sample)
        results["dataset_pusht_ok"] = ds_ok
        print(f"  OK: {ds_ok}  keys: {list(sample.keys())[:6]}")
    except Exception as exc:
        results["dataset_pusht_ok"] = False
        results["dataset_error"]    = str(exc)
        print(f"  Skipped: {exc}")

    # ── 5. Log & finish ──────────────────────────────────────────────────────
    wandb.log(results)
    wandb.finish()

    print(f"\n  DONE {gpu_label}")
    for k, v in results.items():
        print(f"    {k:<30} {v}")
    return results


# ── Modal functions — one per GPU SKU ────────────────────────────────────────

@app.function(gpu="L4", timeout=1800, secrets=_secrets)
def benchmark_l4() -> dict:
    return _run_benchmark("L4")


@app.function(gpu="T4", timeout=1800, secrets=_secrets)
def benchmark_t4() -> dict:
    return _run_benchmark("T4")


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    modal.enable_output()
    print("\n🚀  FastVLA smoke benchmark starting on Modal L4 + T4 ...\n")

    futures = {
        "L4": benchmark_l4.spawn(),
        "T4": benchmark_t4.spawn(),
    }

    results = {}
    for label, fut in futures.items():
        print(f"⏳  Waiting for {label} ...")
        try:
            results[label] = fut.get()
        except Exception as exc:
            print(f"❌  {label} failed: {exc}")
            results[label] = {"error": str(exc)}

    print("\n" + "=" * 60)
    print(json.dumps(results, indent=2))

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📄  Results → benchmark_results.json")
    print("    Next: python examples/update_readme_benchmarks.py")

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  Metric                   │  L4          │  T4          │")
    print("├─────────────────────────────────────────────────────────┤")
    for m in ["inference_avg_ms","control_hz","train_step_avg_ms","train_its","peak_vram_alloc_gb"]:
        l4v = results.get("L4", {}).get(m, "n/a")
        t4v = results.get("T4", {}).get(m, "n/a")
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"│  {m:<25}  │  {fmt(l4v):<12} │  {fmt(t4v):<12} │")
    print("└─────────────────────────────────────────────────────────┘")
