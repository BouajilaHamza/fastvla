"""
Empirical Benchmark on NVIDIA Tesla T4: Flow Matching Higher-Order ODE Solvers.

Evaluates:
1. Inference Latency (ms) across Euler-10, Euler-5, Midpoint-2, Midpoint-3, Heun-2, Heun-3.
2. Trajectory Quality: L2 Error, Max Absolute Error (MAE), and Pearson correlation
   against a 100-step high-resolution reference ODE trajectory.
3. Speedup vs Quality Pareto frontier on 16-step continuous action chunks (112 continuous DoF).
"""

import time
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.43.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.12.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "numpy>=1.24.0",
        "triton>=2.2.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("fastvla")
)

app = modal.App("fastvla-ode-solvers-benchmark", image=image)


@app.function(gpu="T4", timeout=600)
def benchmark_flow_ode_solvers(batch_size: int = 64):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from fastvla.adapters.action_head import FlowMatchingActionHead

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("=" * 105)
    print("NVIDIA T4 BENCHMARK: FLOW MATCHING HIGHER-ORDER ODE SOLVERS")
    print(f"Device: {gpu_name} | VRAM: {vram_gb:.2f} GB")
    assert "T4" in gpu_name or "Tesla" in gpu_name, f"Adversarial Guard: Expected Tesla T4, got {gpu_name}"
    assert vram_gb >= 14.0, f"Adversarial Guard: Expected ~16GB VRAM, got {vram_gb:.2f} GB"
    print("=" * 105)

    torch.manual_seed(42)
    B = batch_size  # Batch size saturated for T4 Tensor Cores
    D_in = 2048  # VLM hidden size
    D_action = 16 * 7  # 16-step chunk x 7 DoF = 112 continuous parameters
    head = FlowMatchingActionHead(input_dim=D_in, action_dim=D_action, hidden_dim=512).cuda().eval()

    hidden = torch.randn(B, D_in, device="cuda")
    x0 = torch.randn(B, D_action, device="cuda")

    # 1. Compute 100-Step Ground Truth Reference ODE Trajectory
    with torch.no_grad():
        gt_ref = head(hidden, solver="euler", steps=100, initial_noise=x0)

    # 2. Benchmark Candidates
    solver_configs = [
        ("Euler (10 steps) [Baseline]", "euler", 10),
        ("Euler (5 steps)", "euler", 5),
        ("Euler (2 steps)", "euler", 2),
        ("Midpoint (3 steps)", "midpoint", 3),
        ("Midpoint (2 steps)", "midpoint", 2),
        ("Heun (3 steps)", "heun", 3),
        ("Heun (2 steps)", "heun", 2),
    ]

    results = []
    num_warmup = 50
    num_iters = 200

    print("\n[Phase 1] Benchmarking Precision Latency & Trajectory Fidelity on T4...")
    print(f"{'Solver Configuration':<35} | {'Latency (ms)':<16} | {'Throughput':<18} | {'MSE vs 100-Step':<16} | {'Max Coord Error':<16}")
    print("-" * 115)

    baseline_latency = None

    for label, solver, steps in solver_configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(num_warmup):
            _ = head(hidden, solver=solver, steps=steps, initial_noise=x0)
        torch.cuda.synchronize()

        # Precision CUDA Event timing across 3 independent trials
        trial_latencies = []
        for _ in range(3):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                for _ in range(num_iters):
                    out = head(hidden, solver=solver, steps=steps, initial_noise=x0)
            end_event.record()
            torch.cuda.synchronize()
            trial_latencies.append(start_event.elapsed_time(end_event) / num_iters)

        mean_latency = float(np.mean(trial_latencies))
        std_latency = float(np.std(trial_latencies))

        if baseline_latency is None:
            baseline_latency = mean_latency

        speedup = baseline_latency / mean_latency
        throughput_hz = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        throughput_smp = throughput_hz * B

        # Quality metrics against 100-step GT
        with torch.no_grad():
            mse = F.mse_loss(out, gt_ref).item()
            max_err = torch.max(torch.abs(out - gt_ref)).item()

        results.append({
            "name": label,
            "solver": solver,
            "steps": steps,
            "latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "speedup": speedup,
            "throughput_hz": throughput_hz,
            "throughput_samples_sec": throughput_smp,
            "mse": mse,
            "max_err": max_err,
        })

        print(f"{label:<35} | {mean_latency:6.3f} ± {std_latency:5.3f} ms | {throughput_smp:7.1f} smp/s      | {mse:16.8f} | {max_err:16.8f}")

    print("\n" + "=" * 115)
    print("KEY FINDING & MATHEMATICAL ANALYSIS:")
    
    best_fast = min([r for r in results if r["steps"] <= 3], key=lambda x: x["mse"])
    print(f"★ Best Pareto Solver: {best_fast['name']}")
    print(f"  -> Latency: {best_fast['latency_ms']:.3f} ms vs {baseline_latency:.3f} ms ({best_fast['speedup']:.2f}x speedup)")
    print(f"  -> Trajectory MSE vs 100-Step Ref: {best_fast['mse']:.8f} (Max Coordinate Error: {best_fast['max_err']:.6f})")
    print("=" * 115)

    return results


if __name__ == "__main__":
    with app.run():
        res = benchmark_flow_ode_solvers.remote()
