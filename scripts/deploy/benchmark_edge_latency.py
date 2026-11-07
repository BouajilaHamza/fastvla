#!/usr/bin/env python3
"""
Edge Latency & Sustained Control Hz Benchmark for FastVLA.

Measures rolling-horizon action chunking throughput, step latency distribution,
and ODE solver performance across CPU, CUDA, and Edge compute devices.
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image

from fastvla.inference import FastVLAPolicyRunner, ActionChunkBuffer, EdgeTelemetryHUD


def benchmark_edge_runner(device: str = "cpu", iterations: int = 200, chunk_size: int = 16, action_dim: int = 7):
    print("=" * 70)
    print("FASTVLA EDGE LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Iterations:       {iterations}")
    print(f"Chunk Horizon:    {chunk_size} steps")
    print(f"Action Dimension: {action_dim}-DoF")
    print("-" * 70)

    runner = FastVLAPolicyRunner(
        device=device,
        chunk_size=chunk_size,
        action_dim=action_dim,
        ode_solver="midpoint",
        ode_steps=2,
    )

    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Warmup
    for _ in range(10):
        _ = runner.step(dummy_img, instruction="warmup")

    runner.telemetry = EdgeTelemetryHUD(window_size=iterations)
    start_time = time.perf_counter()

    for i in range(iterations):
        t0 = time.perf_counter()
        action = runner.step(dummy_img, instruction="التقط الوعاء الأسود")
        step_ms = (time.perf_counter() - t0) * 1000.0

    total_time = time.perf_counter() - start_time
    m = runner.telemetry.get_metrics()
    achieved_hz = iterations / total_time

    print(f"Total Time:             {total_time:.3f} s")
    print(f"Achieved Max Throughput:{achieved_hz:6.1f} Hz")
    print(f"Mean Step Latency:      {m['mean_step_latency_ms']:6.3f} ms")
    print(f"P95 Step Latency:       {m['p95_step_latency_ms']:6.3f} ms")
    print(f"Mean Chunk Generation:  {m['mean_chunk_inference_ms']:6.3f} ms")
    print(f"Control Margin (vs 20Hz): {achieved_hz / 20.0:.1f}x real-time")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    benchmark_edge_runner(device=args.device, iterations=args.iterations)
