"""
Example benchmarking script for FastVLA.
Demonstrates how to benchmark model performance.
"""

import torch
from fastvla import (
    FastVLAModel,
    FastVLAConfig,
    PerformanceProfiler,
    compare_models,
    print_benchmark_results,
)


def main():
    """Main benchmarking function."""
    # Create test batch
    batch_size = 2
    num_cameras = 3
    seq_length = 32

    batch = {
        "pixel_values": torch.randn(batch_size, num_cameras, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
        "labels": torch.randn(batch_size, 7),
    }

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Create models to compare
    config_4bit = FastVLAConfig(
        vision_encoder_name="google/vit-base-patch16-224",
        llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=True,
        use_peft=True,
    )

    config_fp16 = FastVLAConfig(
        vision_encoder_name="google/vit-base-patch16-224",
        llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=False,
        use_peft=True,
    )

    print("Loading models...")
    model_4bit = FastVLAModel.from_pretrained(config=config_4bit).to(device)
    model_fp16 = FastVLAModel.from_pretrained(config=config_fp16).to(device)

    models = {
        "4-bit Quantized": model_4bit,
        "FP16": model_fp16,
    }

    # Benchmark models
    print("\nBenchmarking models...")
    results = compare_models(models, batch, num_iterations=50)

    # Print results
    print_benchmark_results(results)

    # Memory profiling
    print("\nMemory Profiling:")
    profiler = PerformanceProfiler(device=device)

    for name, model in models.items():
        print(f"\n{name}:")
        with profiler.profile(f"forward_{name}"):
            _ = model(**batch)

        gpu_memory = profiler.get_gpu_memory()
        if gpu_memory:
            for gpu_name, mem_info in gpu_memory.items():
                print(
                    f"  {gpu_name}: {mem_info['memory_used_gb']:.2f} GB / {mem_info['memory_total_gb']:.2f} GB"
                )

    # Print summary
    summary = profiler.get_summary()
    print(f"\nTotal profiling time: {summary['total_time_sec']:.2f} seconds")
    print(f"Max memory usage: {summary['max_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
