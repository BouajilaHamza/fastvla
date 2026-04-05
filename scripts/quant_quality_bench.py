"""
Quantization Quality Benchmark for OpenVLA-7B.
Compares Q4-LoRA vs Q8-LoRA on speed AND quality (validation loss).

Usage:
    # Compare Q4 vs Q8
    python quant_quality_bench.py --quant all --steps 100

    # Just Q4
    python quant_quality_bench.py --quant q4 --steps 100

    # Custom settings
    python quant_quality_bench.py --quant all --steps 50 --lr 2e-4 --image_size 196
"""
import argparse
import time
import json
import datetime
import torch
import gc
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def generate_batch(batch_size, seq_len=64, image_size=224, device="cuda"):
    """Synthetic robotics batch."""
    pixel_values = torch.randn(batch_size, 6, image_size, image_size,
                                device=device, dtype=torch.float32)
    input_ids = torch.randint(100, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return pixel_values, input_ids, attention_mask, labels


def generate_fixed_val_set(num_samples=20, seq_len=64, image_size=224, device="cuda"):
    """Fixed validation set for consistent comparison across experiments."""
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    val_batches = []
    for _ in range(num_samples):
        px = torch.randn(1, 6, image_size, image_size, device=device, dtype=torch.float32,
                         generator=generator)
        iid = torch.randint(100, 32000, (1, seq_len), device=device, generator=generator)
        am = torch.ones_like(iid)
        labels = iid.clone()
        val_batches.append((px, iid, am, labels))
    return val_batches


def evaluate(model, val_batches):
    """Compute average validation loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for px, iid, am, labels in val_batches:
            out = model(pixel_values=px, input_ids=iid, attention_mask=am, labels=labels)
            losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


def load_model(quant_mode="q4", image_size=224):
    """Load OpenVLA with specified quantization."""
    clear_gpu()

    if quant_mode == "q4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_mode == "q8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    print(f"  Loading OpenVLA-7B ({quant_mode.upper()}-LoRA)...")
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    load_time = time.time() - t0

    vram_load = torch.cuda.memory_allocated() / 1e9
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Loaded in {load_time:.1f}s | VRAM: {vram_load:.2f} GB | Params: {total_params:,}")

    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✅ LoRA: {trainable:,} trainable ({trainable/total_params*100:.2f}%)")

    return model, vram_load, load_time, trainable, total_params


def run_experiment(quant_mode="q4", steps=100, lr=1e-4, batch_size=1,
                   seq_len=64, image_size=224, val_every=20, val_samples=20):
    """Run full fine-tuning experiment with validation checkpoints."""
    clear_gpu()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {quant_mode.upper()}-LoRA | Steps={steps} | LR={lr} | BS={batch_size}")
    print(f"{'='*70}\n")

    model, vram_load, load_time, trainable, total_params = load_model(quant_mode, image_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )

    # Fixed validation set
    val_batches = generate_fixed_val_set(val_samples, seq_len, image_size)

    # Pre-training validation
    print("  Pre-training validation...")
    pre_val = evaluate(model, val_batches)
    print(f"  ✅ Pre-train val loss: {pre_val:.4f}")

    # Training
    print(f"\n  Training ({steps} steps, val every {val_every})...\n")
    losses = []
    val_checkpoints = []
    step_times = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(steps):
        step_start = time.time()

        px, iid, am, labels = generate_batch(batch_size, seq_len, image_size)
        outputs = model(pixel_values=px, input_ids=iid, attention_mask=am, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item())

        # Validation checkpoint
        if (step + 1) % val_every == 0:
            val_loss = evaluate(model, val_batches)
            peak = torch.cuda.max_memory_allocated() / 1e9
            val_checkpoints.append({
                "step": step + 1,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "peak_vram_gb": peak,
                "step_time_ms": step_time * 1000,
            })
            recent_train = sum(losses[-val_every:]) / val_every
            print(f"    Step {step+1:4d}/{steps}: "
                  f"train={recent_train:.4f} | val={val_loss:.4f} | "
                  f"time={step_time*1000:.0f}ms | VRAM={peak:.2f}GB")

    # Post-training validation
    post_val = evaluate(model, val_batches)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    total_time = sum(step_times)
    avg_step = total_time / len(step_times)

    print(f"\n  {'='*50}")
    print(f"  {quant_mode.upper()}-LoRA Results")
    print(f"  {'='*50}")
    print(f"    Load VRAM:       {vram_load:.2f} GB")
    print(f"    Peak VRAM:       {peak_vram:.2f} GB")
    print(f"    Total time:      {total_time:.1f}s")
    print(f"    Avg step:        {avg_step*1000:.0f}ms")
    print(f"    Steps/sec:       {1/avg_step:.2f}")
    print(f"    Pre-val loss:    {pre_val:.4f}")
    print(f"    Post-val loss:   {post_val:.4f}")
    print(f"    Improvement:     {pre_val - post_val:+.4f}")

    return {
        "quant_mode": quant_mode,
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "image_size": image_size,
        "load_vram_gb": vram_load,
        "peak_vram_gb": peak_vram,
        "total_time_sec": total_time,
        "avg_step_ms": avg_step * 1000,
        "steps_per_sec": 1 / avg_step,
        "trainable_params": trainable,
        "total_params": total_params,
        "pre_val_loss": pre_val,
        "post_val_loss": post_val,
        "val_improvement": pre_val - post_val,
        "val_checkpoints": val_checkpoints,
        "train_losses": losses,
        "step_times_ms": [t * 1000 for t in step_times],
    }


def print_comparison(results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"QUANTIZATION COMPARISON")
    print(f"{'='*70}")

    print(f"\n┌──────────┬────────┬────────┬─────────┬──────────┬──────────┬──────────┐")
    print(f"│ Mode     │ VRAM   │ Step   │ Steps/s │ Pre-val  │ Post-val │ Δ-val    │")
    print(f"├──────────┼────────┼────────┼─────────┼──────────┼──────────┼──────────┤")

    for r in sorted(results, key=lambda x: {"q4": 0, "q8": 1}.get(x["quant_mode"], 2)):
        q = r["quant_mode"].upper()
        print(f"│ {q+'-LoRA':8s} │ {r['load_vram_gb']:5.2f} │ {r['avg_step_ms']:5.0f}ms │ "
              f"{r['steps_per_sec']:7.2f} │ {r['pre_val_loss']:8.4f} │ {r['post_val_loss']:8.4f} │ "
              f"{r['val_improvement']:+8.4f} │")

    print(f"└──────────┴────────┴────────┴─────────┴──────────┴──────────┴──────────┘")

    # Interpretation
    if len(results) == 2:
        q4 = next((r for r in results if r["quant_mode"] == "q4"), None)
        q8 = next((r for r in results if r["quant_mode"] == "q8"), None)
        if q4 and q8:
            speed_ratio = q4["steps_per_sec"] / q8["steps_per_sec"] if q8["steps_per_sec"] > 0 else 0
            val_diff = abs(q4["post_val_loss"] - q8["post_val_loss"])

            print(f"\n  Analysis:")
            print(f"    Speed:    Q4 is {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'} than Q8")
            print(f"    Quality:  Post-val diff = {val_diff:.4f}")

            if val_diff < 0.1:
                print(f"    ✅ Q4 and Q8 converge to similar quality → Q4 is sufficient")
            elif val_diff < 0.5:
                print(f"    ⚠️  Q4 slightly worse than Q8 → acceptable tradeoff for speed")
            else:
                print(f"    ❌ Q4 significantly worse than Q8 → use Q8 for quality-critical tasks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", type=str, default="q4",
                        choices=["q4", "q8", "all"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--val_samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="quant_quality_results.json")
    args = parser.parse_args()

    print(f"\n🔬 OpenVLA-7B Quantization Quality Benchmark")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Torch: {torch.__version__}\n")

    modes = ["q4", "q8"] if args.quant == "all" else [args.quant]
    results = []

    for mode in modes:
        result = run_experiment(
            quant_mode=mode,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch,
            seq_len=args.seq_len,
            image_size=args.image_size,
            val_every=args.val_every,
            val_samples=args.val_samples,
        )
        results.append(result)
        clear_gpu()

    if len(results) > 1:
        print_comparison(results)

    with open(args.output, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "config": {k: v for k, v in vars(args).items() if k not in ("output",)},
            "experiments": results,
        }, f, indent=2, default=str)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
