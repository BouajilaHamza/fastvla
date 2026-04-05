"""
Fine-tune OpenVLA-7B with speed optimizations.
Options: FP16 vision encoder, smaller images, gradient accumulation.

Usage:
    # Default (optimized)
    python finetune_openvla.py --steps 100

    # With FP16 vision encoder (faster)
    python finetune_openvla.py --steps 100 --vision_dtype float16

    # With smaller images (much faster)
    python finetune_openvla.py --steps 100 --image_size 196

    # Full comparison (no optimizations)
    python finetune_openvla.py --steps 100 --vision_dtype float32 --image_size 224
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


def generate_robotics_batch(batch_size, seq_len=64, image_size=224,
                            num_channels=6, dtype=torch.float32, device="cuda"):
    """Generate synthetic robotics batch with configurable image size and dtype."""
    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size,
                                device=device, dtype=dtype)
    input_ids = torch.randint(100, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return pixel_values, input_ids, attention_mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size (224=default, 196=faster, 160=fastest)")
    parser.add_argument("--vision_dtype", type=str, default="float16",
                        choices=["float32", "float16"],
                        help="Vision encoder dtype (float16 is ~30%% faster)")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--quant", type=str, default="q4",
                        choices=["q4", "q8"],
                        help="Quantization mode")
    parser.add_argument("--output", type=str, default="finetune_results.json")
    args = parser.parse_args()

    vision_dtype = torch.float16 if args.vision_dtype == "float16" else torch.float32
    device = "cuda"

    print(f"\n🚀 OpenVLA-7B Fine-Tuning ({args.quant.upper()})")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Steps: {args.steps}, Batch: {args.batch}, LR: {args.lr}")
    print(f"   Image: {args.image_size}×{args.image_size} @ {args.vision_dtype}")
    print(f"   Torch: {torch.__version__}\n")

    # Load model
    print("Loading OpenVLA-7B...")
    t0 = time.time()

    if args.quant == "q4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "q8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    if args.quant != "fp16":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Convert vision encoder to target dtype
    if args.vision_dtype == "float16":
        print("  Converting vision encoder to FP16...")
        model.base_model.model.vision_backbone = model.base_model.model.vision_backbone.half()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Training
    print(f"\n{'='*70}")
    print(f"TRAINING ({args.steps} steps)")
    print(f"{'='*70}\n")

    losses = []
    step_times = []
    torch.cuda.reset_peak_memory_stats()

    # Profile first step
    profile_step = True

    for step in range(args.steps):
        step_start = time.time()

        px, iid, am, labels = generate_robotics_batch(
            args.batch, args.seq_len, args.image_size,
            dtype=vision_dtype, device=device
        )

        outputs = model(pixel_values=px, input_ids=iid, attention_mask=am, labels=labels)
        loss = outputs.loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item() * args.grad_accum)

        if (step + 1) % 10 == 0 or step == 0:
            recent = losses[-10:] if step >= 9 else losses[:step+1]
            avg_loss = sum(recent) / len(recent)
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  Step {step+1:4d}/{args.steps} | "
                  f"loss={avg_loss:.4f} | "
                  f"time={avg_time*1000:.0f}ms | "
                  f"steps/s={1/avg_time:.2f} | "
                  f"VRAM={mem:.2f}GB")

    total_time = sum(step_times)
    avg_step = total_time / len(step_times)
    steps_per_sec = len(step_times) / total_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Avg step:          {avg_step*1000:.0f}ms")
    print(f"  Steps/sec:         {steps_per_sec:.2f}")
    print(f"  Peak VRAM:         {peak_vram:.2f} GB")
    print(f"  Loss:              {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[0]-losses[-1]:.4f})")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {k: v for k, v in vars(args).items() if k != "output"},
        "device": torch.cuda.get_device_name(0),
        "load_time_sec": load_time,
        "total_time_sec": total_time,
        "avg_step_ms": avg_step * 1000,
        "steps_per_sec": steps_per_sec,
        "peak_vram_gb": peak_vram,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_delta": losses[0] - losses[-1],
        "trainable_params": trainable,
        "losses": losses,
        "step_times_ms": [t * 1000 for t in step_times],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
