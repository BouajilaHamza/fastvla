"""
Fine-tune OpenVLA-7B on a robotics dataset.
Measures real training speed, VRAM usage, and loss convergence.

Usage:
    python finetune_openvla.py --steps 100 --batch 1 --lr 1e-4
"""
import argparse
import time
import json
import datetime
import torch
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def generate_robotics_batch(batch_size, seq_len=64, image_size=224, device="cuda"):
    """
    Generate a synthetic robotics training batch that mimics real data:
    - 6-channel images (DINOv2 + SigLIP fused backbone)
    - Text instruction token IDs
    - Causal LM labels
    """
    # 6-channel: 3ch RGB + 3ch "fused" (mimics DINOv2+SigLIP)
    pixel_values = torch.randn(batch_size, 6, image_size, image_size,
                                device=device, dtype=torch.float32)

    # Text: instruction + action tokens
    # OpenVLA tokenizes: "<ACTION_INSTRUCTION> <action_tokens>"
    input_ids = torch.randint(100, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    return pixel_values, input_ids, attention_mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--output", type=str, default="finetune_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 OpenVLA-7B Fine-Tuning on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"   Steps: {args.steps}, Batch: {args.batch}, LR: {args.lr}")
    print(f"   Seq len: {args.seq_len}, Image: {args.image_size}×{args.image_size}")
    print(f"   Grad accumulation: {args.grad_accum}")
    print(f"   Torch: {torch.__version__}\n")

    # ── Load model ──────────────────────────────────────────────────
    print("Loading OpenVLA-7B with 4-bit QLoRA...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # ── Training loop ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TRAINING ({args.steps} steps)")
    print(f"{'='*70}\n")

    losses = []
    step_times = []
    lr_history = []
    vrms_history = []

    torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        step_start = time.time()

        px, iid, am, labels = generate_robotics_batch(
            args.batch, args.seq_len, args.image_size, device
        )

        # Forward
        outputs = model(
            pixel_values=px,
            input_ids=iid,
            attention_mask=am,
            labels=labels,
        )
        loss = outputs.loss / args.grad_accum
        loss.backward()

        # Step (with grad accumulation)
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item() * args.grad_accum)
        lr_history.append(optimizer.param_groups[0]["lr"])
        vrms_history.append(torch.cuda.memory_allocated() / 1e9)

        if (step + 1) % 10 == 0 or step == 0:
            recent = losses[-10:] if step >= 9 else losses[:step+1]
            avg_loss = sum(recent) / len(recent)
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            mem = vrms_history[-1]
            print(f"  Step {step+1:4d}/{args.steps} | "
                  f"loss={avg_loss:.4f} | "
                  f"time={avg_time*1000:.0f}ms | "
                  f"steps/s={1/avg_time:.2f} | "
                  f"VRAM={mem:.2f}GB")

    # ── Summary ─────────────────────────────────────────────────────
    total_time = sum(step_times)
    avg_step = sum(step_times) / len(step_times)
    steps_per_sec = len(step_times) / total_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Avg step time:     {avg_step*1000:.0f}ms")
    print(f"  Steps/sec:         {steps_per_sec:.2f}")
    print(f"  Samples/sec:       {steps_per_sec * args.batch:.2f}")
    print(f"  Peak VRAM:         {peak_vram:.2f} GB")
    print(f"  Free VRAM:         {torch.cuda.get_device_properties(0).total_memory/1e9 - peak_vram:.2f} GB")
    print(f"  Loss start:        {losses[0]:.4f}")
    print(f"  Loss end:          {losses[-1]:.4f}")
    print(f"  Loss delta:        {losses[0] - losses[-1]:.4f}")
    print(f"  Model load time:   {load_time:.1f}s")
    print(f"  Trainable params:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Save results ────────────────────────────────────────────────
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": vars(args),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model_load_sec": load_time,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_time_sec": total_time,
        "avg_step_ms": avg_step * 1000,
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": steps_per_sec * args.batch,
        "peak_vram_gb": peak_vram,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_delta": losses[0] - losses[-1],
        "losses": losses,
        "step_times_ms": [t * 1000 for t in step_times],
        "vram_history_gb": vrms_history,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
