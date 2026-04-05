"""
Fine-tune OpenVLA-7B on PushT Image dataset (real robotics data).
PushT: 48K samples, 96x96 RGB images, 2D push actions.
Small enough for quick validation (~1000 steps = ~25 min, ~$0.08).

Usage:
    python finetune_real.py --dataset lerobot/pusht_image --steps 1000
"""

import argparse
import time
import json
import datetime
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torchvision.transforms as T


# Dataset-specific instruction templates
DATASET_INFO = {
    "lerobot/pusht_image": {
        "instruction": "push the T-shaped block to the target position",
        "image_key": "observation.image",
    },
    "lerobot/libero_10_image": {
        "instruction": "complete the manipulation task",
        "image_key": "observation.images.image",
    },
}


IMAGE_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lerobot/pusht_image")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="finetune_real_results.json")
    args = parser.parse_args()

    device = "cuda"
    ds_info = DATASET_INFO.get(
        args.dataset,
        {
            "instruction": "complete the robotics task",
            "image_key": "observation.image",
        },
    )

    print("\n🔬 OpenVLA-7B Fine-Tuning on Real Dataset")
    print(f"   Dataset: {args.dataset}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Steps: {args.steps}, Batch: {args.batch}, LR: {args.lr}\n")

    # ── Load Dataset ──────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: Loading Dataset (streaming mode)")
    print("=" * 70)

    data_start = time.time()
    ds = load_dataset(args.dataset, streaming=True)
    # Count via streaming
    sample = next(iter(ds["train"]))
    print(f"  Image key: {ds_info['image_key']}")
    print(f"  Image size: {sample[ds_info['image_key']].size}")
    print(f"  Action dim: {len(sample['action'])}")
    print(f'  Instruction: "{ds_info["instruction"]}"')
    print(f"  Loaded in {time.time() - data_start:.1f}s\n")

    # ── Load Model ──────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 2: Loading OpenVLA-7B (4-bit QLoRA)")
    print("=" * 70)

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
    print(f"  ✅ Loaded in {load_time:.1f}s")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✅ LoRA: {trainable:,} trainable ({trainable / total * 100:.2f}%)\n")

    # Load tokenizer separately (OpenVLA doesn't expose .processor via AutoModel)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── Pre-Training Validation ─────────────────────────────────────
    print("=" * 70)
    print("PHASE 3: Pre-Training Validation")
    print("=" * 70)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(5):
            # Get samples from middle of dataset (skip first 10K)
            iterator = iter(ds["train"])
            for _ in range(20000 + i * 1000):
                next(iterator)
            sample = next(iterator)

            img = sample[ds_info["image_key"]].convert("RGB")
            px = (
                IMAGE_TRANSFORM(img).unsqueeze(0).to(device, dtype=torch.float32)
            )  # [1,3,224,224]

            # Duplicate to 6 channels for fused backbone
            px = torch.cat([px, px], dim=1)  # [1,6,224,224]

            # Tokenize instruction
            tokenized = tokenizer(
                [ds_info["instruction"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            labels = input_ids.clone()  # For loss computation

            out = model(
                pixel_values=px,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            val_losses.append(out.loss.item())

    pre_val = sum(val_losses) / len(val_losses)
    print(f"  ✅ Pre-train val loss: {pre_val:.4f}\n")
    model.train()

    # ── Training ────────────────────────────────────────────────────
    print("=" * 70)
    print(f"PHASE 4: Training ({args.steps} steps)")
    print("=" * 70)

    losses = []
    step_times = []
    torch.cuda.reset_peak_memory_stats()

    # Create data iterator
    data_iter = iter(ds["train"])

    for step in range(args.steps):
        step_start = time.time()

        # Get next sample
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(ds["train"])
            sample = next(data_iter)

        # Process image
        img = sample[ds_info["image_key"]].convert("RGB")
        px = IMAGE_TRANSFORM(img).unsqueeze(0).to(device, dtype=torch.float32)
        px = torch.cat([px, px], dim=1)  # [1,6,224,224]

        # Tokenize instruction
        tokenized = tokenizer(
            [ds_info["instruction"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        labels = input_ids.clone()

        # Forward
        outputs = model(
            pixel_values=px,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item())

        if (step + 1) % 50 == 0 or step == 0:
            recent = losses[-50:] if step >= 49 else losses[: step + 1]
            avg_loss = sum(recent) / len(recent)
            avg_time = sum(step_times[-50:]) / len(step_times[-50:])
            mem = torch.cuda.memory_allocated() / 1e9
            print(
                f"  Step {step + 1:5d}/{args.steps} | "
                f"loss={avg_loss:.4f} | "
                f"time={avg_time * 1000:.0f}ms | "
                f"steps/s={1 / avg_time:.2f} | "
                f"VRAM={mem:.2f}GB"
            )

    # Post-training validation
    model.eval()
    post_val_losses = []
    with torch.no_grad():
        for i in range(5):
            iterator = iter(ds["train"])
            for _ in range(30000 + i * 1000):
                next(iterator)
            sample = next(iterator)

            img = sample[ds_info["image_key"]].convert("RGB")
            px = IMAGE_TRANSFORM(img).unsqueeze(0).to(device, dtype=torch.float32)
            px = torch.cat([px, px], dim=1)

            tokenized = model.processor.tokenizer(
                [ds_info["instruction"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            out = model(
                pixel_values=px, input_ids=input_ids, attention_mask=attention_mask
            )
            post_val_losses.append(out.loss.item())

    post_val = sum(post_val_losses) / len(post_val_losses)
    model.train()

    total_time = sum(step_times)
    avg_step = total_time / len(step_times)
    steps_per_sec = len(step_times) / total_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Dataset:           {args.dataset}")
    print(f"  Total time:        {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Avg step:          {avg_step * 1000:.0f}ms")
    print(f"  Steps/sec:         {steps_per_sec:.2f}")
    print(f"  Peak VRAM:         {peak_vram:.2f} GB")
    print(f"  Pre-val loss:      {pre_val:.4f}")
    print(f"  Post-val loss:     {post_val:.4f}")
    print(f"  Improvement:       {pre_val - post_val:+.4f}")
    print(f"  Loss trend:        {losses[0]:.4f} → {losses[-1]:.4f}")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": args.dataset,
        "steps": args.steps,
        "batch_size": args.batch,
        "lr": args.lr,
        "total_time_sec": total_time,
        "avg_step_ms": avg_step * 1000,
        "steps_per_sec": steps_per_sec,
        "peak_vram_gb": peak_vram,
        "pre_val_loss": pre_val,
        "post_val_loss": post_val,
        "val_improvement": pre_val - post_val,
        "train_loss_start": losses[0],
        "train_loss_end": losses[-1],
        "losses": losses,
        "step_times_ms": [t * 1000 for t in step_times],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
