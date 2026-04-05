"""
Fine-tune OpenVLA-7B on PushT Image dataset with the CORRECT objective (Action Prediction).

Previous attempts incorrectly used Causal LM loss on the text instruction.
This script uses discretized action tokens as labels.
"""
import argparse
import time
import json
import datetime
import torch
import torch.nn as nn
import gc
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torchvision.transforms as T
from PIL import Image

# PushT Dataset Stats (calculated from training set)
# We normalize to [-1, 1] for OpenVLA
ACTION_MIN = np.array([12.0, 25.0])
ACTION_MAX = np.array([511.0, 511.0])

IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lerobot/pusht_image")
    parser.add_argument("--steps", type=int, default=200) # Quick result as requested
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="finetune_real_corrected_results.json")
    args = parser.parse_args()

    device = "cuda"
    instruction = "push the T-shaped block to the target position"

    print(f"\n🔬 OpenVLA-7B Fine-Tuning - CORRECTED OBJECTIVE")
    print(f"   Dataset: {args.dataset}")
    print(f"   Steps: {args.steps}, Batch: {args.batch}, LR: {args.lr}\n")

    # 1. Load Model (4-bit QLoRA)
    print("PHASE 1: Loading OpenVLA-7B...")
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
    
    # PEFT Setup
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup discretizer
    # OpenVLA uses 256 bins in [-1, 1]
    n_bins = 256
    bin_centers = np.linspace(-1, 1, n_bins)
    
    # Correct action tokens mapping for OpenVLA
    # action_token = vocab_size - (discretized_val + 1)
    # We'll use the model's actual vocab_size from config
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of

    def discretize_and_tokenize(action):
        # action is [2] in [12, 511]
        # Normalize to [-1, 1]
        norm_action = (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) * 2 - 1
        norm_action = np.clip(norm_action, -1, 1)
        
        # Discretize
        # Find nearest bin center
        discretized = np.argmin(np.abs(norm_action[:, None] - bin_centers), axis=1) # [2]
        
        # OpenVLA expects 7D actions. We'll pad with 0 (which is the 128th bin)
        full_discretized = np.ones(7, dtype=int) * 128
        full_discretized[:2] = discretized
        
        # Map to tokens
        action_tokens = vocab_size - (full_discretized + 1)
        return action_tokens.tolist()

    # 3. Load Dataset (streaming)
    ds = load_dataset(args.dataset, split="train", streaming=True)
    data_iter = iter(ds)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # 4. Training Loop
    print(f"PHASE 2: Training ({args.steps} steps)")
    losses = []
    step_times = []

    model.train()
    for step in range(args.steps):
        t_start = time.time()
        
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(ds)
            sample = next(data_iter)
            
        # Process Image
        img = sample['observation.image'].convert("RGB")
        px = IMAGE_TRANSFORM(img).unsqueeze(0).to(device, dtype=torch.float32)
        px = torch.cat([px, px], dim=1) # Fused backbone
        
        # Process Action
        action_tokens = discretize_and_tokenize(np.array(sample['action']))
        
        # Tokenize Prompt
        # Standard OpenVLA prompt: "In: {instruction}\nOut:"
        prompt = f"In: {instruction}\nOut:"
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        prompt_ids = tokenized_prompt.input_ids[0] # [seq_len]
        
        # Construct multimodal input
        # Final sequence: [PromptTokens] [ActionTokens] [EOS]
        input_ids = torch.cat([
            prompt_ids,
            torch.tensor(action_tokens, dtype=torch.long),
            torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
        ]).unsqueeze(0).to(device)
        
        # Labels: ignore everything except the action tokens and EOS
        labels = input_ids.clone()
        labels[0, :len(prompt_ids)] = -100 # Mask prompt tokens
        # We also need to mask the padding/unknown if any, but here it's precise.
        
        # Forward
        outputs = model(pixel_values=px, input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        dt = time.time() - t_start
        losses.append(loss.item())
        step_times.append(dt)
        
        if (step+1) % 10 == 0 or step == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"  Step {step+1:4d} | Loss: {avg_loss:.4f} | Time: {dt*1000:.0f}ms | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 5. Save Results
    print("\nTraining Complete!")
    results = {
        "dataset": args.dataset,
        "steps": args.steps,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "avg_time_ms": np.mean(step_times) * 1000,
        "losses": losses
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
