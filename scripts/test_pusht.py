"""
Test the fine-tuned OpenVLA-7B on PushT.
This script loads a checkpoint, performs inference on a real sample, and compares predicted vs ground truth actions.
"""
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer
from datasets import load_dataset
import torchvision.transforms as T

# PushT normalization stats
ACTION_MIN = np.array([12.0, 25.0])
ACTION_MAX = np.array([511.0, 511.0])

IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vla-pusht/final")
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    instruction = "push the T-shaped block to the target position"

    print(f"\n🚀 Testing VLA Model: {args.checkpoint}")

    # 1. Load Model and Tokenizer
    # Note: For LoRA, we load the base model then the adapter
    print("PHASE 1: Loading model and adapter...")
    
    # We still need 4-bit to fit on T4 if original was 4-bit
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 2. Get a sample from PushT
    print("PHASE 2: Getting test sample from PushT...")
    ds = load_dataset("lerobot/pusht_image", split="train", streaming=True)
    sample = next(iter(ds))
    
    img = sample['observation.image'].convert("RGB")
    gt_action = np.array(sample['action'])
    
    # 3. Prepare Input
    px = IMAGE_TRANSFORM(img).unsqueeze(0).to(device, dtype=torch.float16)
    px = torch.cat([px, px], dim=1) # Fused backbone
    
    prompt = f"In: {instruction}\nOut:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 4. Generate Actions
    print("PHASE 3: Generating actions...")
    with torch.no_grad():
        # OpenVLA action tokens are the end of the vocab
        # We generate 7 tokens for a full action vector
        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=px,
            max_new_tokens=7,
            do_sample=False
        )
        
    # Extract only the newly generated tokens
    action_token_ids = generated_ids[0, -7:].cpu().numpy()
    
    # 5. De-tokenize and De-normalize
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    
    # token_id = vocab_size - (discretized_val + 1)
    # => discretized_val = vocab_size - token_id - 1
    discretized_actions = vocab_size - action_token_ids - 1
    
    # Convert discretized (0-255) to normalized (-1 to 1)
    bin_centers = np.linspace(-1, 1, 256)
    norm_actions = bin_centers[discretized_actions]
    
    # De-normalize to original range
    # norm = (raw - min) / (max - min) * 2 - 1
    # => raw = (norm + 1) / 2 * (max - min) + min
    pred_actions_raw = (norm_actions + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
    
    print("-" * 30)
    print(f"Instruction: {instruction}")
    print("-" * 30)
    print(f"Ground Truth Action: {gt_action[:2]}")
    print(f"Predicted Action:    {pred_actions_raw[:2]}")
    print("-" * 30)

    # Calculate L2 error
    error = np.linalg.norm(gt_action[:2] - pred_actions_raw[:2])
    print(f"Action L2 Error: {error:.2f} pixels")
    print("-" * 30)

if __name__ == "__main__":
    main()
