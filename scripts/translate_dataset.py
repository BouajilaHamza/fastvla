#!/usr/bin/env python3
"""
Arabic Dataset Translation Script for FastVLA.
Batch-translates English instructions in LeRobot/HF datasets into Arabic.
Usage:
    python scripts/translate_dataset.py --dataset lerobot/pusht_image --output data/pusht_arabic.json
"""

import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import pipeline

def translate_instructions(dataset_name, output_path, model_id="facebook/nllb-200-distilled-600M", batch_size=16):
    """
    Loads a dataset, extracts unique instructions, translates them, and saves a mapping.
    """
    print(f"📥 Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name, split='train')
    
    # 1. Extract unique instructions to save API/GPU costs
    instructions = set()
    instruction_key = "instruction"
    
    # Handle common LeRobot variants
    sample = ds[0]
    if instruction_key not in sample:
        if "language_instruction" in sample:
            instruction_key = "language_instruction"
        else:
            print("⚠️ Could not find 'instruction' key. Defaulting to 'push the block to the goal'.")
            instructions.add("push the block to the goal")
    
    if instruction_key in sample:
        print("🔍 Extracting unique instructions...")
        for item in tqdm(ds, desc="Scanning"):
            inst = item.get(instruction_key)
            if inst:
                instructions.add(inst)
    
    unique_list = list(instructions)
    print(f"✅ Found {len(unique_list)} unique instructions.")

    # 2. Setup Translation Pipeline
    print(f"🚀 Initializing translation model: {model_id}...")
    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline(
        "translation", 
        model=model_id, 
        device=device,
        src_lang="eng_Latn",
        tgt_lang="arb_Arab"
    )

    # 3. Batch Translate
    print("🌍 Translating to Arabic...")
    translations = {}
    for i in range(0, len(unique_list), batch_size):
        batch = unique_list[i : i + batch_size]
        results = translator(batch, max_length=128)
        for eng, res in zip(batch, results):
            translations[eng] = res['translation_text']

    # 4. Save Mapping
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=4)
    
    print(f"✨ Success! Arabic translation mapping saved to: {output_path}")
    print("\nSample Translations:")
    for eng, arb in list(translations.items())[:5]:
        print(f"  {eng}  ->  {arb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate robotics dataset instructions.")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht_image", help="HF dataset name")
    parser.add_argument("--output", type=str, default="data/arabic_mapping.json", help="Path to save mapping")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M", help="Translation model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for translation")
    
    args = parser.parse_args()
    translate_instructions(args.dataset, args.output, args.model, args.batch_size)
