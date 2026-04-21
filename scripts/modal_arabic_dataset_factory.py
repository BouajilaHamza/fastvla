import os
import modal
import json
from pathlib import Path
from dotenv import load_dotenv

# Load local .env secrets (HF_API_KEY, WANDB_API_KEY)
load_dotenv()
hf_key = os.environ.get("HF_API_KEY")
vla_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_key})]

# ── 1. Define Environment ──────────────────────────────────────────────────
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0", "transformers>=4.40.0", "accelerate>=0.28.0",
        "datasets>=2.16.0", "torchvision>=0.17.0", "timm>=0.9.12", 
        "python-dotenv", "tqdm", "sacremoses", "sentencepiece"
    )
)

app = modal.App("fastvla-arabic-dataset-factory")

# ── 2. Data Translation & Push Logic ──────────────────────────────────────
@app.function(
    image=image,
    gpu=None, # No GPU needed for permission check
    timeout=600,
    secrets=vla_secrets
)
def verify_hf_permissions(target_user: str):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login, HfApi

    # Login to HF
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN (HF_API_KEY) not found in secrets.")
    login(token=hf_token)
    
    # Create a tiny dummy dataset
    data = {"test": ["permission check", "arabic", "robotics"]}
    ds = Dataset.from_dict(data)
    dd = DatasetDict({"train": ds})
    
    repo_id = f"{target_user}/hf-permission-test"
    print(f"📦 Attempting to push test dataset to: {repo_id}...")
    dd.push_to_hub(repo_id, private=False)
    print(f"✅ Permission check successful! Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
    return True

@app.function(
    image=image,
    gpu="L4", # Translation is faster on GPU
    timeout=7200,
    secrets=vla_secrets
)
def translate_and_push(dataset_name: str, target_repo: str):
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    from huggingface_hub import login

    # Login to HF
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN (HF_API_KEY) not found in secrets.")
    login(token=hf_token)

    print(f"🌍 Starting Arabic Translation for {dataset_name}...")
    ds = load_dataset(dataset_name)
    
    # 1. Known Instructions / Fallbacks (since LeRobot Parquets often lack the column)
    known_tasks = {
        "lerobot/pusht_image": {0: "push the block to the goal"},
        "lerobot/libero_10_image": {
            0: "Turn on the stove and put the moka pot on it.",
            1: "Close the bottom drawer of the cabinet and put the black bowl on top of it.",
            2: "Put the yellow peach in the basket and put the basket on the shelf.",
            3: "Put the white bowl in the moka pot and put the moka pot on the stove.",
            4: "Put the wine bottle on the wine rack and put the wine rack on the shelf.",
            5: "Put the alphabet soup in the basket and move the basket to the near side of the table.",
            6: "Put the butter in the basket and move the basket to the near side of the table.",
            7: "Put the white mug on the plate and put the plate on the shelf.",
            8: "Put the glass cup on the plate and put the plate on the shelf.",
            9: "Put the white mug on the book and put the book on the shelf."
        }
    }

    # Extract Unique Instructions
    instructions = set()
    dataset_fallbacks = known_tasks.get(dataset_name, {})
    
    # Try to find task strings in dataset metadata if not in known_tasks
    if not dataset_fallbacks and hasattr(ds["train"], "info") and ds["train"].info.description:
        # Some datasets put task descriptions in the info
        print("💡 Attempting to extract tasks from dataset info...")
    
    # If we have fallbacks, use them
    if dataset_fallbacks:
        instructions.update(dataset_fallbacks.values())
    else:
        # Final fallback: scan for instruction column (original logic)
        for split in ds.keys():
            for item in ds[split]:
                inst = item.get("instruction") or item.get("language_instruction")
                if inst: instructions.add(inst)
    
    if not instructions:
        instructions.add("perform the task")
    
    unique_list = list(instructions)
    print(f"🔍 Found {len(unique_list)} unique instructions to translate.")

    # 2. Setup Translation Model (NLLB-200)
    model_id = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn", tgt_lang="arb_Arab")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda")

    # 3. Batch Translate
    mapping = {}
    batch_size = 32
    for i in range(0, len(unique_list), batch_size):
        batch = unique_list[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("arb_Arab"), 
            max_length=128
        )
        results = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        for eng, arb in zip(batch, results):
            mapping[eng] = arb
            print(f"  {eng} -> {arb}")

    # Create task_index -> Arabic mapping if applicable
    index_to_arabic = {}
    if dataset_fallbacks:
        for idx, eng in dataset_fallbacks.items():
            index_to_arabic[idx] = mapping.get(eng, "إدفع الكتلة إلى الهدف")

    # 4. Map Dataset to Arabic
    def map_to_arabic(example):
        # 1. Use task_index if available (Preferred for LeRobot)
        if "task_index" in example and example["task_index"] in index_to_arabic:
            example["instruction"] = index_to_arabic[example["task_index"]]
        
        # 2. Fallback to string replacement if instruction exists
        elif "instruction" in example and example["instruction"] in mapping:
            example["instruction"] = mapping[example["instruction"]]
        
        # 3. Default for PushT if all else fails
        elif dataset_name == "lerobot/pusht_image":
            example["instruction"] = mapping.get("push the block to the goal", "إدفع الكتلة إلى الهدف")
            
        return example

    print("🛠️ Adding 'instruction' column with Arabic text...")
    arabic_ds = ds.map(map_to_arabic)

    # 5. Push to Hub
    print(f"🚀 Pushing Arabic dataset to: {target_repo}...")
    arabic_ds.push_to_hub(target_repo, private=False)
    print(f"✅ Success! Dataset available at: https://huggingface.co/datasets/{target_repo}")

@app.function(image=image, secrets=vla_secrets)
def inspect_keys(dataset_name: str):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split='train', streaming=True)
    sample = next(iter(ds))
    print(f"📊 Keys for {dataset_name}: {list(sample.keys())}")
    # Print the first instruction found
    inst = sample.get("instruction") or sample.get("language_instruction")
    print(f"📝 Sample Instruction: {inst}")

@app.function(image=image, secrets=vla_secrets)
def compare_datasets(original_name: str, translated_name: str):
    from datasets import load_dataset
    print(f"\n🔍 Comparing {original_name} vs {translated_name}")
    
    orig = load_dataset(original_name, split='train', streaming=True)
    trans = load_dataset(translated_name, split='train', streaming=True)
    
    # Check first 5 tasks/samples
    orig_iter = iter(orig)
    trans_iter = iter(trans)
    
    samples_to_check = 100 # Check a larger window to see task shifts
    tasks_found = {}
    
    for i in range(samples_to_check):
        try:
            o_item = next(orig_iter)
            t_item = next(trans_iter)
            
            t_idx = o_item.get("task_index", 0)
            t_inst = t_item.get("instruction")
            
            if t_idx not in tasks_found:
                tasks_found[t_idx] = t_inst
                print(f"📍 Task Index {t_idx}: {t_inst}")
        except StopIteration:
            break
            
    print(f"✅ Comparison complete for {translated_name}")

# ── Orchestrator ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    # Target User Namespace
    hf_username = "hamzabouajila"

    # 0. Comparison/Verification
    print("🧪 Verifying data integrity...")
    compare_datasets.remote("lerobot/pusht_image", f"{hf_username}/ar-pusht-image")
    compare_datasets.remote("lerobot/libero_10_image", f"{hf_username}/ar-libero-10-image")
