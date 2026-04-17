import os
import modal
from dotenv import load_dotenv

# Load local .env to get the HF token for the Modal Secret
load_dotenv()

# Define the production environment matching Kaggle 2x T4
# Acting as a stranger: Installing everything from remote sources
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0", "transformers>=4.38.0", "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0", "peft>=0.9.0", "datasets>=2.16.0",
        "torchvision>=0.17.0", "timm>=0.9.12", "numpy<2.0.0",
        "python-dotenv", "setuptools", "wheel"
    )
    .pip_install("git+https://github.com/unslothai/unsloth.git")
    # Stranger approach: Install FastVLA directly from GitHub
    .pip_install("git+https://github.com/BouajilaHamza/FastVLA.git")
)

app = modal.App("fastvla-pusht-finetuning", image=image)

# Get credentials from your local .env
hf_token = os.environ.get("HF_API_KEY")

@app.function(
    gpu="T4:2", 
    timeout=7200, 
    secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token})]
)
def finetune():
    import torch
    import os
    from fastvla import FastVLAModel, FastVLATrainer
    from fastvla.data.datasets import get_dataset
    
    print("🚀 Initializing Finetuning on 2x T4 GPUs (Modal)...")
    print(f"CUDA Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        # Load real OpenVLA-7B (Tests Surgical Extraction + 4-bit Logic)
        print("📦 Loading OpenVLA-7B (4-bit + LoRA)...")
        model = FastVLAModel.from_pretrained(
            "openvla-7b",
            load_in_4bit=True,
            use_peft=True,
            action_dim=2, # PushT
            hf_token=os.environ.get("HF_TOKEN")
        )
        print("✅ Model loaded successfully!")
        
        # Load real PushT dataset
        print("📥 Loading PushT dataset...")
        dataset = get_dataset("lerobot/pusht_image")
        
        print("🏋️ Initializing Trainer (Accelerate-ready)...")
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            batch_size=4,
            gradient_accumulation_steps=2,
            use_mixed_precision=True,
            use_8bit_optimizer=True,
            max_steps=10, # Verification run
            output_dir="/root/checkpoints"
        )
        
        print("🔥 Starting distributed training loop...")
        trainer.train()
        print("✨ Finetuning verification complete!")
        
    except Exception as e:
        print(f"❌ Finetuning failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    if not hf_token:
        print("❌ Error: HF_API_KEY not found in .env file.")
    else:
        with app.run():
            finetune.remote()
