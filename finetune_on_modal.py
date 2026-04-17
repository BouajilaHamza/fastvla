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
    # Using env var to force cache invalidation for the latest fixes
    .env({"FORCE_REBUILD": "v7"})
    .run_commands("pip install git+https://github.com/BouajilaHamza/fastvla.git")
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
    
    print("🚀 Starting PRODUCTION Finetuning on 2x T4 GPUs (Modal)...")
    
    try:
        # 1. Load Model
        print("📦 Loading OpenVLA-7B (4-bit + LoRA)...")
        model = FastVLAModel.from_pretrained(
            "openvla-7b",
            load_in_4bit=True,
            use_peft=True,
            action_dim=2, # PushT
            hf_token=os.environ.get("HF_TOKEN")
        )
        print("✅ Model loaded successfully!")
        
        # 2. Load Dataset
        print("📥 Loading PushT dataset...")
        dataset = get_dataset("lerobot/pusht_image")
        
        # 3. Initialize Trainer
        print("🏋️ Initializing Trainer...")
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            batch_size=4,
            gradient_accumulation_steps=2,
            use_mixed_precision=True,
            use_8bit_optimizer=True,
            max_steps=500, # Full finetuning run
            output_dir="/root/checkpoints"
        )
        
        # 4. Train
        print("🔥 Starting Training...")
        trainer.train()
        print("✅ Training complete!")

        # 5. Push to Hub
        repo_id = "BouajilaHamza/FastVLA-PushT-OpenVLA"
        print(f"⬆️ Pushing model to Hugging Face Hub: {repo_id}...")
        
        # We unwrap and push
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        unwrapped_model.push_to_hub(
            repo_id=repo_id,
            token=os.environ.get("HF_TOKEN")
        )
        print("✨ PRODUCTION RUN COMPLETE! Model is live on HF Hub.")
        
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
