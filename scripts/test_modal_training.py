import os
import modal
from dotenv import load_dotenv

# Load local .env for HF_API_KEY if needed locally
load_dotenv()

# Define the image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0",
        "peft>=0.9.0",
        "datasets>=2.16.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "numpy>=1.24.0,<2.2.0",
        "pillow>=10.0.0",
        "triton>=2.2.0",
        "huggingface_hub>=0.26.0",
        "python-dotenv",
        "setuptools",
        "wheel"
    )
    .pip_install("git+https://github.com/unslothai/unsloth.git")
    # Install local fastvla in editable mode
    .run_commands("pip install -e .")
)

app = modal.App("fastvla-training-test-distributed", image=image)

# Kaggle-style setup: 2x T4 GPUs
# We pass the HF_TOKEN from .env into the Modal Secret
hf_token = os.environ.get("HF_API_KEY")

@app.function(
    gpu="T4:2", 
    timeout=3600, 
    secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token})]
)
def run_training():
    import torch
    import os
    from fastvla import FastVLAModel, FastVLATrainer, get_dataset
    
    print("🚀 Starting FastVLA Distributed Training Test on Modal 2x T4...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    model_id = "openvla-7b" 
    
    try:
        print(f"📦 Loading {model_id} (Dummy Mode) in Distributed setup...")
        # We test the engine robustness here
        model = FastVLAModel.from_pretrained(
            dummy=True,
            action_dim=2, 
            vocab_size=50257
        )
        print("✅ Model loaded successfully!")
        
        print("📥 Loading dataset...")
        dataset = get_dataset("lerobot/pusht_image")
        
        print("🏋️ Initializing Trainer (Accelerate-ready)...")
        trainer = FastVLATrainer(
            model=model,
            dataset=dataset,
            batch_size=2, # Per-GPU batch size
            max_steps=10, 
            use_mixed_precision=True,
            use_8bit_optimizer=True,
            output_dir="/tmp/checkpoints"
        )
        
        print("🔥 Starting distributed training loop...")
        trainer.train()
        print("✨ Distributed training test completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    modal.enable_output()
    if not hf_token:
        print("❌ Error: HF_API_KEY not found in .env file.")
    else:
        with app.run():
            run_training.remote()
