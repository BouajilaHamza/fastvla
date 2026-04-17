import os
import modal
import time
import torch
from dotenv import load_dotenv

load_dotenv()

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
    .env({"FORCE_REBUILD": "v8"})
    .run_commands("pip install git+https://github.com/BouajilaHamza/fastvla.git")
)

app = modal.App("fastvla-benchmark", image=image)

@app.function(
    gpu="T4:2", 
    timeout=7200, 
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_API_KEY")})]
)
def run_all_benchmarks():
    import torch
    from fastvla import FastVLAModel, FastVLATrainer
    from fastvla.data.datasets import get_dataset
    
    models = ["smolvla-7b", "openvla-7b"]
    
    for model_id in models:
        print(f"\n🚀 Benchmarking {model_id} on 2x T4 GPUs...")
        
        # 1. Load Model
        try:
            model = FastVLAModel.from_pretrained(
                model_id,
                load_in_4bit=True,
                use_peft=True,
                action_dim=2,
                hf_token=os.environ.get("HF_TOKEN")
            )
            
            # 2. Load Dataset
            dataset = get_dataset("lerobot/pusht_image")
            
            # 3. Trainer
            trainer = FastVLATrainer(
                model=model,
                dataset=dataset,
                batch_size=4,
                gradient_accumulation_steps=2,
                max_steps=50, 
                output_dir="/tmp/checkpoints"
            )
            
            # 4. Measure
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"✅ {model_id} Benchmark Complete: {duration:.2f}s for 50 steps (~{duration/50:.2f}s/step)")
        except Exception as e:
            print(f"❌ {model_id} benchmark failed: {e}")

if __name__ == "__main__":
    with app.run():
        run_all_benchmarks.remote()
