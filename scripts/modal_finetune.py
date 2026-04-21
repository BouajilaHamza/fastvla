import os
import modal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
hf_key = os.environ.get("HF_API_KEY")
wandb_key = os.environ.get("WANDB_API_KEY")
vla_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_key, "WANDB_API_KEY": wandb_key})]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0", "transformers>=4.40.0", "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0", "peft>=0.9.0", "datasets>=2.16.0",
        "torchvision>=0.17.0", "timm>=0.9.12", "numpy<2.0.0",
        "python-dotenv", "tqdm", "gymnasium", "opencv-python",
        "sacremoses", "sentencepiece"
    )
    .pip_install("git+https://github.com/unslothai/unsloth.git")
    .add_local_dir(Path(__file__).parent.parent, remote_path="/root/project", copy=True)
    .run_commands("pip install -e /root/project")
)

app = modal.App("fastvla-arabic-finetune")
volume = modal.Volume.from_name("fastvla-data", create_if_missing=True)

@app.function(image=image, gpu="L4", timeout=12000, volumes={"/data": volume}, secrets=vla_secrets)
def finetune(dataset_id="hamzabouajila/ar-pusht-image", max_steps=2000):
    from fastvla import FastVLAModel, FastVLATrainer
    import os

    output_dir = "/data/checkpoints/arabic-vla-hero"
    os.makedirs(output_dir, exist_ok=True)
    
    volume.reload()
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_step = max([int(d.split("-")[1]) for d in checkpoints])
        if latest_step >= max_steps:
            print(f"✅ Training already complete (step {latest_step} >= {max_steps}). Skipping.")
            return

    model = FastVLAModel.from_pretrained(
        "openvla-7b", load_in_4bit=True, use_peft=True, action_dim=2,
        hf_token=os.environ.get("HF_TOKEN"), gradient_checkpointing=True
    )

    trainer = FastVLATrainer(
        model=model, train_dataset=dataset_id, batch_size=12,
        gradient_accumulation_steps=2, max_steps=max_steps,
        output_dir=output_dir, save_steps=250, logging_steps=10
    )
    
    if checkpoints:
        latest_cp = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        print(f"🔄 Resuming from: {latest_cp}")
        trainer.load_checkpoint(os.path.join(output_dir, latest_cp))
    
    trainer.train()
    volume.commit()

@app.local_entrypoint()
def main():
    finetune.remote()
