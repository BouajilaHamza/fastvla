import os
import modal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
hf_key = os.environ.get("HF_API_KEY")
vla_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_key})]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0", "transformers>=4.40.0", "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0", "peft>=0.9.0", "datasets>=2.16.0",
        "torchvision>=0.17.0", "timm>=0.9.12", "numpy<2.0.0",
        "python-dotenv", "tqdm"
    )
    .add_local_dir(Path(__file__).parent.parent, remote_path="/root/project", copy=True)
    .run_commands("pip install -e /root/project")
)

app = modal.App("fastvla-arabic-eval")
volume = modal.Volume.from_name("fastvla-data", create_if_missing=True)

@app.function(image=image, gpu="L4", timeout=3600, volumes={"/data": volume}, secrets=vla_secrets)
def evaluate():
    import torch
    from fastvla import FastVLAModel
    import os

    output_dir = "/data/checkpoints/arabic-vla-hero"
    volume.reload()
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print("❌ No checkpoints found in /data/checkpoints/arabic-vla-hero")
        return

    latest_cp = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    cp_path = os.path.join(output_dir, latest_cp)
    print(f"📊 Loading latest checkpoint for evaluation: {latest_cp}")

    model = FastVLAModel.from_pretrained(
        model_name_or_path=cp_path,
        load_in_4bit=True,
        action_dim=2,
        hf_token=os.environ.get("HF_TOKEN")
    )
    
    # Simple Inference Check (Arabic)
    arabic_command = "دفع الحجر إلى الهدف"
    print(f"🧪 Testing inference with: {arabic_command}")
    
    pixel_values = torch.randn(1, 1, 3, 384, 384).cuda()
    tokenizer = model.tokenizer
    input_ids = tokenizer(arabic_command, return_tensors="pt")["input_ids"].cuda()
    
    with torch.no_grad():
        action, _ = model(pixel_values=pixel_values, input_ids=input_ids)
    
    print(f"✅ Inference successful. Predicted Action Shape: {action.shape}")

@app.local_entrypoint()
def main():
    evaluate.remote()
