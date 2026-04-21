import os
import modal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
hf_key = os.environ.get("HF_API_KEY")
if not hf_key:
    raise ValueError("HF_API_KEY not found in .env")

vla_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_key})]

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0", "transformers>=4.40.0", "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0", "peft>=0.9.0", "datasets>=2.16.0",
        "torchvision>=0.17.0", "timm>=0.9.12", "numpy<2.0.0",
        "python-dotenv"
    )
    .run_commands("pip install git+https://github.com/BouajilaHamza/fastvla.git")
)

app = modal.App("fastvla-pusht-upload")
volume = modal.Volume.from_name("fastvla-data")

@app.function(image=image, volumes={"/data": volume}, secrets=vla_secrets)
def upload(repo_id="hamzabouajila/fastvla-pusht-model"):
    from fastvla import FastVLAModel
    import os

    # Based on the user's previous runs, it seems checkpoints are here
    output_dir = "/data/checkpoints/arabic-vla-hero"
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print(f"❌ No checkpoints found in {output_dir}")
        return

    latest_cp = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    cp_path = os.path.join(output_dir, latest_cp)
    print(f"📦 Uploading latest checkpoint: {latest_cp} from {cp_path} to {repo_id}")

    model = FastVLAModel.from_pretrained(
        model_name_or_path=cp_path,
        load_in_4bit=True,
        hf_token=os.environ.get("HF_TOKEN")
    )
    model.push_to_hub(repo_id, token=os.environ.get("HF_TOKEN"))
    print(f"✨ Model published: https://huggingface.co/{repo_id}")

@app.local_entrypoint()
def main():
    upload.remote()
