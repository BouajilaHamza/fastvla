"""
Empirical Apples-to-Apples Side-by-Side Benchmark on NVIDIA Tesla T4 (16GB):
Standard 4-Bit SOTA (256 uncompressed visual tokens) vs FastVLA Turbo (64 compressed tokens via SpatialPatchMerger).

Trained on identical Arabic Push-T batches from hamzabouajila/ar-pusht-image.
"""

import os
import time
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.43.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.12.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "datasets>=2.20.0",
        "pillow>=10.0.0",
        "pyarrow>=15.0.0",
        "numpy>=1.24.0",
        "triton>=2.2.0",
        "huggingface_hub>=0.26.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("fastvla")
)

app = modal.App("fastvla-fair-pusht-comparison", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]


@app.function(gpu="T4", timeout=600, secrets=secrets)
def run_empirical_side_by_side(num_steps: int = 30, batch_size: int = 2):
    import io
    import torch
    import torch.nn as nn
    import timm
    import torchvision.transforms as T
    from PIL import Image
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb
    from fastvla.adapters.token_compressor import SpatialPatchMerger
    from fastvla.adapters.action_head import FlowMatchingActionHead
    import numpy as np

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 Running Side-by-Side Benchmark on {gpu_name} (16GB VRAM)...")

    # 1. Download parquet and prepare identical batches
    local_path = hf_hub_download(
        repo_id="hamzabouajila/ar-pusht-image",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(local_path)
    instructions = table["instruction"].to_pylist()
    actions = table["action"].to_pylist()

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batches = []
    chunk_size = 16
    for i in range(num_steps):
        imgs, trajs, instrs = [], [], []
        for b in range(batch_size):
            anchor = (i * batch_size + b) * 5 % (len(table) - chunk_size - 1)
            raw_img_dict = table["observation.image"][anchor].as_py()
            img = Image.open(io.BytesIO(raw_img_dict["bytes"])).convert("RGB")
            imgs.append(transform(img))
            
            chunk = []
            for s in range(chunk_size):
                act = actions[anchor + s]
                chunk.append([float(act[0]) / 256.0 - 1.0, float(act[1]) / 256.0 - 1.0])
            trajs.append(torch.tensor(chunk, dtype=torch.float32).flatten())
            instrs.append(instructions[anchor])
            
        tok = tokenizer([f"المهمة: {x}" for x in instrs], max_length=48, padding="max_length", truncation=True, return_tensors="pt")
        batches.append({
            "images": torch.stack(imgs),
            "trajectories": torch.stack(trajs),
            "input_ids": tok["input_ids"],
        })

    # Shared Vision Tower
    vision_encoder = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=True, num_classes=0).to(device=device, dtype=torch.float16).eval()

    vlm_dim = 2048
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    # -------------------------------------------------------------
    # RUN 1: Standard 4-Bit SOTA (Uncompressed 256 Visual Tokens)
    # -------------------------------------------------------------
    print("\n[Run 1/2] Benchmarking Standard 4-Bit SOTA (Uncompressed 256 tokens)...")
    proj_uncompressed = nn.Linear(1152, vlm_dim).to(device=device, dtype=torch.float16)
    llm_1 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    llm_1 = get_peft_model(llm_1, lora_config)
    head_1 = FlowMatchingActionHead(input_dim=vlm_dim, action_dim=32, hidden_dim=512).to(device=device, dtype=torch.float16)
    
    opt_1 = bnb.optim.PagedAdamW8bit([
        {"params": [p for p in llm_1.parameters() if p.requires_grad], "lr": 2e-4},
        {"params": proj_uncompressed.parameters(), "lr": 1e-4},
        {"params": head_1.parameters(), "lr": 3e-4},
    ])

    torch.cuda.reset_peak_memory_stats()
    times_1 = []
    # Warmup
    for step, b in enumerate(batches):
        t0 = time.perf_counter()
        opt_1.zero_grad()
        with torch.no_grad():
            raw_vis = vision_encoder.forward_features(b["images"].to(device=device, dtype=torch.float16))
        vis_emb = proj_uncompressed(raw_vis) # [B, 256, vlm_dim]
        text_emb = llm_1.get_input_embeddings()(b["input_ids"].to(device=device)).to(device=device, dtype=torch.float16)
        fused = torch.cat([vis_emb, text_emb], dim=1) # [B, 256 + 48 = 304, vlm_dim]
        out = llm_1(inputs_embeds=fused, output_hidden_states=True)
        loss = head_1.compute_loss(out.hidden_states[-1][:, -1, :], b["trajectories"].to(device=device, dtype=torch.float16))
        loss.backward()
        opt_1.step()
        torch.cuda.synchronize()
        if step >= 5: # exclude warmup
            times_1.append((time.perf_counter() - t0) * 1000.0)

    vram_1_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    step_ms_1 = float(np.mean(times_1))
    print(f"  Standard 4-Bit SOTA (256 tokens): {step_ms_1:.1f} ms/step | Peak VRAM: {vram_1_gb:.2f} GB")

    del llm_1, proj_uncompressed, head_1, opt_1
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # RUN 2: FastVLA Turbo (Compressed 64 Visual Tokens via SpatialPatchMerger)
    # -------------------------------------------------------------
    print("\n[Run 2/2] Benchmarking FastVLA Turbo (Compressed 64 tokens via 2x2 SpatialPatchMerger)...")
    compressor = SpatialPatchMerger(in_dim=1152, out_dim=vlm_dim, merge_factor=2).to(device=device, dtype=torch.float16)
    llm_2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    llm_2 = get_peft_model(llm_2, lora_config)
    head_2 = FlowMatchingActionHead(input_dim=vlm_dim, action_dim=32, hidden_dim=512).to(device=device, dtype=torch.float16)
    
    opt_2 = bnb.optim.PagedAdamW8bit([
        {"params": [p for p in llm_2.parameters() if p.requires_grad], "lr": 2e-4},
        {"params": compressor.parameters(), "lr": 1e-4},
        {"params": head_2.parameters(), "lr": 3e-4},
    ])

    torch.cuda.reset_peak_memory_stats()
    times_2 = []
    # Warmup
    for step, b in enumerate(batches):
        t0 = time.perf_counter()
        opt_2.zero_grad()
        with torch.no_grad():
            raw_vis = vision_encoder.forward_features(b["images"].to(device=device, dtype=torch.float16))
        vis_emb = compressor(raw_vis) # [B, 64, vlm_dim]
        text_emb = llm_2.get_input_embeddings()(b["input_ids"].to(device=device)).to(device=device, dtype=torch.float16)
        fused = torch.cat([vis_emb, text_emb], dim=1) # [B, 64 + 48 = 112, vlm_dim]
        out = llm_2(inputs_embeds=fused, output_hidden_states=True)
        loss = head_2.compute_loss(out.hidden_states[-1][:, -1, :], b["trajectories"].to(device=device, dtype=torch.float16))
        loss.backward()
        opt_2.step()
        torch.cuda.synchronize()
        if step >= 5: # exclude warmup
            times_2.append((time.perf_counter() - t0) * 1000.0)

    vram_2_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    step_ms_2 = float(np.mean(times_2))
    print(f"  FastVLA Turbo (64 tokens): {step_ms_2:.1f} ms/step | Peak VRAM: {vram_2_gb:.2f} GB")

    speedup = step_ms_1 / step_ms_2
    vram_saved_gb = vram_1_gb - vram_2_gb
    vram_reduction_pct = (vram_1_gb - vram_2_gb) / vram_1_gb * 100.0

    return {
        "sota_step_ms": step_ms_1,
        "sota_vram_gb": vram_1_gb,
        "fastvla_step_ms": step_ms_2,
        "fastvla_vram_gb": vram_2_gb,
        "speedup": speedup,
        "vram_saved_gb": vram_saved_gb,
        "vram_reduction_pct": vram_reduction_pct,
    }


@app.local_entrypoint()
def main():
    print("🚀 Triggering Empirical Side-by-Side 4-Bit Benchmark on Modal T4...")
    res = run_empirical_side_by_side.remote(num_steps=30, batch_size=2)
    print("\n" + "=" * 80)
    print("🏆 EMPIRICAL 100% APPLES-TO-APPLES VERDICT (NVIDIA Tesla T4):")
    print(f"   Standard 4-Bit SOTA (256 tokens): {res['sota_step_ms']:.1f} ms/step | Peak VRAM: {res['sota_vram_gb']:.2f} GB")
    print(f"   FastVLA Turbo (64 tokens):       {res['fastvla_step_ms']:.1f} ms/step | Peak VRAM: {res['fastvla_vram_gb']:.2f} GB")
    print(f"   👉 Measured Speedup:              {res['speedup']:.2f}x FASTER")
    print(f"   👉 Measured VRAM Reduction:       -{res['vram_reduction_pct']:.1f}% (-{res['vram_saved_gb']:.2f} GB)")
    print("=" * 80)
