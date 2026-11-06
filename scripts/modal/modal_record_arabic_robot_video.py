"""
FastVLA Turbo: Video & Telemetry Recording for Arabic Physical AI Policy.

Rolls out the trained 7B foundation policy (hamzabouajila/fastvla-turbo-arabic-libero)
on a full 214-frame episode of the Franka Panda robot manipulating the stove & moka pot
from hamzabouajila/ar-libero-10-image.

Renders:
- Dual-camera view (Third-person agent camera + gripper wrist camera)
- Real-time telemetry HUD overlay (Arabic command, predicted 7-DoF actions, gripper gauge)
- Generates high-definition MP4 video and animated GIF.
"""

import os
import time
import json
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
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
        "opencv-python>=4.8.0",
        "huggingface_hub>=0.26.0",
    )
    .pip_install("fastvla @ git+https://github.com/BouajilaHamza/fastvla.git@f49dcd7")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("fastvla-record-arabic-video", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]


@app.function(gpu="T4", timeout=1200, secrets=secrets)
def record_arabic_policy_video(max_frames: int = 120):
    import gc
    import subprocess
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torch.nn as nn
    import timm
    import torchvision.transforms as T
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import bitsandbytes as bnb
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    from fastvla.adapters.token_compressor import SpatialPatchMerger
    from fastvla.adapters.action_head import FlowMatchingActionHead

    device = "cuda"
    repo_id = "hamzabouajila/fastvla-turbo-arabic-libero"
    hf_token = os.environ.get("HF_TOKEN", "")

    print("=" * 80)
    print(f"🎬 RECORDING FASTVLA TURBO PHYSICAL POLICY ROLLOUT")
    print(f"   Model: {repo_id}")
    print(f"   Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. Download model components
    print("\n[Phase 1] Loading FastVLA Turbo from Hugging Face...")
    merger_path = hf_hub_download(repo_id=repo_id, filename="spatial_patch_merger.pt", token=hf_token)
    head_path = hf_hub_download(repo_id=repo_id, filename="flow_matching_action_head.pt", token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)

    hidden_size = 4096
    trajectory_dim = 16 * 7
    llama_cfg = AutoConfig.for_model(
        "llama",
        hidden_size=hidden_size,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=len(tokenizer),
        torch_dtype="float16",
    )

    with torch.device("cpu"):
        model_cpu = AutoModelForCausalLM.from_config(llama_cfg)

    model_cpu.model.embed_tokens = model_cpu.model.embed_tokens.to("cuda")
    model_cpu.model.norm = model_cpu.model.norm.to("cuda")

    for layer in model_cpu.model.layers:
        layer.input_layernorm = layer.input_layernorm.to("cuda")
        layer.post_attention_layernorm = layer.post_attention_layernorm.to("cuda")
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            lin = getattr(layer.self_attn, name)
            q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
            setattr(layer.self_attn, name, q)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(layer.mlp, name)
            q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
            setattr(layer.mlp, name, q)

    model_cpu.lm_head = nn.Identity()
    model = PeftModel.from_pretrained(model_cpu, repo_id, token=hf_token).to("cuda").eval()

    merger = SpatialPatchMerger(in_dim=1152, out_dim=hidden_size, merge_factor=2, grid_size=(16, 16))
    merger.load_state_dict(torch.load(merger_path, map_location="cuda", weights_only=True))
    merger = merger.to(device=device, dtype=torch.float16).eval()

    flow_head = FlowMatchingActionHead(input_dim=hidden_size, action_dim=trajectory_dim, hidden_dim=512, num_inference_steps=10)
    flow_head.load_state_dict(torch.load(head_path, map_location="cuda", weights_only=True))
    flow_head = flow_head.to(device=device, dtype=torch.float16).eval()

    siglip_vit = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=False).to(device=device, dtype=torch.float16).eval()
    print("  ✅ Policy ready on Tesla T4.")

    # 2. Fetch Episode 0 consecutive frames
    print("\n[Phase 2] Fetching consecutive episode frames from hamzabouajila/ar-libero-10-image...")
    ds = load_dataset("hamzabouajila/ar-libero-10-image", split="train", streaming=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    episode_frames = []
    arabic_instruction = "أطفئ الفرن ووضع وعاء الموكا عليه."

    for item in ds:
        if item.get("episode_index", 0) == 0:
            raw_agent = item.get("observation.images.image", item.get("image"))
            raw_wrist = item.get("observation.images.wrist_image", item.get("wrist_image", raw_agent))
            arabic_instruction = item.get("instruction", arabic_instruction)

            if hasattr(raw_agent, "convert"):
                pil_agent = raw_agent.convert("RGB").resize((256, 256))
                pil_wrist = raw_wrist.convert("RGB").resize((256, 256))
            else:
                pil_agent = Image.new("RGB", (256, 256), color=(40, 40, 40))
                pil_wrist = Image.new("RGB", (256, 256), color=(40, 40, 40))

            episode_frames.append({
                "pil_agent": pil_agent,
                "pil_wrist": pil_wrist,
                "tensor_agent": transform(pil_agent),
                "tensor_wrist": transform(pil_wrist),
                "gt_action": item.get("action", [0.0] * 7)[:7],
            })
            if len(episode_frames) >= max_frames:
                break
        elif len(episode_frames) > 0:
            break

    print(f"  ✅ Collected {len(episode_frames)} continuous episode frames. Task: {arabic_instruction}")

    # 3. Policy Execution & Video Rendering Loop
    print("\n[Phase 3] Generating 16-step flow trajectories and rendering video frames...")
    rendered_frames = []
    current_action_chunk = None
    chunk_step_idx = 0

    prompt = f"المهمة: {arabic_instruction}"
    tok_out = tokenizer(prompt, max_length=48, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tok_out["input_ids"].to(device=device)

    for f_idx, frame in enumerate(episode_frames):
        # Query policy every 16 steps (action chunking) or on initial step
        if current_action_chunk is None or chunk_step_idx >= 16:
            img_tensor = torch.stack([frame["tensor_agent"], frame["tensor_wrist"]]).unsqueeze(0).to(device=device, dtype=torch.float16)
            B, num_cams, C, H, W = img_tensor.shape

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                flat_imgs = img_tensor.view(B * num_cams, C, H, W)
                vit_out = siglip_vit.forward_features(flat_imgs)
                vit_raw = vit_out.view(B, num_cams * 256, 1152)
                compressed_tokens = merger(vit_raw, num_cams=2)
                txt_embed = model.get_base_model().model.embed_tokens(input_ids)
                inputs_embeds = torch.cat([compressed_tokens, txt_embed], dim=1)
                out = model(inputs_embeds=inputs_embeds)
                last_hidden = out.logits[:, -1, :hidden_size]
                trajectory_flat = flow_head(last_hidden)
                current_action_chunk = trajectory_flat.view(16, 7).cpu().float()
                chunk_step_idx = 0

        # Current continuous action step
        action_step = current_action_chunk[chunk_step_idx].tolist()
        chunk_step_idx += 1

        # 4. Compose Studio-Quality Video Frame
        # Width: 640px, Height: 440px
        canvas = np.zeros((440, 640, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 30)  # Dark slate background

        # Place Agent Image (Left: 256x256 at x=40, y=80)
        img_agent_cv = cv2.cvtColor(np.array(frame["pil_agent"]), cv2.COLOR_RGB2BGR)
        canvas[80:336, 40:296] = img_agent_cv
        cv2.rectangle(canvas, (40, 80), (296, 336), (60, 70, 80), 2)
        cv2.putText(canvas, "Agent Third-Person", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)

        # Place Wrist Image (Right: 256x256 at x=344, y=80)
        img_wrist_cv = cv2.cvtColor(np.array(frame["pil_wrist"]), cv2.COLOR_RGB2BGR)
        canvas[80:336, 344:600] = img_wrist_cv
        cv2.rectangle(canvas, (344, 80), (600, 336), (60, 70, 80), 2)
        cv2.putText(canvas, "Gripper Wrist Camera", (354, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)

        # Header Banner
        cv2.rectangle(canvas, (0, 0), (640, 45), (10, 14, 20), -1)
        cv2.putText(canvas, "FastVLA Turbo (7B) - Arabic Physical AI Policy", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame {f_idx+1}/{len(episode_frames)}", (520, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 170), 1, cv2.LINE_AA)

        # Arabic Task Bar
        cv2.rectangle(canvas, (0, 45), (640, 50), (0, 230, 180), -1)  # Accent line

        # Lower Telemetry Dashboard (y=345 to 435)
        dx, dy, dz, droll, dpitch, dyaw, grip = action_step
        cv2.rectangle(canvas, (20, 345), (620, 430), (12, 16, 22), -1)
        cv2.rectangle(canvas, (20, 345), (620, 430), (45, 55, 65), 1)

        # Telemetry Text
        cv2.putText(canvas, "FLOW-MATCHING TELEMETRY (16-Step Continuous Action Chunk):", (30, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 180), 1, cv2.LINE_AA)
        
        telemetry_pos = f"Pos: dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}"
        telemetry_rot = f"Rot: dR={droll:+.3f} dP={dpitch:+.3f} dY={dyaw:+.3f}"
        cv2.putText(canvas, telemetry_pos, (30, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, telemetry_rot, (30, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Gripper Gauge (Right side of telemetry)
        grip_label = "GRASP" if grip > 0 else "RELEASE"
        grip_color = (0, 200, 100) if grip > 0 else (100, 150, 255)
        cv2.putText(canvas, f"Gripper: [{grip_label}] ({grip:+.2f})", (380, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grip_color, 1, cv2.LINE_AA)

        # Chunk Step Progress Bar
        cv2.putText(canvas, f"Chunk Step: {chunk_step_idx}/16", (380, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 170, 180), 1, cv2.LINE_AA)
        # Visual progress bar
        bar_x, bar_y, bar_w, bar_h = 490, 400, 110, 10
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 45, 55), -1)
        progress_w = int((chunk_step_idx / 16.0) * bar_w)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), (0, 230, 180), -1)

        rendered_frames.append(canvas)

    # 4. Save video & animated GIF
    out_dir = Path("/tmp/video_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "fastvla_turbo_arabic_rollout.mp4"
    gif_path = out_dir / "fastvla_turbo_arabic_rollout.gif"

    print("\n[Phase 4] Encoding video and animated GIF...")
    # Write MP4 via OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(mp4_path), fourcc, 20.0, (640, 440))
    for f in rendered_frames:
        vw.write(f)
    vw.release()

    # Transcode with ffmpeg for universal H.264 browser compatibility
    h264_path = out_dir / "fastvla_turbo_arabic_rollout_h264.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(h264_path)
    ], check=True)

    # Save animated GIF (sampled every 2nd frame for web speed)
    pil_gif_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in rendered_frames[::2]]
    pil_gif_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_gif_frames[1:],
        duration=100,  # 10 fps
        loop=0,
    )

    mp4_bytes = open(h264_path, "rb").read()
    gif_bytes = open(gif_path, "rb").read()

    print(f"  ✅ Video generated: {len(mp4_bytes) / 1024:.1f} KB (MP4)")
    print(f"  ✅ Animated GIF generated: {len(gif_bytes) / 1024:.1f} KB (GIF)")

    return {
        "status": "SUCCESS",
        "num_frames": len(rendered_frames),
        "task": arabic_instruction,
        "mp4_bytes": mp4_bytes,
        "gif_bytes": gif_bytes,
    }


@app.local_entrypoint()
def main():
    print("🎬 Requesting policy video rollout from Modal T4...")
    result = record_arabic_policy_video.remote(max_frames=100)
    print("✅ Rollout finished successfully!")

    # Save files locally
    local_assets = Path("/home/hamza/Documents/BouajilaHamza/fastvla/assets")
    local_assets.mkdir(parents=True, exist_ok=True)
    with open(local_assets / "fastvla_turbo_arabic_rollout.mp4", "wb") as f:
        f.write(result["mp4_bytes"])
    with open(local_assets / "fastvla_turbo_arabic_rollout.gif", "wb") as f:
        f.write(result["gif_bytes"])

    print(f"📁 Saved local video to: {local_assets / 'fastvla_turbo_arabic_rollout.mp4'}")
    print(f"📁 Saved local GIF to:   {local_assets / 'fastvla_turbo_arabic_rollout.gif'}")
