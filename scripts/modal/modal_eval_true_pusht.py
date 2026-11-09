"""
FastVLA: Scalable High-Throughput Training & Ground-Truth Closed-Loop Evaluation on Push-T.

Features:
- Streaming PyTorch Dataset & DataLoader with multi-worker prefetching (zero RAM OOM).
- Augmented batch sizing (e.g. batch_size=16) to maximize GPU utilization.
- 4-Bit NF4 Quantization + LoRA (r=16) on Qwen-1.5B + SigLIP + 2x2 Spatial Token Compressor.
- Continuous Flow Matching Action Head (16-step continuous chunk, 32 continuous DoF).
- Native Gymnasium Push-T (gym_pusht/PushT-v0) closed-loop evaluation over 10 test episodes.
- Saves authentic rollout video and empirical JSON metrics.
"""

import os
import io
import time
import json
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
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
        "safetensors>=0.4.0",
        "triton>=2.2.0",
        "gymnasium>=0.29.0",
        "gym_pusht>=0.1.5",
        "pymunk>=6.2.0",
        "pygame>=2.5.0",
        "imageio>=2.30.0",
        "imageio-ffmpeg>=0.4.8",
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SDL_VIDEODRIVER": "dummy",
    })
    .add_local_python_source("fastvla")
)

app = modal.App("fastvla-true-pusht-eval", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]


@app.function(gpu="A10G", timeout=14400, secrets=secrets)
def run_end_to_end_pusht_evaluation(
    num_train_steps: int = 10000,
    batch_size: int = 16,
    num_eval_episodes: int = 10,
    max_eval_steps: int = 300,
    eval_every: int = 2500,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import timm
    import torchvision.transforms as T
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb
    from PIL import Image
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import imageio
    import gymnasium as gym
    import gym_pusht

    from fastvla.adapters.token_compressor import SpatialPatchMerger
    from fastvla.adapters.action_head import FlowMatchingActionHead

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("=" * 80)
    print(f"🚀 FASTVLA HIGH-THROUGHPUT TRAINING & EVALUATION ({gpu_name}, {total_vram_gb:.1f} GB VRAM)")
    print(f"   Target Steps:    {num_train_steps} optimizer steps")
    print(f"   Batch Size:      {batch_size} (Augmented to maximize GPU utilization)")
    print(f"   Total Samples:   {num_train_steps * batch_size:,} trajectory chunks")
    print(f"   Closed-Loop:     {num_eval_episodes} physical episodes in gym_pusht/PushT-v0")
    print("=" * 80)

    # 1. Download Dataset Parquet
    print("\n📥 Downloading 'hamzabouajila/ar-pusht-image' from Hugging Face...")
    hf_token = os.environ.get("HF_TOKEN")
    local_path = hf_hub_download(
        repo_id="hamzabouajila/ar-pusht-image",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    table = pq.read_table(local_path)
    print(f"  ✅ Parquet loaded: {len(table)} frames across demonstrations.")

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build streaming Dataset
    class PushTStreamDataset(Dataset):
        def __init__(self, table, chunk_size=16, total_samples=500000):
            self.chunk_size = chunk_size
            self.total_samples = total_samples
            self.images_raw = table["observation.image"]
            self.instructions = table["instruction"].to_pylist()
            self.actions = table["action"].to_pylist()
            episode_indices = table["episode_index"].to_pylist()

            self.episodes = {}
            for idx, ep in enumerate(episode_indices):
                if ep < 180:  # Train on first 180 demonstration episodes
                    if ep not in self.episodes:
                        self.episodes[ep] = []
                    self.episodes[ep].append(idx)
            self.train_eps = sorted(list(self.episodes.keys()))
            print(f"  ✅ Dataset ready: {len(self.train_eps)} episodes indexed.")

        def __len__(self):
            return self.total_samples

        def __getitem__(self, idx):
            chosen_ep = self.train_eps[idx % len(self.train_eps)]
            frame_list = self.episodes[chosen_ep]
            start_f = np.random.randint(0, max(1, len(frame_list) - self.chunk_size))
            anchor_idx = frame_list[start_f]

            raw_img_dict = self.images_raw[anchor_idx].as_py()
            if isinstance(raw_img_dict, dict) and "bytes" in raw_img_dict and raw_img_dict["bytes"] is not None:
                img = Image.open(io.BytesIO(raw_img_dict["bytes"])).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            pv = transform(img)
            instr = self.instructions[anchor_idx]

            chunk_actions = []
            for step in range(self.chunk_size):
                step_idx = frame_list[min(start_f + step, len(frame_list) - 1)]
                raw_act = self.actions[step_idx]
                act = [float(raw_act[0]) / 256.0 - 1.0, float(raw_act[1]) / 256.0 - 1.0]
                chunk_actions.append(act)

            flat_traj = torch.tensor(chunk_actions, dtype=torch.float32).flatten()
            return pv, flat_traj, f"المهمة: {instr}"

    def collate_fn(batch):
        pvs = torch.stack([item[0] for item in batch])
        trajs = torch.stack([item[1] for item in batch])
        prompts = [item[2] for item in batch]
        tok = tokenizer(
            prompts,
            max_length=48,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "images": pvs,
            "trajectories": trajs,
            "input_ids": tok["input_ids"],
            "instructions": prompts,
        }

    dataset = PushTStreamDataset(table, chunk_size=16, total_samples=num_train_steps * batch_size * 2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )
    data_iter = iter(dataloader)

    # 2. Build Model Architecture
    vlm_dim = 1536
    action_chunk_dim = 16 * 2  # 32 continuous DoF

    print("\n[Phase 1] Building FastVLA Model Architecture...")
    vision_encoder = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=True, num_classes=0)
    vision_encoder = vision_encoder.to(device=device, dtype=torch.float16).eval()

    compressor = SpatialPatchMerger(in_dim=1152, out_dim=vlm_dim, merge_factor=2).to(device=device, dtype=torch.float16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)

    action_head = FlowMatchingActionHead(
        input_dim=vlm_dim,
        action_dim=action_chunk_dim,
        hidden_dim=512,
        num_inference_steps=2,
        solver="midpoint",
    ).to(device=device, dtype=torch.float16)

    # 3. Setup Optimizer with scaled learning rate
    trainable_params = [
        {"params": [p for p in llm.parameters() if p.requires_grad], "lr": 3e-4},
        {"params": compressor.parameters(), "lr": 2e-4},
        {"params": action_head.parameters(), "lr": 5e-4},
    ]
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=1e-5)

    print(f"\n[Phase 2] Executing {num_train_steps} High-Throughput Steps (Batch size {batch_size})...")
    torch.cuda.reset_peak_memory_stats()
    t_train_start = time.perf_counter()
    losses = []
    step_latencies = []

    for step in range(1, num_train_steps + 1):
        t0 = time.perf_counter()
        batch = next(data_iter)

        optimizer.zero_grad()
        img_batch = batch["images"].to(device=device, dtype=torch.float16, non_blocking=True)
        tgt_trajectory = batch["trajectories"].to(device=device, dtype=torch.float16, non_blocking=True)
        ids = batch["input_ids"].to(device=device, non_blocking=True)

        with torch.no_grad():
            raw_vis = vision_encoder.forward_features(img_batch)
        compressed_vis = compressor(raw_vis)

        text_emb = llm.get_input_embeddings()(ids).to(device=device, dtype=torch.float16)
        fused_emb = torch.cat([compressed_vis, text_emb], dim=1)

        llm_out = llm(inputs_embeds=fused_emb, output_hidden_states=True)
        pooled = llm_out.hidden_states[-1][:, -1, :]

        loss = action_head.compute_loss(pooled, tgt_trajectory)
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(dt_ms)
        losses.append(loss.item())

        if step % 100 == 0 or step == 1:
            mean_loss_100 = np.mean(losses[-100:]) if len(losses) >= 100 else losses[-1]
            samples_per_sec = (batch_size / (dt_ms / 1000.0))
            current_vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            print(
                f"  Step [{step:5d}/{num_train_steps}] | Flow Loss: {mean_loss_100:.4f} | "
                f"Latency: {dt_ms:.1f} ms ({samples_per_sec:5.1f} smp/s) | VRAM: {current_vram_gb:.2f} GB"
            )

        # Periodic checkpoint
        if step % 2500 == 0:
            ckpt_dir = Path(f"/tmp/checkpoints/step_{step}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(action_head.state_dict(), ckpt_dir / "action_head.pt")
            torch.save(compressor.state_dict(), ckpt_dir / "compressor.pt")
            llm.save_pretrained(ckpt_dir / "llm_adapter")
            print(f"  💾 Saved checkpoint at step {step} to {ckpt_dir}")

    total_train_sec = time.perf_counter() - t_train_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    init_loss = float(np.mean(losses[:50]))
    final_loss = float(np.mean(losses[-50:]))

    print("\n" + "=" * 80)
    print(f"✅ Training Finished in {total_train_sec/60.0:.1f} minutes")
    print(f"   Initial Flow Loss: {init_loss:.4f} -> Final Flow Loss: {final_loss:.4f} ({(init_loss-final_loss)/init_loss*100.0:+.1f}%)")
    print(f"   Peak VRAM Utilization: {peak_vram_gb:.2f} GB / {total_vram_gb:.1f} GB ({peak_vram_gb/total_vram_gb*100.0:.1f}%)")
    print("=" * 80)

    # 4. Closed-Loop Physical Simulation Evaluation
    print(f"\n[Phase 3] Running Closed-Loop Physical Simulation ({num_eval_episodes} Episodes)...")
    llm.eval()
    compressor.eval()
    action_head.eval()

    eval_instruction = "ادفع حرف T إلى المنطقة المستهدفة"
    tok_eval = tokenizer(
        f"المهمة: {eval_instruction}",
        max_length=48,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    episode_results = []
    video_frames = []
    all_control_latencies = []
    replan_interval = 8  # Receding horizon replanning every 8 steps

    for ep_idx in range(num_eval_episodes):
        seed = 1000 + ep_idx
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos", max_episode_steps=max_eval_steps)
        obs, info = env.reset(seed=seed)

        actions_taken = []
        max_coverage = 0.0
        step_count = 0
        is_success = False
        action_buffer = []

        while step_count < max_eval_steps:
            if len(action_buffer) == 0:
                img_pil = Image.fromarray(obs["pixels"]).convert("RGB")
                img_tensor = transform(img_pil).unsqueeze(0).to(device=device, dtype=torch.float16)

                t_inf = time.perf_counter()
                with torch.no_grad():
                    raw_vis = vision_encoder.forward_features(img_tensor)
                    compressed_vis = compressor(raw_vis)
                    text_emb = llm.get_input_embeddings()(tok_eval).to(device=device, dtype=torch.float16)
                    fused_emb = torch.cat([compressed_vis, text_emb], dim=1)
                    llm_out = llm(inputs_embeds=fused_emb, output_hidden_states=True)
                    pooled = llm_out.hidden_states[-1][:, -1, :]

                    pred_traj = action_head(pooled, solver="midpoint", steps=2)
                    torch.cuda.synchronize()

                inf_ms = (time.perf_counter() - t_inf) * 1000.0
                all_control_latencies.append(inf_ms)

                chunk_np = pred_traj[0].view(16, 2).cpu().numpy()
                unnorm_chunk = np.clip((chunk_np + 1.0) * 256.0, 0.0, 512.0)
                action_buffer.extend(unnorm_chunk[:replan_interval].tolist())

            action_to_step = np.array(action_buffer.pop(0), dtype=np.float32)
            obs, reward, terminated, truncated, step_info = env.step(action_to_step)
            actions_taken.append(action_to_step)
            step_count += 1

            cov = float(step_info.get("coverage", 0.0))
            if cov > max_coverage:
                max_coverage = cov

            if cov >= 0.95 or step_info.get("is_success", False):
                is_success = True

            if ep_idx == 0:
                video_frames.append(obs["pixels"])

            if terminated or truncated:
                break

        env.close()

        actions_arr = np.array(actions_taken)
        jerk = float(np.mean(np.linalg.norm(np.diff(actions_arr, n=3, axis=0), axis=-1))) if len(actions_arr) > 3 else 0.0

        episode_results.append({
            "episode": ep_idx,
            "seed": seed,
            "steps": step_count,
            "max_coverage": max_coverage,
            "final_coverage": cov,
            "success": is_success,
            "trajectory_jerk": jerk,
        })

        print(f"  Episode {ep_idx+1:02d}/{num_eval_episodes} (Seed {seed}): Steps={step_count:3d} | Max Cov={max_coverage*100.0:5.1f}% | Final Cov={cov*100.0:5.1f}% | Success={is_success}")

    video_bytes = None
    if video_frames:
        try:
            buf = io.BytesIO()
            imageio.mimsave(buf, video_frames, format="mp4", fps=25)
            video_bytes = buf.getvalue()
            print(f"  🎥 Rendered Episode 0 rollout video ({len(video_frames)} frames).")
        except Exception as e:
            print(f"  ⚠️ Video encoding note: {e}")

    successes = sum(1 for ep in episode_results if ep["success"])
    success_rate = (successes / num_eval_episodes) * 100.0
    mean_max_coverage = float(np.mean([ep["max_coverage"] for ep in episode_results])) * 100.0
    mean_final_coverage = float(np.mean([ep["final_coverage"] for ep in episode_results])) * 100.0
    mean_control_latency = float(np.mean(all_control_latencies)) if all_control_latencies else 0.0
    control_hz = 1000.0 / mean_control_latency if mean_control_latency > 0 else 0.0
    mean_jerk = float(np.mean([ep["trajectory_jerk"] for ep in episode_results]))

    summary = {
        "status": "COMPLETED",
        "benchmark": "Gymnasium Push-T (gym_pusht/PushT-v0)",
        "model_architecture": "FastVLA (SigLIP + SpatialPatchMerger + Qwen-1.5B 4-bit LoRA + Continuous Flow Matching)",
        "num_train_steps": num_train_steps,
        "batch_size": batch_size,
        "total_samples_trained": num_train_steps * batch_size,
        "total_train_minutes": float(total_train_sec / 60.0),
        "initial_flow_loss": init_loss,
        "final_flow_loss": final_loss,
        "loss_reduction_pct": float((init_loss - final_loss) / init_loss * 100.0),
        "peak_vram_gb": peak_vram_gb,
        "vram_utilization_pct": float(peak_vram_gb / total_vram_gb * 100.0),
        "eval_episodes": num_eval_episodes,
        "success_rate_pct": success_rate,
        "mean_max_coverage_pct": mean_max_coverage,
        "mean_final_coverage_pct": mean_final_coverage,
        "mean_inference_latency_ms": mean_control_latency,
        "control_frequency_hz": control_hz,
        "mean_trajectory_jerk": mean_jerk,
        "episode_details": episode_results,
    }

    return {
        "summary": summary,
        "video_bytes": video_bytes,
    }


@app.local_entrypoint()
def main(
    train_steps: int = 10000,
    batch_size: int = 16,
    eval_episodes: int = 10,
    max_steps: int = 300,
):
    print(f"🚀 Triggering 10,000-Step FastVLA Training & Closed-Loop Simulation on Modal (Batch size {batch_size})...")
    res = run_end_to_end_pusht_evaluation.remote(
        num_train_steps=train_steps,
        batch_size=batch_size,
        num_eval_episodes=eval_episodes,
        max_eval_steps=max_steps,
    )

    summary = res["summary"]
    video_bytes = res["video_bytes"]

    out_json = "data/pusht_10k_ground_truth_benchmark_results.json"
    os.makedirs("data", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Benchmark summary written to {out_json}")

    if video_bytes:
        video_path = "assets/fastvla_pusht_10k_rollout.mp4"
        os.makedirs("assets", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        print(f"📹 Physical rollout video saved to {video_path}")

    print("\n" + "=" * 80)
    print("🏆 FASTVLA 10,000-STEP GROUND-TRUTH CLOSED-LOOP VERDICT:")
    print("=" * 80)
    print(f"Benchmark Environment     : {summary['benchmark']}")
    print(f"Total Steps & Batch Size  : {summary['num_train_steps']} steps (Batch size {summary['batch_size']})")
    print(f"Total Samples Trained     : {summary['total_samples_trained']:,}")
    print(f"Total Training Time       : {summary['total_train_minutes']:.1f} minutes")
    print(f"Initial Flow Loss         : {summary['initial_flow_loss']:.4f}")
    print(f"Final Flow Loss           : {summary['final_flow_loss']:.4f} ({summary['loss_reduction_pct']:+.1f}%)")
    print(f"Peak VRAM                 : {summary['peak_vram_gb']:.2f} GB ({summary['vram_utilization_pct']:.1f}% GPU utilization)")
    print(f"Closed-Loop Success Rate  : {summary['success_rate_pct']:.1f}% ({int(summary['success_rate_pct']/10)}/{summary['eval_episodes']} episodes)")
    print(f"Mean Peak Coverage        : {summary['mean_max_coverage_pct']:.1f}%")
    print(f"Mean Final Coverage       : {summary['mean_final_coverage_pct']:.1f}%")
    print(f"Inference Latency (ODE)   : {summary['mean_inference_latency_ms']:.2f} ms")
    print(f"Control Rate              : {summary['control_frequency_hz']:.1f} Hz")
    print(f"Trajectory Jerk           : {summary['mean_trajectory_jerk']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
