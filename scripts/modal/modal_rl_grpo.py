#!/usr/bin/env python3
"""
Modal serverless entrypoint for FastVLA GRPO Reinforcement Learning.

Features:
1. Zero-Copy Reference Policy evaluation via PEFT adapter disabling.
2. Group Relative Policy Optimization (GRPO) without an actor-critic value head.
3. Continuous 16-step trajectory action chunks (32 DoF for Push-T).
4. Parallel group trajectory sampling with single-pass vision amortization.
"""

import os
import modal

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.43.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.12.0",
        "datasets>=2.20.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "triton>=2.2.0",
        "huggingface_hub>=0.26.0",
        "psutil",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("fastvla")
)

app = modal.App("fastvla-rl-grpo", image=_image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]


@app.function(gpu="T4", timeout=1800, secrets=_secrets)
def run_grpo_training(epochs: int = 50, group_size: int = 4):
    import time
    import torch
    from fastvla import FastVLAModel, FastVLAConfig
    from fastvla.rl.grpo import GRPOTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 80)
    print(f"🚀 EXECUTING FASTVLA ZERO-COPY GRPO ON {gpu_name}")
    print(f"   Epochs: {epochs} | Group Size: {group_size} trajectories/prompt")
    print(f"   Action Space: 16-Step Continuous Trajectory Chunk (32 DoF)")
    print("=" * 80)

    # FastVLA 16-step continuous configuration
    cfg = FastVLAConfig(
        dummy=True,
        vision_hidden_size=256,
        llm_hidden_size=256,
        llm_num_layers=4,
        num_attention_heads=4,
        vocab_size=1000,
        action_dim=32,  # 16-step continuous 2D trajectory chunk
        action_hidden_dim=128,
        use_rl=True,
        use_peft=False,
    )
    model = FastVLAModel(cfg).to(device)

    trainer = GRPOTrainer(
        model=model,
        group_size=group_size,
        beta=0.01,
        learning_rate=2e-5,
        action_std=0.05,
        use_8bit_optimizer=False,  # Use standard AdamW for lightweight test
    )

    print("\n[Phase 1] Starting Critic-Free GRPO Optimization Cycles...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    losses = []
    rewards_history = []

    for ep in range(epochs):
        # Sample an observation (e.g. tabletop observation image + prompt)
        obs = torch.randn(1, 1, 3, 224, 224, device=device)
        input_ids = torch.randint(0, 1000, (1, 16), device=device)

        # 1. Parallel Group Sampling (G continuous trajectory candidates)
        actions, log_probs, mu = trainer.sample_group_trajectories(obs, input_ids, group_size=group_size)

        # 2. Simulated Environment Evaluation (e.g. physical T-block goal alignment reward)
        # Trajectories closer to target have higher reward
        target_traj = torch.zeros(32, device=device)
        group_rewards = []
        for g_idx in range(group_size):
            cand = actions[g_idx]
            dist_err = float(torch.norm(cand - target_traj).item())
            # Reward: 1.0 at perfect match, down to 0.0
            r = float(max(0.0, 1.0 - dist_err / 10.0))
            group_rewards.append(r)

        # Store group in GRPO replay buffer
        trainer.store_group(obs, input_ids, actions, log_probs, group_rewards)

        # 3. Policy update with group-relative advantage & reference KL penalty
        metrics = trainer.update()
        losses.append(metrics["total_loss"])
        rewards_history.append(metrics["mean_reward"])

        # Adaptive exploration noise update
        current_std = trainer.update_action_std(metrics["mean_reward"], threshold=0.7)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  Epoch [{ep+1:03d}/{epochs}] | "
                f"Policy Loss: {metrics['policy_loss']:+.4f} | "
                f"KL Penalty: {metrics['kl_penalty']:.4f} | "
                f"Mean Reward: {metrics['mean_reward']:.4f} | "
                f"Action Std: {current_std:.4f}"
            )

    elapsed_s = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 80)
    print("🏆 FASTVLA GRPO REINFORCEMENT LEARNING SUMMARY:")
    print(f"   Initial Reward:    {rewards_history[0]:.4f} -> Final Reward: {rewards_history[-1]:.4f}")
    print(f"   Peak VRAM on GPU:  {peak_vram_gb:.2f} GB")
    print(f"   Throughput:        {epochs / elapsed_s:.1f} GRPO cycles/s ({elapsed_s * 1000.0 / epochs:.1f} ms/cycle)")
    print("=" * 80)

    return {
        "status": "success",
        "initial_reward": rewards_history[0],
        "final_reward": rewards_history[-1],
        "peak_vram_gb": peak_vram_gb,
        "epochs": epochs,
        "elapsed_s": elapsed_s,
    }


@app.local_entrypoint()
def main(epochs: int = 50, group_size: int = 4):
    run_grpo_training.remote(epochs=epochs, group_size=group_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    main(epochs=args.epochs, group_size=args.group_size)
