import os
import modal
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
hf_key = os.environ.get("HF_API_KEY")
vla_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_key})]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch>=2.2.0", "transformers>=4.40.0", "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0", "peft>=0.9.0", "datasets>=2.16.0",
        "torchvision>=0.17.0", "timm>=0.9.12", "numpy<2.0.0",
        "python-dotenv", "tqdm", "gymnasium", "opencv-python",
        "moviepy", "pygame", "pymunk==6.6.0", "gym-pusht==0.1.4"
    )
    .add_local_dir(Path(__file__).parent.parent, remote_path="/root/project", copy=True)
    .run_commands("pip install -e /root/project")
)

app = modal.App("fastvla-success-benchmark")
volume = modal.Volume.from_name("fastvla-data")

@app.function(image=image, gpu="L4", timeout=7200, volumes={"/data": volume}, secrets=vla_secrets)
def evaluate_success(model_name_or_path: str, is_base: bool, episodes: int = 10):
    import sys
    sys.path.append("/root/project")
    import torch
    import numpy as np
    import gymnasium as gym
    import gym_pusht
    from fastvla import FastVLAModel
    from torchvision import transforms as T
    import time

    print(f"🧐 Evaluating {'BASE' if is_base else 'FastVLA'} | Model: {model_name_or_path}")
    
    if is_base:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        # Official OpenVLA loader
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = base_model # Use official model for baseline
    else:
        model = FastVLAModel.from_pretrained(
            model_name_or_path,
            load_in_4bit=True,
            action_dim=2,
            device_map="cuda"
        )
    model.eval()

    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((model.config.image_size if not is_base else 224, model.config.image_size if not is_base else 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    instruction = "إدفع الكتلة إلى الهدف" if not is_base else "push the T-shaped block to the target position"
    successes = 0
    total_reward = 0
    max_coverage = 0
    latencies = []

    ACTION_MIN = np.array([12.0, 25.0])
    ACTION_MAX = np.array([511.0, 511.0])

    def denormalize_action(norm_action):
        return (norm_action + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_max_cov = 0
        
        if is_base:
            prompt = f"In: {instruction}\nOut:"
            inputs = processor(prompt, obs["pixels"]).to("cuda", dtype=torch.float16)
        else:
            input_ids = model.tokenizer(instruction, return_tensors="pt")["input_ids"].cuda()
        
        for step in range(300):
            img = obs["pixels"] if isinstance(obs, dict) else obs
            
            start_time = time.time()
            with torch.no_grad():
                if is_base:
                    # Official OpenVLA action prediction
                    raw_action = model.predict_action(img, instruction, unnorm_key="pusht")
                else:
                    px = preprocess(img).unsqueeze(0).unsqueeze(0).cuda().to(torch.float16)
                    action_preds, _ = model(pixel_values=px, input_ids=input_ids)
                    # Use the first action in the chunk for simulation control
                    action = action_preds[0].cpu().numpy()[:2]
                    raw_action = denormalize_action(action)
            
            latencies.append(time.time() - start_time)
            obs, reward, terminated, truncated, info = env.step(raw_action)
            
            ep_reward += reward
            ep_max_cov = max(ep_max_cov, info.get('coverage', 0))
            
            if info.get('is_success', False) or ep_max_cov > 0.9:
                successes += 1
                break
            if terminated or truncated: break
            
        total_reward += ep_reward
        max_coverage = max(max_coverage, ep_max_cov)
        print(f"  Episode {ep+1}/{episodes} | Max Coverage: {ep_max_cov:.2%}")

    results = {
        "model": model_name_or_path,
        "is_base": is_base,
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "best_coverage": max_coverage,
        "avg_latency_ms": np.mean(latencies) * 1000
    }
    return results

@app.function(image=image, gpu="L4", timeout=7200, volumes={"/data": volume}, secrets=vla_secrets)
def evaluate_success(model_name_or_path: str, is_base: bool, episodes: int = 5):
    import sys
    sys.path.append("/root/project")
    import torch
    import numpy as np
    import gymnasium as gym
    import gym_pusht
    from fastvla import FastVLAModel
    from torchvision import transforms as T
    import time
    from transformers import AutoModel, AutoProcessor

    print(f"🧐 Evaluating {'BASE' if is_base else 'FastVLA'} | Model: {model_name_or_path}")
    
    if is_base:
        # Robust loader for Baseline
        model = AutoModel.from_pretrained(
            model_name_or_path,
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="cuda"
        )
        # Note: OpenVLA usually needs a processor for official inference
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        model = FastVLAModel.from_pretrained(
            model_name_or_path,
            load_in_4bit=True,
            action_dim=2,
            chunk_size=1, # MATCH OLD TRAINING
            device_map="cuda"
        )
    model.eval()

    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)), # Match training EXACTLY
        T.ToTensor(),
    ])

    instruction = "دفع الحجر إلى الهدف" # MATCHED TO DATASET EXACTLY
    successes = 0
    total_reward = 0
    max_coverage = 0
    latencies = []

    ACTION_MIN = np.array([12.0, 25.0])
    ACTION_MAX = np.array([511.0, 511.0])

    def denormalize_action(norm_action):
        return (norm_action + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_max_cov = 0
        
        if not is_base:
            input_ids = model.tokenizer(instruction, return_tensors="pt")["input_ids"].cuda()
        
        for step in range(300):
            img = obs["pixels"] if isinstance(obs, dict) else obs
            
            start_time = time.time()
            with torch.no_grad():
                if is_base:
                    # Official OpenVLA predict_action helper
                    raw_action = model.predict_action(img, instruction, unnorm_key="pusht")
                else:
                    px = preprocess(img).unsqueeze(0).unsqueeze(0).cuda().to(torch.float16)
                    action_preds, _ = model(pixel_values=px, input_ids=input_ids)
                    action = action_preds[0].cpu().numpy()[:2]
                    raw_action = denormalize_action(action)
            
            latencies.append(time.time() - start_time)
            obs, reward, terminated, truncated, info = env.step(raw_action)
            
            ep_reward += reward
            ep_max_cov = max(ep_max_cov, info.get('coverage', 0))
            
            if info.get('is_success', False) or ep_max_cov > 0.9:
                successes += 1
                break
            if terminated or truncated: break
            
        total_reward += ep_reward
        max_coverage = max(max_coverage, ep_max_cov)
        print(f"  Episode {ep+1}/{episodes} | Max Coverage: {ep_max_cov:.2%}")

    avg_latency = np.mean(latencies)
    results = {
        "model": model_name_or_path,
        "is_base": is_base,
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "best_coverage": max_coverage,
        "avg_latency_ms": avg_latency * 1000,
        "hz": 1.0 / avg_latency if avg_latency > 0 else 0
    }
    return results

@app.local_entrypoint()
def main():
    # Evaluate FastVLA (OLD Hero Checkpoint)
    print("\n--- STAGE 1: FastVLA EVALUATION (OLD CHECKPOINT-2000) ---")
    # This checkpoint was trained with raw coordinates and chunk_size=1
    checkpoint = "/data/checkpoints/arabic-vla-hero/checkpoint-2000"
    fast_results = evaluate_success.remote(checkpoint, is_base=False, episodes=10)

    print("\n" + "="*50)
    print("🚀 PUSH-T PERFORMANCE REPORT (OLD CHECKPOINT-2000)")
    print("="*50)
    print(f"Success Rate   : {fast_results['success_rate']*100:.1f}%")
    print(f"Best Coverage  : {fast_results['best_coverage']*100:.1f}%")
    print(f"Avg Latency    : {fast_results['avg_latency_ms']:.1f} ms")
    print(f"Frequency      : {fast_results['hz']:.2f} Hz")
    print("-" * 50)
    print("NOTE: This run includes the new Tanh and Normalization logic.")

    with open("results/benchmark_old_cp2000.json", "w") as f:
        import json
        json.dump(fast_results, f, indent=2)

