import os
import modal
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
hf_key = os.environ.get("HF_API_KEY")
vla_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_key})]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.4.0", "transformers>=4.40.0", "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0", "peft>=0.9.0", "datasets>=2.16.0",
        "torchvision>=0.17.0", "timm>=0.9.12", "numpy<2.0.0",
        "python-dotenv", "tqdm", "gymnasium", "opencv-python",
        "pygame", "pymunk==6.6.0", "gym-pusht==0.1.4"
    )
    .pip_install("git+https://github.com/unslothai/unsloth.git")
    .add_local_dir(Path(__file__).parent.parent, remote_path="/root/project", copy=True)
    .run_commands("pip install -e /root/project")
)

app = modal.App("fastvla-high-fidelity-comp")
volume = modal.Volume.from_name("fastvla-data")

@app.function(image=image, gpu="L4", timeout=7200, volumes={"/data": volume}, secrets=vla_secrets)
def run_comparison(episodes=5):
    import gymnasium as gym
    import gym_pusht
    import torch
    import numpy as np
    from fastvla import FastVLAModel
    import torchvision.transforms as T
    
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    preprocess = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])
    instruction = "دفع الحجر إلى الهدف"
    
    ACTION_MIN = np.array([12.0, 25.0])
    ACTION_MAX = np.array([511.0, 511.0])
    def denorm(a): return (a + 1.0) / 2.0 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN

    # Note: OLD CP was 2-layer, NEW CP is 7B. 
    # The library handles this via config.
    models_to_test = [
        {
            "name": "OLD (Hero) CP-2000",
            "path": "/data/checkpoints/arabic-vla-hero/checkpoint-2000",
            "chunk_size": 1,
            "desc": "2-Layer, No-Norm, Single-Step"
        },
        {
            "name": "NEW (Precision) CP-3000",
            "path": "/data/checkpoints/arabic-vla-precision-optimized/checkpoint-3000",
            "chunk_size": 4,
            "desc": "7B-Full, Normalized, 4-Step Chunking"
        }
    ]
    
    overall_results = {}

    for m_cfg in models_to_test:
        print(f"\n🚀 EVALUATING: {m_cfg['name']} ({m_cfg['desc']})")
        try:
            model = FastVLAModel.from_pretrained(
                model_name_or_path=m_cfg['path'],
                load_in_4bit=True,
                action_dim=2,
                chunk_size=m_cfg['chunk_size'],
                hf_token=os.environ.get("HF_TOKEN")
            )
            model.vision_encoder.model.config.interpolate_pos_encoding = True
            model.eval()

            successes = 0
            coverages = []
            
            for ep_idx in range(episodes):
                obs, _ = env.reset()
                max_ep_cov = 0
                input_ids = model.tokenizer(instruction, return_tensors="pt")["input_ids"].cuda()
                
                # PushT limit is 300 steps
                curr_step = 0
                while curr_step < 300:
                    img = obs['pixels'] if isinstance(obs, dict) else obs
                    px = preprocess(img).unsqueeze(0).unsqueeze(0).cuda().half()
                    
                    with torch.no_grad():
                        action_preds, _ = model(pixel_values=px, input_ids=input_ids)
                    
                    # --- TEMPORAL EXECUTION ---
                    # If chunk_size > 1, execute all actions in the chunk
                    actions_to_exec = action_preds.cpu().numpy()
                    # Reshape if flattened: [chunk_size * 2] -> [chunk_size, 2]
                    actions_to_exec = actions_to_exec.reshape(-1, 2)
                    
                    for i in range(len(actions_to_exec)):
                        if curr_step >= 300: break
                        
                        raw_action = denorm(actions_to_exec[i])
                        obs, reward, done, trunc, info = env.step(raw_action)
                        max_ep_cov = max(max_ep_cov, info.get('coverage', 0))
                        curr_step += 1
                        if done or trunc: break
                    
                    if done or trunc: break
                
                coverages.append(max_ep_cov)
                if max_ep_cov > 0.90: successes += 1
                print(f"  Ep {ep_idx+1} | Max Coverage: {max_ep_cov*100:.2f}%")
            
            overall_results[m_cfg['name']] = {
                "success": successes / episodes,
                "max_cov": np.max(coverages),
                "avg_cov": np.mean(coverages)
            }
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Failed to evaluate {m_cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return overall_results

@app.local_entrypoint()
def main():
    results = run_comparison.remote(episodes=5)
    
    print("\n" + "="*70)
    print("🏆 HIGH-FIDELITY BATTLE: OLD HERO VS NEW PRECISION")
    print("="*70)
    print(f"{'Model Checkpoint':<25} | {'Success':<10} | {'Max Cov':<10} | {'Avg Cov':<10}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<25} | {res['success']*100:>8.1f}% | {res['max_cov']*100:>8.2f}% | {res['avg_cov']*100:>8.2f}%")
    print("="*70)
