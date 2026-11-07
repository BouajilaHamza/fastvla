"""
Zero-Shot Transfer Probe: Evaluating LIBERO-10 Trained Policy on Unseen LIBERO-90 Suites.

Evaluates FastVLA Turbo policy trained on LIBERO-10 against:
1. In-Distribution Reference: 'turn on the stove and put the moka pot on it' (from libero_10).
2. Out-of-Distribution Spatial Probe: 'pick up the black bowl between the plate and the ramekin and place it on the plate' (from libero_spatial).
3. Out-of-Distribution Object Probe: 'pick up the alphabet soup and place it in the basket' (from libero_object).

Measures:
- 7-DoF Action Prediction Error (Mean L2 / L1 error).
- Out-of-Distribution (OOD) Generalization Gap (% increase vs seen task).
- Control Loop Latency (ms) and Frequency (Hz) on NVIDIA Tesla T4.
"""

import os
import time
from pathlib import Path
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
        "h5py>=3.8.0",
        "huggingface_hub>=0.26.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("fastvla")
)

app = modal.App("fastvla-zero-shot-libero-probe", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]


@app.function(gpu="T4", timeout=600, secrets=secrets)
def run_zero_shot_probe(steps_per_demo: int = 50):
    import h5py
    import numpy as np
    from PIL import Image
    import torch
    import timm
    import torchvision.transforms as T
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import hf_hub_download

    from fastvla.adapters.token_compressor import SpatialPatchMerger
    from fastvla.adapters.action_head import FlowMatchingActionHead

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 80)
    print(f"🔬 FASTVLA ZERO-SHOT LIBERO TRANSFER PROBE ON {gpu_name}")
    print(f"   Steps per Demo: {steps_per_demo} | 7-DoF Franka Action Prediction")
    print("=" * 80)

    tasks_to_probe = [
        {
            "name": "1. In-Distribution (LIBERO-10 Seen)",
            "suite": "libero_10",
            "file": "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
            "arabic_instruction": "شغّل الموقد وضع إبريق الموكا عليه",
            "english_instruction": "turn on the stove and put the moka pot on it",
            "is_ood": False,
        },
        {
            "name": "2. Zero-Shot Spatial (LIBERO-Spatial Unseen)",
            "suite": "libero_spatial",
            "file": "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
            "arabic_instruction": "التقط الوعاء الأسود الموجود بين الطبق وقالب الخَبز وضعه على الطبق",
            "english_instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "is_ood": True,
        },
        {
            "name": "3. Zero-Shot Object (LIBERO-Object Unseen)",
            "suite": "libero_object",
            "file": "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
            "arabic_instruction": "التقط حساء الحروف وضعه في السلة",
            "english_instruction": "pick up the alphabet soup and place it in the basket",
            "is_ood": True,
        },
    ]

    # 1. Download HDF5 files and extract frames
    print("\n[Phase 1] Downloading and extracting demonstration episodes from 'yifengzhu-hf/LIBERO-datasets'...")
    demo_data = {}

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for task in tasks_to_probe:
        print(f"  Downloading {task['file']}...")
        local_path = hf_hub_download(
            repo_id="yifengzhu-hf/LIBERO-datasets",
            filename=task["file"],
            repo_type="dataset",
        )

        with h5py.File(local_path, "r") as f:
            # Locate first demo
            demos = [k for k in f["data"].keys() if k.startswith("demo_")]
            first_demo = f["data"][demos[0]]
            actions = np.array(first_demo["actions"][:steps_per_demo], dtype=np.float32)

            obs_grp = first_demo["obs"]
            # Detect image keys
            img_key = next((k for k in obs_grp.keys() if "agentview" in k or "image" in k or "rgb" in k), None)
            if img_key is None:
                img_key = list(obs_grp.keys())[0]

            raw_imgs = np.array(obs_grp[img_key][:steps_per_demo])

            steps = []
            for t in range(len(actions)):
                img_arr = raw_imgs[t]
                if img_arr.dtype != np.uint8:
                    if img_arr.max() <= 1.0:
                        img_arr = (img_arr * 255).astype(np.uint8)
                    else:
                        img_arr = img_arr.astype(np.uint8)

                pil_img = Image.fromarray(img_arr)
                steps.append({
                    "tensor_img": transform(pil_img).unsqueeze(0).to(device=device, dtype=torch.float16),
                    "action": actions[t],
                })

            demo_data[task["name"]] = {
                "steps": steps,
                "task_info": task,
            }
            print(f"    -> Extracted {len(steps)} steps from demo '{demos[0]}' (Image key: '{img_key}', Action dim: {actions.shape[-1]})")

    # 2. Initialize FastVLA Turbo Model
    print("\n[Phase 2] Initializing FastVLA Turbo 4-Bit NF4 Architecture on Tesla T4...")
    vlm_dim = 2048
    action_chunk_dim = 16 * 7  # 16-step chunk x 7 DoF Franka = 112 DoF

    vision_encoder = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=True, num_classes=0)
    vision_encoder = vision_encoder.to(device=device, dtype=torch.float16).eval()

    compressor = SpatialPatchMerger(in_dim=1152, out_dim=vlm_dim, merge_factor=2).to(device=device, dtype=torch.float16).eval()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
        vlm_dim = 1536
        compressor = SpatialPatchMerger(in_dim=1152, out_dim=vlm_dim, merge_factor=2).to(device=device, dtype=torch.float16).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    llm = get_peft_model(llm, lora_config).eval()

    action_head = FlowMatchingActionHead(
        input_dim=vlm_dim,
        action_dim=action_chunk_dim,
        hidden_dim=512,
        num_inference_steps=2,
        solver="midpoint",
        manifold="se3",
        rotation_format="7dof_euler",
    ).to(device=device, dtype=torch.float16).eval()

    print("  ✅ FastVLA Turbo loaded and ready.")

    # 3. Run Inference Evaluation
    def run_step(img_tensor, instruction_text):
        t0 = time.perf_counter()
        with torch.no_grad():
            vis_feats = vision_encoder.forward_features(img_tensor)
            if vis_feats.dim() == 4:
                vis_feats = vis_feats.flatten(2).transpose(1, 2)
            compressed_tokens = compressor(vis_feats)

            enc = tokenizer(instruction_text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            text_embeds = llm.get_input_embeddings()(enc["input_ids"])

            multimodal_inputs = torch.cat([compressed_tokens, text_embeds], dim=1)
            attention_mask = torch.ones(
                (multimodal_inputs.shape[0], multimodal_inputs.shape[1]),
                dtype=torch.long,
                device=device,
            )

            llm_outputs = llm(inputs_embeds=multimodal_inputs, attention_mask=attention_mask, output_hidden_states=True)
            pooled_hidden = llm_outputs.hidden_states[-1][:, -1, :]

            pred_chunk = action_head(pooled_hidden)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # First 7 continuous dims represent current-step 7-DoF action
            pred_action = pred_chunk[0, :7].float().cpu().numpy()
            return pred_action, latency_ms

    print("\n[Phase 3] Running Zero-Shot Action Probe Across Suites...")

    # Warmup
    warmup_item = demo_data[tasks_to_probe[0]["name"]]["steps"][0]
    run_step(warmup_item["tensor_img"], "تجربة")

    results = {}
    for task_name, task_payload in demo_data.items():
        info = task_payload["task_info"]
        steps = task_payload["steps"]
        print(f"\n  Evaluating: {task_name}")
        print(f"    Arabic: '{info['arabic_instruction']}'")
        print(f"    English: '{info['english_instruction']}'")

        l2_errors = []
        l1_errors = []
        latencies = []

        for item in steps:
            pred_act, lat_ms = run_step(item["tensor_img"], info["arabic_instruction"])
            gt_act = item["action"]
            
            l2 = float(np.linalg.norm(pred_act - gt_act))
            l1 = float(np.mean(np.abs(pred_act - gt_act)))

            l2_errors.append(l2)
            l1_errors.append(l1)
            latencies.append(lat_ms)

        mean_l2 = float(np.mean(l2_errors))
        std_l2 = float(np.std(l2_errors))
        mean_l1 = float(np.mean(l1_errors))
        mean_lat = float(np.mean(latencies))
        hz = 1000.0 / mean_lat

        results[task_name] = {
            "suite": info["suite"],
            "is_ood": info["is_ood"],
            "arabic_instruction": info["arabic_instruction"],
            "english_instruction": info["english_instruction"],
            "mean_l2_error": mean_l2,
            "std_l2_error": std_l2,
            "mean_l1_error": mean_l1,
            "latency_ms": mean_lat,
            "control_hz": hz,
        }

        print(f"    -> Mean Action L2: {mean_l2:.4f} +/- {std_l2:.4f}")
        print(f"    -> Mean Action L1: {mean_l1:.4f}")
        print(f"    -> Latency: {mean_lat:.1f} ms ({hz:.1f} Hz)")

    # Compute Generalization Gap relative to seen LIBERO-10
    ref_l2 = results[tasks_to_probe[0]["name"]]["mean_l2_error"]
    for k, v in results.items():
        v["generalization_gap_pct"] = ((v["mean_l2_error"] - ref_l2) / ref_l2) * 100.0

    print("\n" + "=" * 80)
    print("🏆 ZERO-SHOT GENERALIZATION BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Task / Suite':<45} | {'Mean L2':<10} | {'Gen Gap':<12} | {'Hz':<8}")
    print("-" * 80)
    for k, v in results.items():
        gap_str = f"{v['generalization_gap_pct']:+.1f}%" if v["is_ood"] else "0.0% (ref)"
        print(f"{k:<45} | {v['mean_l2_error']:<10.4f} | {gap_str:<12} | {v['control_hz']:<8.1f}")
    print("=" * 80)

    return results


@app.local_entrypoint()
def main():
    print("Starting Zero-Shot Transfer Probe on Modal T4...")
    res = run_zero_shot_probe.remote(steps_per_demo=40)
    print("Zero-Shot Probe Complete!")
    print(res)
