"""
HuggingFace Baseline vs FastVLA — Modal L4 & T4
================================================
Runs the unmodified upstream openvla/openvla-7b model via plain HF
transformers (bf16, trust_remote_code) and records the same metrics
collected by examples/modal_production_benchmark.py — so the two JSON
artifacts can be diffed apples-to-apples.

Usage:
    WANDB_API_KEY=... HF_API_KEY=... \\
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \\
    modal run --detach examples/modal_baseline_benchmark.py
"""

import os
import json
import time
import modal

_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3", add_python="3.11")
    .apt_install("git")
    .pip_install(
        # OpenVLA prismatic code targets exactly these versions.
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "timm==0.9.16",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.11.0",
        "datasets>=2.20.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0,<2.0.0",
        "pillow>=10.0.0",
        "huggingface_hub>=0.26.0",
        "wandb>=0.18.0",
        "sentencepiece",
    )
)

app = modal.App("fastvla-baseline-benchmark", image=_image)

_secrets = [
    modal.Secret.from_dict({
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HF_TOKEN":      os.environ.get("HF_API_KEY", ""),
        "HF_API_KEY":    os.environ.get("HF_API_KEY", ""),
    })
]


def _bench_openvla_baseline(num_iters: int, train_steps: int,
                            B_inf: int, B_tr: int, load_in_4bit: bool) -> dict:
    """Plain HF transformers OpenVLA-7B baseline."""
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    res = {"model": "openvla-7b-baseline", "load_in_4bit": load_in_4bit}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    model_id = "openvla/openvla-7b"

    print(f"\n  ── HF baseline {model_id} (4bit={load_in_4bit}) ──")

    t0 = time.time()
    try:
        kwargs = dict(
            trust_remote_code=True,
            token=hf_token,
            # OpenVLAForActionPrediction predates `_supports_sdpa`; force eager
            # to skip the attribute lookup transformers does for sdpa selection.
            attn_implementation="eager",
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = {"": 0}
            # Note: omit low_cpu_mem_usage — combination with OpenVLA's custom
            # PrismaticForConditionalGeneration.__init__ triggers .to() on
            # bnb-quantised submodules, which raises.
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["low_cpu_mem_usage"] = True

        # OpenVLA's auto_map only registers AutoModelForVision2Seq, which is
        # missing on the older transformers shipped in the nvcr base image.
        # Load the custom class directly via auto_map -> dynamic module.
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        class_ref = config.auto_map["AutoModelForVision2Seq"]
        ModelClass = get_class_from_dynamic_module(
            class_ref, model_id, token=hf_token
        )
        model = ModelClass.from_pretrained(model_id, config=config, **kwargs)
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        # Vanilla QLoRA baseline = 4-bit base + LoRA adapters via PEFT.
        if load_in_4bit:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=32,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    except Exception as exc:
        res["error_load"] = str(exc)
        print(f"    LOAD FAILED: {exc}")
        return res
    res["model_load_s"] = round(time.time() - t0, 2)
    print(f"    Loaded in {res['model_load_s']}s")

    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # OpenVLA's fused vision tower expects 6 channels (DINOv2 + SigLIP
    # concatenated by the upstream PrismaticProcessor) at 224x224.
    H = W = 224
    pixel = torch.randn(B_inf, 6, H, W, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 32000, (B_inf, 32), device=device)
    attention_mask = torch.ones_like(input_ids)

    # ── Inference ────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            for _ in range(3):
                model(pixel_values=pixel, input_ids=input_ids,
                      attention_mask=attention_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies = []
        with torch.no_grad():
            for _ in range(num_iters):
                t = time.perf_counter()
                model(pixel_values=pixel, input_ids=input_ids,
                      attention_mask=attention_mask)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t) * 1000)
        avg = sum(latencies) / len(latencies)
        res.update({
            "inference_avg_ms": round(avg, 2),
            "inference_min_ms": round(min(latencies), 2),
            "inference_p95_ms": round(sorted(latencies)[int(0.95 * num_iters)], 2),
            "control_hz":       round(1000.0 / avg, 2),
            "vram_after_inf_gb": round(
                torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0, 3
            ),
        })
        print(f"    INF avg {avg:.1f} ms  |  {res['control_hz']} Hz  |  "
              f"VRAM {res['vram_after_inf_gb']} GB")
    except Exception as exc:
        res["error_inference"] = str(exc)
        print(f"    INFERENCE FAILED: {exc}")

    # ── Training step ────────────────────────────────────────────────────
    try:
        pixel_tr = torch.randn(B_tr, 6, H, W, device=device, dtype=torch.bfloat16)
        ids_tr = torch.randint(0, 32000, (B_tr, 32), device=device)
        mask_tr = torch.ones_like(ids_tr)
        labels = ids_tr.clone()  # standard CE on token outputs
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        model.train()
        step_times = []
        for step in range(train_steps):
            t = time.perf_counter()
            optimizer.zero_grad()
            out = model(
                pixel_values=pixel_tr,
                input_ids=ids_tr,
                attention_mask=mask_tr,
                labels=labels,
            )
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            if step >= 2:
                step_times.append(elapsed)
            if step % 2 == 0:
                print(f"      step {step:3d}  loss={float(loss):.4f}  {elapsed*1000:.0f} ms")
        avg = sum(step_times) / len(step_times) * 1000 if step_times else 0
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else 0
        res.update({
            "train_step_avg_ms":  round(avg, 2),
            "train_its":          round(1000.0 / avg, 2) if avg else 0,
            "peak_vram_alloc_gb": round(peak, 3),
            "peak_vram_res_gb":   round(reserved, 3),
        })
        print(f"    TRAIN {avg:.0f} ms/step  |  {res['train_its']} it/s  |  "
              f"peak VRAM {peak:.2f} GB")
    except Exception as exc:
        res["error_train"] = str(exc)
        print(f"    TRAIN FAILED: {exc}")

    return res


def _run_baseline(gpu_label: str, load_in_4bit: bool) -> dict:
    import torch
    import wandb

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_total  = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available() else 0
    )

    print(f"\n{'='*60}")
    print(f"  HF Baseline — OpenVLA-7B — {gpu_label}  (4bit={load_in_4bit})")
    print(f"  Device  : {device_name}")
    print(f"  VRAM    : {vram_total:.1f} GB")
    print(f"{'='*60}")

    wandb.init(
        project="fastvla-baseline-benchmark",
        name=f"baseline-{gpu_label}-{'4bit' if load_in_4bit else 'bf16'}-{int(time.time())}",
        config={"gpu": gpu_label, "device": device_name,
                "vram_gb": vram_total, "load_in_4bit": load_in_4bit},
        reinit=True,
    )

    per = _bench_openvla_baseline(
        num_iters=30, train_steps=10, B_inf=1, B_tr=1, load_in_4bit=load_in_4bit,
    )

    all_results = {
        "gpu_label": gpu_label,
        "device_name": device_name,
        "vram_total_gb": vram_total,
        "load_in_4bit": load_in_4bit,
        "openvla_baseline": per,
    }
    wandb.log({k: v for k, v in per.items() if isinstance(v, (int, float, bool))})
    wandb.finish()

    print(f"\n  DONE {gpu_label}")
    for k, v in per.items():
        print(f"    {k:<28} {v}")
    return all_results


# L4: bf16 baseline (16.5 GB fits in 22 GB). T4: must use 4-bit (15 GB).

@app.function(gpu="L4", timeout=3600, secrets=_secrets)
def baseline_l4_bf16() -> dict:
    return _run_baseline("L4", load_in_4bit=False)


@app.function(gpu="L4", timeout=3600, secrets=_secrets)
def baseline_l4_4bit() -> dict:
    return _run_baseline("L4", load_in_4bit=True)


@app.function(gpu="T4", timeout=3600, secrets=_secrets)
def baseline_t4_4bit() -> dict:
    return _run_baseline("T4", load_in_4bit=True)


@app.local_entrypoint()
def main():
    modal.enable_output()
    print("\n🚀  HF OpenVLA-7B baseline — Modal L4(bf16,4bit) + T4(4bit)\n")

    futures = {
        "L4_bf16": baseline_l4_bf16.spawn(),
        "L4_4bit": baseline_l4_4bit.spawn(),
        "T4_4bit": baseline_t4_4bit.spawn(),
    }

    results = {}
    for label, fut in futures.items():
        print(f"⏳  Waiting for {label} ...")
        try:
            results[label] = fut.get()
        except Exception as exc:
            print(f"❌  {label} failed: {exc}")
            results[label] = {"error": str(exc)}

    print("\n" + "=" * 60)
    print(json.dumps(results, indent=2))

    with open("baseline_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📄  Results → baseline_benchmark_results.json")
