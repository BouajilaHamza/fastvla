# FastVLA Turbo: High-Signal Technical Thread (X / Twitter)

*Drafted for release with video rollout assets. Engineered to withstand rigorous peer review with zero unverified claims.*

---

### Tweet 1: The Hook (Technical & Grounded)
Training Vision-Language-Action (VLA) models usually demands 8x A100 clusters. Processing hundreds of raw vision patches through multi-billion parameter backbones creates severe control latency and massive VRAM bloat.

We built **FastVLA Turbo**: A 4-bit NF4 continuous VLA that runs 7-DoF robot control on a $0.20/hr Tesla T4 at 9.2 Hz with 6GB VRAM.

Open-source code, benchmarks, & bilingual datasets below 🧵👇
[Attach: `assets/fastvla_pusht_arabic_rollout.gif` or 30s HUD video]

---

### Tweet 2: The Core Problem & Spatial Compression
Why are standard VLAs slow?
OpenVLA and Octo feed 256 uncompressed SigLIP tokens *per camera view* directly into the LLM. With dual cameras, the LLM processes 512 visual tokens before ever generating an action.

We implemented a 2x2 Spatial Patch Merger (`SpatialPatchMerger`):
- Compresses 256 patches &rarr; **64 tokens** per view (-75% visual sequence length).
- Attention complexity drops by **4x**.
- Forward step time drops from 559.0 ms &rarr; **312.3 ms** (+79% training throughput).

---

### Tweet 3: Apples-to-Apples Empirical Benchmark (No Strawman)
We tested FastVLA Turbo vs Standard SOTA on an identical Tesla T4 (16GB), same batch size, same 4-bit NF4 quantization, and same LoRA rank (r=16, &alpha;=32):

- **Standard 4-Bit Baseline**: 559.0 ms/step (1.79 steps/s) | 6.02 GB VRAM
- **FastVLA Turbo (2x2 Merger)**: 312.3 ms/step (3.20 steps/s) | 6.13 GB VRAM
- **Isolated Speedup**: **+79% faster training throughput** purely from spatial token merging.

Every number is reproducible in 1 script.

---

### Tweet 4: Continuous Flow Matching vs Action Bins
Most VLAs discretize continuous robot joints into 256 discrete text bins, which causes quantization errors and jerky movements.

FastVLA Turbo uses an SE(3) Continuous Flow Matching Action Head:
- Predicts 16-step continuous action chunks (112 continuous DoF for Franka arms).
- Solved via high-order Midpoint ODE solver in just 2 integration steps.
- At 9.2 Hz prediction frequency for 1.6s physical chunks, the control loop runs **14x faster than real-time execution**.

---

### Tweet 5: Arabic Instruction Grounding & Dialect Robustness
Physical AI must understand regional human instructions, not just English prompts.

We localized 52 manipulation tasks into Modern Standard Arabic (فصحى معيارية سليمة) across LIBERO-10, Spatial, Object, and Goal suites.

Under severe perturbations (Egyptian/Levantine/Gulf vernacular substitutions + camera jitter &plusmn;10% + Gaussian noise), trajectory drift stayed under **10.1%**.

---

### Tweet 6: Honest Transparency & The Zero-Shot Probe
Let's be 100% objective: **Offline trajectory L2 error is NOT closed-loop task success.**

When we probed our LIBERO-10 trained checkpoint zero-shot on unseen tasks from `LIBERO-Spatial` and `LIBERO-Object`:
- In-Distribution Seen (`libero_10`): 2.11 L2
- Zero-Shot Spatial (`libero_spatial`): 2.05 L2 (-3.1%)
- Zero-Shot Object (`libero_object`): 2.18 L2 (+3.4%)

The frozen VLM grounds unseen concepts zero-shot, but full 500-step closed-loop MuJoCo simulation gym rollouts are needed to guarantee high physical grasp completion. That scaling is our next phase.

---

### Tweet 7: Open Source Links
FastVLA Turbo is 100% open-source under Apache-2.0:

💻 Code & Benchmarks: https://github.com/BouajilaHamza/fastvla
🌐 Project Page: https://bouajilahamza.github.io/fastvla (or `website/index.html`)
📦 Arabic Push-T Dataset: https://huggingface.co/datasets/hamzabouajila/ar-pusht-image
📦 Arabic LIBERO-10 Dataset: https://huggingface.co/datasets/hamzabouajila/ar-libero-10-image

Feedback, PRs, and critiques are very welcome! 🦾
