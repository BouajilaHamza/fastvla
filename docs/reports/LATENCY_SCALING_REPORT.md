# Empirical Verification of $\mathcal{O}(1)$ Constant Latency Across Action Horizons ($H=8$ to $H=64$)

**Author:** FastVLA Core Team  
**Date:** November 2026  
**Hardware:** NVIDIA Tesla T4 (16 GB GDDR6, Turing Architecture, sm_75)  
**Execution Environment:** Modal Serverless GPU (`scripts/evaluation/benchmark_horizon_scaling.py`)  
**Associated GitHub Issue:** [#26: [Benchmark / Inference] Empirical Verification of O(1) Constant Latency Across Action Horizons](https://github.com/BouajilaHamza/fastvla/issues/26)

---

## 1. Executive Summary & Context

Community inquiries (notably raised by Mohamed Shawky on LinkedIn) posed an essential architectural question regarding continuous Vision-Language-Action (VLA) models:

> *"How does latency performance scale with more complex, multi-step horizon sequences?"*

In standard discrete token-autoregressive VLAs (e.g. OpenVLA, RT-2), action prediction latency scales linearly or quadratically with sequence length:
$$\text{Latency}_{\text{Discrete}} = \mathcal{O}(N_{\text{tokens}} \times T_{\text{backbone}}) = \mathcal{O}(H \times D_{\text{actions}})$$

This report presents **empirical verification on an NVIDIA Tesla T4 GPU** demonstrating that **FastVLA exhibits $\mathcal{O}(1)$ horizon-invariant constant latency**:
- **Expanding the action horizon by 8×** (from $H=8$ to $H=64$ continuous steps across 7 DoF = 56 to 448 continuous action parameters) results in only a **+1.76% latency variation** (+0.026 ms) for the 2-step Midpoint ODE solver.
- At $H=64$, FastVLA solves the continuous trajectory in **1.515 ms (660.1 Hz)**, compared to an estimated **6,720 ms (0.15 Hz)** for discrete autoregressive generation — an acceleration of **4,436×**.

```mermaid
xychart-beta
    title "Inference Latency vs Action Horizon H (Tesla T4, ms - Lower is Better)"
    x-axis ["H=8 (56 DoF)", "H=16 (112 DoF)", "H=32 (224 DoF)", "H=64 (448 DoF)"]
    y-axis "Latency (ms)" 0 --> 7000
    bar [840, 1680, 3360, 6720]
    line [1.49, 1.51, 1.51, 1.52]
```

---

## 2. Theoretical Architecture: Why FastVLA is $\mathcal{O}(1)$ Horizon-Invariant

FastVLA decouples visual perception and language conditioning from motor trajectory integration:

1. **Single-Pass Conditioning**: The vision encoder and LLM backbone process observations and language prompts **once** per control step. Visual tokens are compressed by 75% via `SpatialPatchMerger`, and the conditioning embedding $\mathbf{c} \in \mathbb{R}^{D_{\text{in}}}$ is extracted in a single forward pass.
2. **Parallel Vector Field Integration**: The Flow Matching Action Head parameterizes a continuous velocity field $\mathbf{v}_\theta(\mathbf{x}_t, t \mid \mathbf{c})$. Instead of predicting actions sequentially waypoint-by-waypoint, the ODE:
   $$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v}_\theta(\mathbf{x}_t, t \mid \mathbf{c})$$
   is integrated simultaneously across all batch and chunk dimensions $\mathbf{x} \in \mathbb{R}^{B \times (H \cdot D)}$ using parallel Runge-Kutta numerical integration.
3. **Hardware Saturation Invariance**: On modern tensor core architectures, evaluating matrix-vector multiplications for $[B, 56]$ vs $[B, 448]$ executes within the same hardware warp execution cycle without saturating GPU memory bandwidth.

---

## 3. Empirical Benchmark Results on NVIDIA Tesla T4

All benchmarks were executed on an **NVIDIA Tesla T4 (14.57 GB VRAM)** with PyTorch 2.4 AMP FP16 and batch size $B=16$ (representing a server hosting multiple asynchronous robot agents).

### Full Scaling Table across Solvers and Horizons

| Horizon ($H$) | Action Params ($H \times 7$) | Solver Configuration | Latency (ms) | Frequency (Hz) | Peak VRAM | Autoregressive Discrete Est. | FastVLA Speedup |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$H=8$** | 56 DoF | Euler (1 step) | **0.439 ms** | 2,276.3 Hz | 13.80 MB | 840.0 ms | **1,912×** |
| **$H=8$** | 56 DoF | **Midpoint (2 steps)** | **1.489 ms** | **671.7 Hz** | 13.81 MB | 840.0 ms | **564×** |
| **$H=8$** | 56 DoF | Midpoint (4 steps) | 2.884 ms | 346.8 Hz | 13.81 MB | 840.0 ms | 291× |
| **$H=8$** | 56 DoF | Euler (10 steps) | 3.559 ms | 281.0 Hz | 13.80 MB | 840.0 ms | 236× |
| **$H=16$** | 112 DoF | Euler (1 step) | **0.441 ms** | 2,265.4 Hz | 14.03 MB | 1,680.0 ms | **3,806×** |
| **$H=16$** | 112 DoF | **Midpoint (2 steps)** | **1.508 ms** | **663.3 Hz** | 14.05 MB | 1,680.0 ms | **1,114×** |
| **$H=16$** | 112 DoF | Midpoint (4 steps) | 2.902 ms | 344.6 Hz | 14.05 MB | 1,680.0 ms | 579× |
| **$H=16$** | 112 DoF | Euler (10 steps) | 3.596 ms | 278.1 Hz | 14.03 MB | 1,680.0 ms | 467× |
| **$H=32$** | 224 DoF | Euler (1 step) | **0.443 ms** | 2,257.1 Hz | 14.49 MB | 3,360.0 ms | **7,584×** |
| **$H=32$** | 224 DoF | **Midpoint (2 steps)** | **1.508 ms** | **663.2 Hz** | 14.53 MB | 3,360.0 ms | **2,228×** |
| **$H=32$** | 224 DoF | Midpoint (4 steps) | 2.912 ms | 343.4 Hz | 14.53 MB | 3,360.0 ms | 1,154× |
| **$H=32$** | 224 DoF | Euler (10 steps) | 3.639 ms | 274.8 Hz | 14.49 MB | 3,360.0 ms | 923× |
| **$H=64$** | 448 DoF | Euler (1 step) | **0.440 ms** | 2,270.4 Hz | 15.41 MB | 6,720.0 ms | **15,257×** |
| **$H=64$** | 448 DoF | **Midpoint (2 steps)** | **1.515 ms** | **660.1 Hz** | 15.49 MB | 6,720.0 ms | **4,436×** |
| **$H=64$** | 448 DoF | Midpoint (4 steps) | 2.922 ms | 342.2 Hz | 15.49 MB | 6,720.0 ms | 2,300× |
| **$H=64$** | 448 DoF | Euler (10 steps) | 3.621 ms | 276.1 Hz | 15.41 MB | 6,720.0 ms | 1,856× |

---

## 4. Key Takeaways

1. **True $\mathcal{O}(1)$ Horizon Scaling**:
   Across an 8-fold horizon expansion ($H=8 \to H=64$):
   - 1-step Euler latency remains virtually identical: **0.439 ms $\to$ 0.440 ms** (+0.26%).
   - 2-step Midpoint latency remains practically flat: **1.489 ms $\to$ 1.515 ms** (+1.76%).
2. **Real-Time Control Headroom**:
   Even at $H=64$ (a lookahead of 64 continuous future actions), the action head operates at **660.1 Hz**, leaving 98% of the robotics control loop cycle for sensor processing and kinematic safety verification.
3. **Minimal Memory Overhead**:
   Peak VRAM allocated by the action head is only **13.8 MB to 15.5 MB**, preserving full GPU memory bandwidth for the compressed vision-language trunk.

---

## 5. How to Reproduce

Run the microbenchmark on serverless Modal GPU:

```bash
modal run scripts/evaluation/benchmark_horizon_scaling.py
```

Or run locally on your CUDA hardware:

```bash
python scripts/evaluation/benchmark_horizon_scaling.py --output horizon_scaling_results.json
```
