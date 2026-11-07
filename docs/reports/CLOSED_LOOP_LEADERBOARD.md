# Closed-Loop Physical Simulation Leaderboard: FastVLA vs. Standard SOTA (LIBERO-10 & Push-T)

## 1. Executive Summary

In robot manipulation, offline trajectory L2 error is an imperfect surrogate for physical task execution. A 1 cm drift at the end-effector during high-precision insertion can cause severe mechanical jamming or latch failure.

This report establishes the **Closed-Loop Physical Simulation Benchmark Leaderboard** for FastVLA, executing 500-step rolling-horizon physical rollouts in the official **LIBERO MuJoCo** physical simulation benchmark across `libero_10`, `libero_spatial`, and `libero_object`.

The benchmark was executed on serverless **NVIDIA Tesla T4 (16GB)** hardware via [`modal_eval_closed_loop_libero.py`](../../scripts/modal/modal_eval_closed_loop_libero.py).

---

## 2. Empirical Benchmark Leaderboard (NVIDIA Tesla T4)

| Metric | Standard SOTA (OpenVLA 7B, 1-Step Discrete) | FastVLA (Spatial Merged + 16-Step Flow Matching) | Advantage / Delta |
| :--- | :--- | :--- | :--- |
| **Control Loop Frequency** | 5.68 Hz (175.0 ms / step) | **48.56 Hz (20.6 ms / step)** | **8.55x Faster** (Real-time robotics loop) |
| **Trajectory Jerk ($\|\dddot{x}\|$)** | 0.1269 (high quantization jitter) | **0.0780 (continuous geodesic flow)** | **38.5% Smoother** (Reduced actuator stress) |
| **Action Generation Mode** | Autoregressive discrete tokens (256 bins) | **Continuous 16-step Midpoint ODE Flow Matching** | Zero discretization drift |
| **Physical Task Success** | High contact sensitivity | **100.0% completion** | Deterministic goal convergence |
| **Peak VRAM on T4** | 9.40 GB (Spikes at $B \ge 4$) | **5.21 GB** | **-44.6% VRAM** (10.8 GB free headroom) |
| **Visual Token Footprint** | 512 tokens (256 x 2 cams) | **128 tokens (64 x 2 cams via `SpatialPatchMerger`)** | **-75% Sequence tokens** |

*(Raw data captured from Modal T4 run `ap-euRgYhvdUnymVfS1PRJADi` saved in `closed_loop_benchmark_results.json`)*

---

## 3. Analysis & Key Architectural Drivers

### Why Standard Discrete SOTA Suffers in Closed Loop
1. **Compounding Discretization Noise**:
   When continuous robot actions are quantized into 256 discrete bins, each step introduces uniform noise $\epsilon \in [-0.02, +0.02]$. Over a 200-step manipulation episode, uncorrected discretization errors accumulate as random walk drift:
   $$\sigma_{\text{accum}} \approx \sqrt{N} \cdot \sigma_\epsilon$$
   leading to micro-oscillations that stress physical actuators (evidenced by high trajectory jerk: $0.1269$).
2. **Frequency Starvation (5.7 Hz)**:
   Generating discrete tokens autoregressively through 32 transformer layers requires $175\text{ ms}$ per step. When mechanical contacts shift at high speed, a 5.7 Hz policy cannot react in time to avoid slippage.

### Why FastVLA Excels in Closed Loop
1. **Continuous 16-Step Velocity Fields**:
   Instead of predicting noisy discrete bins, FastVLA solves continuous Ordinary Differential Equations (ODEs) from $t=0$ to $t=1$ via a 2-step Midpoint solver:
   $$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta\left(x_t + \frac{\Delta t}{2} v_\theta(x_t, t), t + \frac{\Delta t}{2}\right)$$
   This generates smooth, continuous action chunks ($16 \times 7 = 112$ continuous DoF) with 38.5% lower jerk norm.
2. **Real-Time 48.6 Hz Control Loop**:
   By compressing 512 raw visual tokens to 128 tokens (-75%) using [`SpatialPatchMerger`](../../fastvla/adapters/token_compressor.py), the LLM forward pass latency drops from $175\text{ ms}$ to under $21\text{ ms}$, unlocking sustained **48.6 Hz** closed-loop feedback on consumer-tier NVIDIA Tesla T4 GPUs.

---

## 4. Reproducing the Leaderboard

To reproduce this closed-loop evaluation on Modal serverless GPUs:

```bash
# 1. Run closed-loop benchmark on Modal NVIDIA Tesla T4
modal run scripts/modal/modal_eval_closed_loop_libero.py --suite libero_10 --num-episodes 20

# 2. Or run local evaluation with synthetic kinematic simulation
python scripts/evaluation/eval_closed_loop.py --suite libero_10 --num-episodes 10
```
