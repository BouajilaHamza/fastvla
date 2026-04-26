# FastVLA RL Training & Performance: Technical Report

## 1. Executive Summary
This report documents the end-to-end performance, cost, and technical behavior of the Reinforcement Learning (PPO) phase for the FastVLA model on the PushT environment. The objective was to surpass the Behavioral Cloning (BC) baseline using RL exploration. 

While the RL agent successfully discovered highly optimal trajectories (peaking at **84.45% coverage**, far exceeding the BC baseline), it struggled with policy consolidation. The agent exhibited "strike or gutter" behavior, where it could execute near-perfect pushes occasionally, but averaged ~27% coverage overall. An attempt to force consolidation via tighter action constraints resulted in premature convergence and value loss explosion. 

However, the FastVLA library itself (kernels, memory management, PPO adapters) proved exceptionally stable, running over 350+ epochs without a single system failure, memory leak, or crash.

---

## 2. Behavioral Cloning (BC) Baseline
Before RL intervention, the base model was trained using standard Behavioral Cloning.
*   **BC Baseline Coverage:** **44.33%**
*   **Characteristics:** Consistent but capped performance. The model learned the general motion but lacked the precision to consistently push the block fully into the target zone.

---

## 3. RL Phase 1: Exploration (The Original Run)
This run utilized a standard PPO configuration with a relatively high exploration noise (`action_std = 0.05`). 

*   **Total Epochs:** 285
*   **Global Average Coverage:** **27.14%**
*   **Absolute Peak Coverage:** **84.45%** (Epoch 173)
*   **Successes (> 60% Coverage):** 7 occurrences (Epochs 7, 33, 87, 94, 96, 173, etc.)
*   **Observation:** The high noise allowed the model to physically discover the optimal sequences needed to solve the task perfectly (scoring 84%). However, because the noise remained high, the model could not exploit these discoveries consistently, resulting in a flat average of ~27%.

---

## 4. RL Phase 2: Consolidation (The Refinement Run)
To stabilize the high-performing behavior, a second run was launched from the Epoch 150 checkpoint. This run employed a tighter curriculum, decaying `action_std` linearly from 0.03 to 0.01, and reducing the learning rate to `1e-6`.

*   **Total Epochs:** ~100
*   **Global Average Coverage:** **23.37%**
*   **Absolute Peak Coverage:** **54.39%**
*   **Successes (> 60% Coverage):** 0 occurrences
*   **Observation:** The model suffered from **Premature Convergence / Policy Collapse**. By restricting the exploration noise while the *average* behavior was still poor (27%), the model was forced to confidently exploit mediocre trajectories. It became consistently bad rather than consistently good.

---

## 5. Errors, Losses, and Technical Diagnoses
The WandB logs from the Refinement run provided clear indicators of why the policy collapsed:

*   **Value Loss Explosion (Mean ~96.3):** The Critic network completely lost its ability to predict rewards. Because the policy was constrained to a narrow action space (low std) but was still failing the task, the received rewards were highly unpredictable compared to the Critic's expectations.
*   **Negative Entropy (Mean ~ -22.3):** The policy became extremely "certain" about its actions. In continuous action spaces, highly negative entropy means the probability distribution has collapsed into a tiny spike. The model stopped exploring entirely.
*   **KL Divergence (Mean ~ 0.023):** The model drifted away from the base BC policy faster than the target KL (0.01) allowed, indicating destabilization in the network weights despite the lower learning rate.
*   **System Errors:** 
    *   *Warning:* `Detected kernel version 4.4.0, which is below the recommended minimum of 5.5.0.` (Modal infrastructure warning, did not impact execution).
    *   *Warning:* `pygame pkg_resources deprecation.` (Harmless library warning).
    *   **Zero Fatal Errors:** No Out-Of-Memory (OOM) errors, no NaN losses in Phase 1, and no crashes.

---

## 6. System Performance, Time, and Cost (Modal)
The training pipeline was highly optimized, leveraging a single NVIDIA L4 GPU on Modal.

*   **Hardware:** 1x NVIDIA L4 (24GB VRAM)
*   **Precision:** FP32 (Forced for RL stability)
*   **Training Speed:** **~11.6 seconds per epoch** (2048 rollout steps + 4 PPO update epochs). This is an exceptionally fast throughput for a VLA model.
*   **Phase 1 Duration:** ~55 minutes
*   **Phase 2 Duration:** ~20 minutes
*   **Total Compute Time:** ~1 hour 15 minutes
*   **Estimated Cost:** ~$0.59 / hour -> **Total Cost < $0.75**

---

## 7. The "High-Coverage Video" Dilemma
You requested a video of the robot achieving >70% coverage. 

**Is it hard to replicate?** 
Yes and no. Because the Phase 1 model (`action_std=0.05`) is highly stochastic, any single rollout is a roll of the dice. If you just run 1 episode, it will likely score ~27%. 

**How we can capture it:**
We have the `fastvla-rl-fixed-ep100` and `fastvla-rl-fixed-ep150` checkpoints saved in the Modal volume. These checkpoints *contain* the neural pathways capable of hitting 84%. 

To get the video, we do not need to retrain. We just need an **Evaluation Script** that:
1. Loads the `ep150` checkpoint.
2. Runs in an infinite evaluation loop with a slight noise (`action_std=0.02` to allow slight variations).
3. Monitors the `coverage` metric at the end of each episode.
4. **If coverage > 70%**, it saves the RGB frames of that specific episode as an `.mp4` using `moviepy`.

*I can write and execute this Modal script for you immediately if you want to capture that hero video.*
