# FastVLA RL Integration Plan

## Background & Motivation
Phase 1 of FastVLA (Behavior Cloning) has achieved high performance (7x faster, 1.4x more precise, tiny VRAM footprint). To achieve 100% Task Success, Reinforcement Learning (RL) is required to fine-tune the behavior-cloned "Teacher Policy" through reward-based loops. FastVLA uses a continuous action head (regression) by default, while most HF ecosystem tools (like TRL) are built around discrete tokens. The RL integration must be flexible enough to handle current continuous action heads, future discrete/flow-matching models, and both online/offline RL methods, all without degrading the library's high-performance Unsloth/Triton foundations.

## Scope & Impact
- **Scope:** Add RL capabilities (PPO for online, WBC/DPO for offline) to FastVLA.
- **Impact:** Introduces a new `fastvla/rl/` module and an optional `ValueHead` to the core model. Will not affect inference speed or BC training performance when RL is not active.

## Proposed Solution
1. **Decoupled Value Head Injection:** Dynamically inject a lightweight `ValueHead` into `FastVLAModel` that attaches to the LLM's final hidden states (the same ones fed to the action head) and returns value predictions. This avoids wrapping the model in a rigid TRL wrapper that assumes discrete tokens.
2. **Abstract RL Orchestrator:** Create an agnostic RL module (`fastvla/rl/`) that relies on the `BaseActionHead` interface.
    - **Online RL (PPO):** A custom, highly optimized loop built with `accelerate` handling rollout generation, GAE, and policy/value updates natively to maintain Unsloth/Triton speeds.
    - **Offline RL (DPO / WBC):** Support for DPO (for discrete) and Weighted Behavior Cloning or Continuous DPO (for continuous) configurable on demand.
3. **RL Registry Pattern:** Implement an `RLTrainer` factory that automatically routes to the correct RL loop based on the model's action head type and the user's config.

## Alternatives Considered
- **TRL Integration:** Wrapping FastVLA in `trl.AutoModelForCausalLMWithValueHead`. Rejected because TRL heavily assumes discrete token generation for actions, which would require complex, brittle workarounds for our continuous action heads and could compromise our Unsloth/Triton speed optimizations.

## Implementation Plan
### Phase 1: Core RL Components (Value Head & Abstractions)
- Create `fastvla/rl/__init__.py`.
- Add `ValueHead` component to `fastvla/adapters/action_head.py` or a dedicated `fastvla/adapters/value_head.py`.
- Update `FastVLAModel` in `fastvla/model.py` to optionally initialize and route hidden states to the `ValueHead` when an RL flag is provided.

### Phase 2: Online RL (Accelerate PPO)
- Create `fastvla/rl/ppo.py`.
- Implement a custom PPO loop using `accelerate` (rollout generation, advantages calculation, actor/critic loss computation).
- Ensure compatibility with continuous actions (predicting mean/std and sampling from normal distribution during rollouts) and discrete actions (categorical sampling).

### Phase 3: Offline RL (WBC / DPO)
- Create `fastvla/rl/offline.py`.
- Implement Weighted Behavior Cloning (WBC) for continuous actions.
- Implement DPO variant for continuous/discrete actions as needed.

### Phase 4: Trainer Factory & API
- Create `fastvla/rl/trainer.py` containing the `RLTrainer` factory.
- Ensure seamless integration with the existing `FastVLAConfig` and model registry.

## Verification
- **Unit Tests:** Add tests for `ValueHead` forward passes.
- **Integration Tests:** Create a mock RL environment and verify that the PPO loop runs without crashing and loss converges on a dummy task.
- **Performance Tests:** Benchmark the model forward pass with and without the `ValueHead` to ensure no regression in base inference speed. Verify memory footprint during PPO.

## Migration & Rollback
- The changes are strictly additive. If the RL module causes issues, users can continue using the standard `FastVLATrainer` and the base `FastVLAModel` without the value head flag, ensuring backward compatibility with Phase 1.