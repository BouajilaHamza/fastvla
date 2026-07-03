# FastVLA Research Roadmap — Efficient Arabic-Instructable VLA on Commodity GPUs

> **North Star.** Fine-tune a Vision-Language-Action model that reliably follows
> **Arabic** manipulation instructions (LIBERO-AR) on a **single L4 or T4 GPU**,
> reaching task success within a small margin of the English baseline, with
> training that fits in ≤16 GB.
>
> This is a **research** roadmap, not an integration checklist. We treat existing
> LLM-efficiency tooling (QLoRA, paged optimizers, checkpointing, FlashAttention,
> `torch.compile`) as the *floor* — already shipped in v0.3.0 — and spend our
> effort on the parts that are genuinely **VLA-specific** and, for the language
> side, genuinely **novel**.

---

## 0. Framing — where the FLOPs actually go

The lesson from LLM compute history is **not** "factorize everything." It is:
**spend FLOPs only where they move the loss.** For a VLA, per-step cost is
approximately:

```
cost  ∝  (LLM depth)  ×  (sequence length)  ×  (fraction of tokens backpropagated)
```

Empirically, on a 7B-class VLA (e.g. OpenVLA):

| Component            | Share of compute/memory | Implication |
| :------------------- | :---------------------- | :---------- |
| LLM backbone         | **~85–90%**             | Everything expensive lives here |
| Vision tower         | ~5%                     | Cheap in FLOPs, but drives the data pipeline |
| Action head          | rounding error          | A *statistical* bottleneck (variance, train/infer mismatch), never a wall-clock one |

Of the LLM cost, **sequence length is dominated by image tokens** (OpenVLA feeds
~256 image tokens vs ~20 text tokens). This gives us three VLA-native levers that
map directly onto the cost equation — and they are the backbone of Thrust A:

1. **Cut visual tokens** → shrinks *sequence length* through the expensive trunk.
2. **Freeze + cache towers** → shrinks *fraction of tokens backpropagated* and kills the image pipeline.
3. **Non-autoregressive action head** → removes multi-token action decoding at inference.

The Arabic mission then adds a second, orthogonal axis the cost equation hides:
**what the tokens *mean*.** Most open VLA backbones are English-centric; Arabic is
morphologically rich, right-to-left, diglossic (MSA vs dialect), and tokenizes
inefficiently. Grounding Arabic instructions in robot behavior — cheaply, without
destroying the visual grounding already in the backbone — is the novel core
(Thrust B). **The intersection of A and B is the contribution.**

---

## Research Theses (falsifiable)

Every phase below exists to confirm or kill one of these. If the data says no, we
report it and pivot — that is the point of doing research instead of shipping tools.

- **H1 — Visual token budget.** Reducing image tokens 256→128→64 (via pooling /
  token-merging / a learned resampler) preserves LIBERO success within a small
  margin while cutting training step time and KV memory roughly proportionally to
  the token reduction.
- **H2 — Frozen-tower caching.** With a frozen vision encoder, precomputing and
  caching embeddings once removes the vision tower and the image-decode/augment
  path from every epoch after the first, with **zero** accuracy loss.
- **H3 — Non-AR action head.** A flow-matching / regression action head matches or
  beats autoregressive action-token decoding on LIBERO success at a fraction of
  the inference cost. *(v0.3.0 shipped the head; this phase measures the claim.)*
- **H4 — Gradient routing / knowledge insulation.** Routing action-loss gradients
  away from the backbone (while letting an auxiliary language objective through)
  prevents Arabic instruction-tuning from eroding visual grounding, and improves
  sample efficiency (steps-to-success), not just peak memory.
- **H5 — Arabic grounding (novel core).** Naive multilingual transfer is
  insufficient: a measurable English→Arabic **grounding gap** exists on LIBERO-AR.
  It can be closed cheaply — without full-backbone retraining — via a combination
  of (a) tokenizer/embedding adaptation for Arabic, and (b) a lightweight
  language-grounding adapter aligned to the instruction manifold.

---

## Phase 0 — Baselines, data, and instrumentation *(foundation)*

**Goal:** make the two gaps (compute, language) *measurable* before optimizing
anything. No optimization is credible without the baseline it beats.

### 0.1 LIBERO-AR dataset (deliverable in itself)
- Translate LIBERO task instructions to Arabic in **two registers**: Modern
  Standard Arabic (MSA) and at least one dialect (e.g. Egyptian or Gulf).
- Machine-translate → **human validation** pass for correctness and naturalness.
- Version and publish the instruction set + generation script; keep the robot
  trajectories from stock LIBERO (only the language conditioning changes).
- Report **tokenizer fertility** (tokens/word) for the backbone tokenizer on
  Arabic vs English — this feeds H1's sequence-length budget.

### 0.2 Evaluation harness
- LIBERO success-rate eval across suites (Spatial, Object, Goal, Long / LIBERO-10/90).
- Metrics logged per run: success rate, train it/s, peak VRAM (alloc + reserved),
  inference/control Hz, tokens/step. Wire into the existing Modal + W&B harness.

### 0.3 Baselines to beat
- **B-EN**: backbone fine-tuned on English LIBERO — the accuracy ceiling.
- **B-AR-naive**: same recipe, Arabic instructions, no language adaptation — exposes the H5 grounding gap.
- **B-cost**: current FastVLA v0.3.0 numbers — the efficiency floor.

**Exit criteria:** LIBERO-AR published; harness reproduces B-EN and B-AR-naive on
L4 and T4; the English→Arabic grounding gap is a concrete number.

---

## Phase 1 — Efficiency core (Thrust A) *(the L4/T4 mission)*

Tests **H1, H2, H3**. Each lands as a config-flag experiment in FastVLA with an
ablation, not a silent default.

### 1.1 Visual token budget (`token_budget`) — **highest payoff**
- Implement pluggable token reducers: mean/attention pooling, token merging
  (ToMe-style), and a learned Perceiver-style resampler.
- Ablate 256 → 128 → 64 → 32 image tokens; plot success vs step-time vs VRAM.
- **Research question:** where is the accuracy cliff, and does it move for Arabic
  (longer text token budget) vs English?

### 1.2 Frozen-tower feature caching (`cache_vision_features`)
- Precompute frozen vision embeddings to disk once; stream from cache thereafter.
- Measure per-epoch wall-clock with vs without cache (isolate image-pipeline cost
  from GEMM cost). Confirm bit-exact accuracy (H2).

### 1.3 Non-AR action head validation
- Head-to-head: flow-matching / regression head vs autoregressive action-token
  decoding — LIBERO success, inference Hz, train stability.
- Confirm the v0.3.0 STE / CFM fixes hold up on a real task, not just dummy tensors.

**Exit criteria:** a documented Pareto frontier (success vs it/s vs VRAM) on L4 and
T4; each lever's contribution isolated; H1–H3 confirmed or falsified with numbers.

---

## Phase 2 — Arabic grounding research (Thrust B) *(the novel core)*

Tests **H4, H5**. This is the part that is *research*, not tuning.

### 2.1 Characterize the grounding gap
- Decompose B-AR-naive failures: is it *tokenization* (Arabic fragments badly),
  *lexical grounding* (words not mapped to objects/actions), or *instruction
  syntax* (RTL / morphology)? Controlled probes per hypothesis.

### 2.2 Tokenizer & embedding adaptation
- Evaluate Arabic tokenizer fertility; test vocabulary extension / embedding
  re-initialization for high-fertility Arabic tokens.
- Ablate: does cheaper Arabic tokenization alone (fewer text tokens) recover
  success, or is grounding the real blocker?

### 2.3 Language-grounding adapter
- A lightweight adapter aligning Arabic instruction embeddings to the space the
  backbone already grounds (leveraging English↔Arabic parallel instructions from
  Phase 0). Train adapter + action head; keep backbone mostly frozen.
- **Cross-lingual transfer probe:** train grounding on English, test zero-shot on
  Arabic — measure how much transfers "for free" vs needs explicit alignment.

### 2.4 Gradient routing / knowledge insulation (`grad_routing`)
- First-class API to declare per-tower gradient policies (vision / language /
  action as distinct subspaces with independent schedules & precision).
- Test H4: does insulating the backbone from action-loss gradients (while an
  auxiliary language objective flows) protect visual grounding during Arabic
  tuning and improve steps-to-success?

**Exit criteria:** the grounding gap is closed to within the target margin of
B-EN; an ablation attributes the gain to specific mechanisms (tokenizer vs adapter
vs routing); dialect vs MSA robustness reported.

---

## Phase 3 — Integration: efficient Arabic VLA end-to-end

- Combine the winning Phase-1 efficiency config with the winning Phase-2 grounding
  config. Verify the levers **compose** (token reduction must not re-open the
  grounding gap — a real risk, since Arabic needs richer language conditioning).
- Full LIBERO-AR sweep on L4 **and** T4: success per suite, MSA vs dialect,
  it/s, peak VRAM, control Hz.
- **Success definition:** Arabic LIBERO success within target margin of English,
  training within ≤16 GB, on a single commodity GPU.

---

## Phase 4 — Release, reproducibility, write-up

- `v0.4.0`: `token_budget`, `cache_vision_features`, `grad_routing`, Arabic
  grounding adapter, LIBERO-AR loader — all behind documented configs with ablations.
- Reproducible one-command Modal benchmark for every headline number.
- Technical report: the cost-equation framing, the three efficiency levers, the
  Arabic grounding-gap analysis, and the composition result. Negative results
  included.

---

## Non-goals (for now)
- On-policy RL (PPO/GRPO). The mission is supervised/behavior-cloning fine-tuning;
  RL is a different, smaller regime and out of scope until the BC pipeline is solid.
- Per-tower mixed-precision micro-optimization — negligible payoff while the action
  head is rounding-error-sized.
- Difficulty-conditional / early-exit action compute — high complexity, speculative
  payoff; revisit only if Phase 1 leaves accuracy on the table.

## Key risks & open questions
- **Composition risk:** aggressive visual-token reduction may starve exactly the
  cross-modal signal Arabic grounding needs. Phase 3 must test this directly.
- **Data quality:** LIBERO-AR results are only as good as the translation. Human
  validation is not optional.
- **Dialect generalization:** MSA success may not transfer to dialectal phrasing;
  we report both rather than averaging the gap away.
- **Backbone choice:** an Arabic-capable or multilingual backbone may change every
  number here — treated as an explicit ablation axis, not a fixed assumption.

---

## Tracking
Progress is tracked in the pinned roadmap issue. Each phase's exit criteria are the
merge gate for that phase's work. Hypotheses are updated in-place with
CONFIRMED / FALSIFIED / PARTIAL as evidence lands.

---

## Implementation Progress

### Landed
- **H1 scaffold — visual token budget** (`fastvla/adapters/token_reducer.py`):
  four strategies (`mean_pool`, `attention_pool`, `perceiver`, `token_merge`),
  config flags `visual_token_budget` / `token_reduction_strategy`, wired into
  `FastVLAModel.forward`. Unit-tested (shapes, gradient flow, in-model forward).
- **Fusion mode** (`config.fusion_mode`): the existing pipeline fuses via
  **cross-attention** (text = query, visual = K/V), so visual tokens are *not*
  part of the LLM sequence — reducing them cuts cross-attention KV + projection
  cost, not the trunk. Added a `concat` mode that prepends the reduced visual
  tokens to the LLM input, so the token budget **directly shrinks the LLM
  sequence length** — the honest realization of H1's headline lever. Verified:
  8→4 visual tokens takes a (8 text + 8 visual)=16 sequence down to 12.
- **Phase 0 Arabic foundations** (`fastvla/data/arabic.py`): `tokenizer_fertility`
  (the H5 metric), `ArabicInstructionTranslator` (dict / NLLB / lexicon backends),
  seed MSA lexicon, and `LIBEROArabicDataset` (`libero_ar` in the dataset
  factory) that swaps only the language conditioning to isolate the language axis.

### Next
- Benchmark H1 on real GPUs: Pareto frontier (success vs it/s vs VRAM) across
  budgets {256,128,64,32} × strategies × fusion modes, on L4 and T4.
- Phase 0.3 baselines B-EN / B-AR-naive on LIBERO to measure the grounding gap.
- Bootstrap the validated LIBERO-AR corpus (NLLB → human validation → `dict`).
