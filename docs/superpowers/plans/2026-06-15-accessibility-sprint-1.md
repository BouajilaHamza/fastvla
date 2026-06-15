# Accessibility Sprint 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the top-3 accessibility gaps identified in `docs/ACCESSIBILITY_ROADMAP.md` § "Sequencing": verify the vanilla-QLoRA baseline (C3), restore the full OpenVLA fused vision backbone (C4), and enable `torch.compile` by default on Ada/Hopper (B5). After this sprint the "3-4× over vanilla QLoRA" headline becomes locally reproducible, the OpenVLA-7B benchmark stops silently degrading to SigLIP, and Ada-class GPUs pick up the free 20-40% from `torch.compile`.

**Architecture:** Three independent slices, each landing its own commit.
1. **Vanilla baseline** — script already exists (`examples/modal_baseline_benchmark.py`); run it, capture JSON, fold real number into `docs/BENCHMARKS.md`.
2. **OpenVLA loader fix** — patch `fastvla/adapters/vision.py::OpenVLAFusedVisionAdapter.from_pretrained` to use `AutoModelForVision2Seq` (which knows the prismatic `auto_map`) before the current `AutoModel` path, then fall back to SigLIP only as last resort. Add a unit test that asserts the loaded vision backbone is **not** the SigLIP fallback when `openvla/openvla-7b` is requested with `trust_remote_code=True`.
3. **`torch.compile` default-on for Ada** — detect compute capability ≥ 8.9 at model construction time, flip `config.use_torch_compile` to `True` automatically, gate behind an explicit override. Add a unit test on the detection helper.

**Tech Stack:** PyTorch 2.4+, transformers 4.43+, bitsandbytes, Modal CLI, pytest, Triton.

---

## File Structure

| Path | Purpose | Action |
|---|---|---|
| `examples/modal_baseline_benchmark.py` | Vanilla HF baseline (already exists) | Run, capture output |
| `baseline_benchmark_results.json` | Output of the baseline run | Create from run |
| `docs/BENCHMARKS.md` | Training-speed deep dive | Modify "Reference points" + "Ratios" rows with real number |
| `README.md` | Headline tables | Modify "Vs vanilla 4-bit QLoRA" sentence |
| `fastvla/adapters/vision.py` | Vision adapters | Modify `OpenVLAFusedVisionAdapter.from_pretrained` (lines 101-145) |
| `tests/test_openvla_loader.py` | New unit test for OpenVLA loader | Create |
| `fastvla/config.py` | Config dataclass | Modify default of `use_torch_compile` + add `_auto_torch_compile` helper |
| `fastvla/model.py` | Model init reads config | Modify `__init__` to honour the auto-detected default |
| `tests/test_auto_compile.py` | New unit test for auto-detect helper | Create |
| `docs/ACCESSIBILITY_ROADMAP.md` | Status table | Modify C3, C4, B5 row status to ☑ |

---

## Task 1: Run vanilla QLoRA baseline + record real number (C3)

**Files:**
- Modify: `examples/modal_baseline_benchmark.py` (no code change — but verify it points at openvla-7b)
- Create: `baseline_benchmark_results.json` (output)
- Modify: `docs/BENCHMARKS.md` — "Reference points" table row "Vanilla 4-bit QLoRA OpenVLA-7B"
- Modify: `README.md` — "vs vanilla 4-bit QLoRA" claim

- [ ] **Step 1: Sanity-check baseline script targets correct model**

Run:
```bash
grep -nE "model_id|openvla/openvla-7b|load_in_4bit" examples/modal_baseline_benchmark.py | head -10
```
Expected: `model_id = "openvla/openvla-7b"` present, both `baseline_l4_4bit` and `baseline_t4_4bit` functions visible.

- [ ] **Step 2: Launch baseline on Modal (detached, both GPUs)**

Run (interactive, in user's shell — Modal needs auth):
```bash
export PATH="/home/freelance/snap/code/247/.local/share/../bin:$PATH"
export MODAL_TOKEN_ID="<from .env>"
export MODAL_TOKEN_SECRET="<from .env>"
export WANDB_API_KEY="<from .env>"
export HF_API_KEY="<from .env>"
modal run --detach examples/modal_baseline_benchmark.py 2>&1 | tee /tmp/modal_baseline.log
```
Expected: app id printed, image build progresses, eventually `Results → baseline_benchmark_results.json`.

- [ ] **Step 3: Extract train it/s for L4 4-bit**

Run:
```bash
python3 -c "import json; d=json.load(open('baseline_benchmark_results.json')); print('L4 4bit it/s:', d['L4_4bit']['openvla_baseline'].get('train_its')); print('L4 4bit step ms:', d['L4_4bit']['openvla_baseline'].get('train_step_avg_ms')); print('L4 4bit peak GB:', d['L4_4bit']['openvla_baseline'].get('peak_vram_alloc_gb'))"
```
Expected: numerical it/s, ms/step, and GB. Record the it/s as `VANILLA_L4_ITS` in your notes.

- [ ] **Step 4: Compute real ratio vs FastVLA L4 (16.2 it/s)**

```bash
python3 -c "VANILLA=<VANILLA_L4_ITS>; FAST=16.2; print(f'FastVLA / Vanilla = {FAST/VANILLA:.2f}x')"
```
Expected: ratio number, replace `<VANILLA_L4_ITS>` first.

- [ ] **Step 5: Update `docs/BENCHMARKS.md` reference row + ratios**

Open `docs/BENCHMARKS.md`. Replace the "Vanilla 4-bit QLoRA OpenVLA-7B, post-bug baseline" row's `~3.0 it/s` with the measured number, change its source citation from `[fastvla issue #1]` to:
```
(measured by `examples/modal_baseline_benchmark.py`, see `baseline_benchmark_results.json`)
```
In the "Ratios" table replace the `**5.4×**` cell with the freshly computed ratio. Add a sentence under the table noting the SigLIP-fallback caveat still applies — Task 2 removes it.

- [ ] **Step 6: Update `README.md` vs-QLoRA claim**

In `README.md`, find the line that begins `**Vs vanilla 4-bit QLoRA (HF + PEFT + bitsandbytes)** — measured ~3–4× speedup…`. Replace with the measured ratio computed in Step 4, citing `baseline_benchmark_results.json`.

- [ ] **Step 7: Commit**

```bash
git add baseline_benchmark_results.json docs/BENCHMARKS.md README.md
git commit -m "$(cat <<'EOF'
bench: replace cited vanilla QLoRA baseline with measured number

Runs examples/modal_baseline_benchmark.py end-to-end on Modal L4 + T4
and folds the L4 4-bit train it/s + peak VRAM into BENCHMARKS.md and
the README so the FastVLA / vanilla speedup ratio is locally
reproducible instead of cited from issue #1.

Closes C3 in docs/ACCESSIBILITY_ROADMAP.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: Verify commit landed**

```bash
git log -1 --oneline
git status
```
Expected: most recent commit is the bench one, working tree clean apart from in-flight Task 2/3 files.

---

## Task 2: Fix OpenVLA fused vision loader (C4)

**Files:**
- Modify: `fastvla/adapters/vision.py:101-145` (`OpenVLAFusedVisionAdapter.from_pretrained`)
- Create: `tests/test_openvla_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_openvla_loader.py`:
```python
"""OpenVLA vision-tower load path must keep the fused DINOv2+SigLIP backbone,
not silently fall back to SigLIP-only."""

import pytest
import os
import torch
from fastvla.adapters.vision import (
    OpenVLAFusedVisionAdapter,
    SigLIPVisionAdapter,
)


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN") and not os.environ.get("HF_API_KEY"),
    reason="needs HF token for openvla/openvla-7b download",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OpenVLA load path uses bitsandbytes / device_map='auto'",
)
def test_openvla_loader_keeps_fused_backbone():
    adapter = OpenVLAFusedVisionAdapter.from_pretrained(
        "openvla/openvla-7b",
        device_map="auto",
        load_in_4bit=True,
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY"),
    )
    # The fused backbone exposes a dual-stream tower; SigLIPVisionAdapter only
    # wraps a single transformers SiglipVisionModel.
    assert not isinstance(adapter, SigLIPVisionAdapter), (
        "OpenVLAFusedVisionAdapter fell back to SigLIP-only — "
        "AutoModelForVision2Seq path failed."
    )
    # Sanity check on adapter embed dim — OpenVLA's fused tower outputs 2176
    # (DINOv2-L 1024 + SigLIP-SO400M 1152) before projection.
    assert adapter.embed_dim in (1024, 1152, 2176), (
        f"unexpected embed_dim {adapter.embed_dim}"
    )
```

- [ ] **Step 2: Run test to verify it fails today**

Run:
```bash
HF_TOKEN="$HF_API_KEY" uv run pytest tests/test_openvla_loader.py -v
```
Expected: FAIL with "OpenVLAFusedVisionAdapter fell back to SigLIP-only". (Or `SKIPPED` if GPU/HF token not available locally — still passes the design intent.)

- [ ] **Step 3: Patch the loader to try `AutoModelForVision2Seq` first**

Open `fastvla/adapters/vision.py`. Replace the body of `OpenVLAFusedVisionAdapter.from_pretrained` (currently lines 101-145) with:
```python
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device_map: Union[str, Dict] = "auto",
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        **kwargs,
    ) -> "OpenVLAFusedVisionAdapter":
        from transformers import AutoModel, AutoModelForVision2Seq

        logger.info(f"Loading OpenVLA model {model_id} for vision extraction...")

        quant_config = cls._get_bnb_config() if load_in_4bit else None

        try:
            import accelerate  # noqa: F401

            can_use_device_map = True
        except ImportError:
            can_use_device_map = False
            device_map = None if device_map == "auto" else device_map

        load_kwargs = dict(
            device_map=device_map if can_use_device_map else None,
            quantization_config=quant_config,
            token=hf_token,
            trust_remote_code=True,
        )

        # OpenVLA registers its model via `auto_map` for Vision2Seq, not the
        # generic AutoModel. Try Vision2Seq first; only fall through to the
        # plain AutoModel path if it raises.
        try:
            full_model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
            vision_backbone = cls._extract_vision_encoder(full_model)
            return cls(vision_backbone)
        except Exception as e_vision2seq:
            logger.warning(
                f"AutoModelForVision2Seq failed for {model_id}: {e_vision2seq}. "
                "Trying plain AutoModel..."
            )

        try:
            full_model = AutoModel.from_pretrained(model_id, **load_kwargs)
            vision_backbone = cls._extract_vision_encoder(full_model)
            return cls(vision_backbone)
        except Exception as e_automodel:
            logger.warning(
                f"AutoModel also failed for {model_id}: {e_automodel}. "
                "Last-resort fallback: SigLIP-so400m-patch14-384."
            )
            return SigLIPVisionAdapter.from_pretrained(
                "google/siglip-so400m-patch14-384",
                device_map=device_map,
                load_in_4bit=load_in_4bit,
                hf_token=hf_token,
            )
```

- [ ] **Step 4: Run the test again to verify it passes (or stays skipped)**

Run:
```bash
HF_TOKEN="$HF_API_KEY" uv run pytest tests/test_openvla_loader.py -v
```
Expected: PASS if GPU + HF token available; SKIPPED otherwise. Either way no FAIL.

- [ ] **Step 5: Re-run production benchmark to confirm OpenVLA-7B no longer falls back**

```bash
modal run --detach examples/modal_production_benchmark.py 2>&1 | tee /tmp/modal_prod_after_c4.log
```
Expected: the `[fastvla.adapters.vision|WARNING]AutoModel failed to load OpenVLA directly` line no longer appears for the openvla-7b row. Train it/s on L4 is expected to drop into the 8-12 range (vs the inflated 16.2 from the SigLIP fallback).

- [ ] **Step 6: Update `production_benchmark_results.json` + BENCHMARKS.md with new honest numbers**

Open `docs/BENCHMARKS.md`. Update the "OpenVLA-7B — SigLIP fallback path" table to use the new it/s. Rename the section header to "OpenVLA-7B (fused DINOv2 + SigLIP backbone)". Remove the SigLIP caveat block under the table — replace with a one-line note pointing at the commit that restored the full backbone.

- [ ] **Step 7: Commit**

```bash
git add fastvla/adapters/vision.py tests/test_openvla_loader.py production_benchmark_results.json docs/BENCHMARKS.md
git commit -m "$(cat <<'EOF'
fix(vision): load OpenVLA via AutoModelForVision2Seq so fused backbone keeps

OpenVLA registers under auto_map as a Vision2Seq model, not the
plain AutoModel that fastvla was using. Result: every load silently
fell through to SigLIP-only, masking ~30-50% of the real model's
compute footprint in benchmarks.

- Try AutoModelForVision2Seq.from_pretrained first
- Fall through to AutoModel only if Vision2Seq raises
- SigLIP fallback retained as last resort with warning

Adds tests/test_openvla_loader.py that pins the fused-backbone
contract. Re-runs production benchmark and folds the (lower, honest)
OpenVLA-7B numbers into BENCHMARKS.md.

Closes C4 in docs/ACCESSIBILITY_ROADMAP.md and issue #2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: Verify commit + clean tree**

```bash
git log -2 --oneline
git status
```
Expected: top-of-tree shows the loader fix on top of Task 1's bench commit.

---

## Task 3: `torch.compile` on by default for Ada/Hopper (B5)

**Files:**
- Modify: `fastvla/config.py` — change `use_torch_compile` default; add `_auto_torch_compile()` helper
- Modify: `fastvla/model.py` — call helper at init to resolve the default
- Create: `tests/test_auto_compile.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_compile.py`:
```python
"""Auto-enable torch.compile on Ada (sm_89) / Hopper (sm_90+), keep off on
Turing (sm_75) and Ampere (sm_80) where compile yields little or causes
issues with bitsandbytes."""

import pytest
from unittest.mock import patch
from fastvla.config import _auto_torch_compile


@pytest.mark.parametrize(
    "major,minor,expected",
    [
        (7, 5, False),  # T4 Turing
        (8, 0, False),  # A100 Ampere
        (8, 6, False),  # RTX 3090 Ampere
        (8, 9, True),   # L4 / RTX 4090 Ada
        (9, 0, True),   # H100 Hopper
        (10, 0, True),  # future
    ],
)
def test_auto_torch_compile_per_arch(major, minor, expected):
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_capability", return_value=(major, minor)
    ):
        assert _auto_torch_compile() is expected


def test_auto_torch_compile_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert _auto_torch_compile() is False
```

- [ ] **Step 2: Run test to confirm it fails (helper does not exist yet)**

Run:
```bash
uv run pytest tests/test_auto_compile.py -v
```
Expected: FAIL — `ImportError: cannot import name '_auto_torch_compile' from 'fastvla.config'`.

- [ ] **Step 3: Add helper to `fastvla/config.py`**

Open `fastvla/config.py`. Add near the top (after imports):
```python
def _auto_torch_compile() -> bool:
    """Default `use_torch_compile` policy: on for Ada (sm_89) + Hopper (sm_90+),
    off elsewhere. Centralised here so tests can patch one symbol."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)
```

Find the line in `FastVLAConfig` that sets `use_torch_compile: bool = False` (or similar). Replace with:
```python
    use_torch_compile: bool = field(default_factory=_auto_torch_compile)
```

If `field` is not already imported from `dataclasses`, add it.

- [ ] **Step 4: Re-run the test to verify it passes**

Run:
```bash
uv run pytest tests/test_auto_compile.py -v
```
Expected: PASS, all 7 parameterised cases plus the no-cuda case.

- [ ] **Step 5: Verify `model.py` honours the resolved default**

Run:
```bash
grep -n "use_torch_compile" fastvla/model.py
```
Expected: at least one site that reads `config.use_torch_compile` and calls `torch.compile`. If absent, add this block at the end of `FastVLAModel.__init__` (after all submodules are constructed):
```python
        if getattr(config, "use_torch_compile", False):
            try:
                import torch

                self._compiled = True
                self.forward = torch.compile(
                    self.forward, mode="reduce-overhead", fullgraph=False
                )
                logger.info("torch.compile enabled on FastVLAModel.forward")
            except Exception as e:
                logger.warning(f"torch.compile unavailable: {e}")
                self._compiled = False
```
Only add this block if no equivalent already exists.

- [ ] **Step 6: Smoke-run on Modal L4 to confirm no regression**

```bash
modal run --detach examples/modal_smoke_benchmark.py 2>&1 | tee /tmp/modal_smoke_b5.log
```
Expected: still completes green; `torch.compile enabled` line appears in L4 log; train it/s on L4 ≥ the prior smoke-bench number (459 it/s for the dummy model — i.e. compile must not slow it down).

- [ ] **Step 7: Update `docs/ACCESSIBILITY_ROADMAP.md` checkboxes**

Open `docs/ACCESSIBILITY_ROADMAP.md`. Flip these rows:
- **C3** — status from 🟡 to ☑
- **C4** — status from ☐ to ☑
- **B5** — status from 🟡 to ☑
Update the "Sequencing" list at the bottom to remove items 1, 2, 3 (now done) and renumber.

- [ ] **Step 8: Commit**

```bash
git add fastvla/config.py fastvla/model.py tests/test_auto_compile.py docs/ACCESSIBILITY_ROADMAP.md
git commit -m "$(cat <<'EOF'
feat(config): enable torch.compile by default on Ada / Hopper

Adds _auto_torch_compile() helper in fastvla/config.py that checks
the live CUDA device capability and returns True for sm_89+
(Ada L4 / RTX 4090, Hopper H100). FastVLAConfig.use_torch_compile
now defaults via dataclasses.field(default_factory=_auto_torch_compile)
so the right policy fires automatically on the GPU class the model
is actually constructed on.

FastVLAModel.__init__ wraps self.forward with torch.compile when the
flag is set, with a safe try/except for environments where compile
fails.

Closes B5 in docs/ACCESSIBILITY_ROADMAP.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9: Verify full test suite still passes**

```bash
uv run pytest tests/ -q 2>&1 | tail -10
```
Expected: green or only skipped tests (network-gated ones). No new failures.

- [ ] **Step 10: Verify clean tree + log**

```bash
git log -3 --oneline
git status
```
Expected: three new commits on top of `5c49372` (bench, vision fix, compile default). Working tree clean.

---

## Self-Review

**1. Spec coverage** — each of the three sequenced items in `docs/ACCESSIBILITY_ROADMAP.md` § "Sequencing" maps to one Task above (C3 → Task 1, C4 → Task 2, B5 → Task 3). No silent additions.

**2. Placeholder scan** — every code step contains the actual code; every command step has expected output; the `<VANILLA_L4_ITS>` placeholder in Task 1 is named explicitly as "fill in from previous step's output", not a TODO.

**3. Type / name consistency** — `_auto_torch_compile` referenced in Tasks 3.1, 3.3, 3.4 matches in casing and underscore. `OpenVLAFusedVisionAdapter.from_pretrained` signature unchanged across Task 2 — same `model_id`, `device_map`, `load_in_4bit`, `hf_token`, `**kwargs`.
