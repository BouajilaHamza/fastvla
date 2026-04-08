# Comprehensive Fix Plan: FastVLA Training Pipeline

## Errors Found (From Kaggle Runs)

### Error #1: Dtype Mismatch in Action Head ✅ FIXED (commit 7282670)
```
RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::Half != float
```
- **Location:** `fastvla/kernels/action_head.py:52`
- **Cause:** Mixed precision autocast converts activations to FP16, weights stay FP32
- **Status:** FIXED in previous commit
- **Files fixed:** `action_head.py`, `model.py`, `adapters/action_head.py`, `action.py`

---

### Error #2: FP16 Gradient Unscale Failure ❌ NOT FIXED
```
ValueError: Attempting to unscale FP16 gradients.
```
- **Location:** `torch/amp/grad_scaler.py:261` → triggered by `accelerator.clip_grad_norm_()` at `training.py:176`
- **Cause:** 4-bit quantized models produce FP16 gradients. GradScaler expects FP32 gradients and refuses to unscale FP16.
- **When it happens:** Any training run with `load_in_4bit=True` + `use_mixed_precision=True`
- **Call chain:**
  ```
  trainer.train() → train_step() → accelerator.backward(loss) → accelerator.clip_grad_norm_()
  → accelerator.unscale_gradients() → scaler.unscale_(opt) → raises ValueError
  ```

---

## Root Cause Analysis: Why Error #2 Happens

When you load a model with 4-bit quantization (`load_in_4bit=True`):

1. **Model parameters are stored in 4-bit**, computed in FP16
2. **Accelerator** enables mixed precision → wraps forward pass with `autocast(dtype=torch.float16)`
3. **Gradients flow back in FP16** because that's the compute dtype of the 4-bit model
4. **Accelerator's GradScaler** tracks gradients for mixed precision stability
5. **`clip_grad_norm_()` calls `unscale_gradients()`** which tries to divide FP16 gradients by the loss scale
6. **PyTorch's GradScaler** refuses: `"Attempting to unscale FP16 gradients."`

This is a **fundamental incompatibility** between 4-bit quantization and gradient scaling.

---

## Comprehensive Fix Plan

### Fix #1: Disable Gradient Scaling for 4-Bit Models

**File:** `fastvla/training.py`  
**Lines:** 73-85 (Accelerator initialization)  
**Severity:** 🔴 Critical — blocks all 4-bit training with mixed precision

**Current code:**
```python
mixed_precision = "no"
if use_mixed_precision:
    if torch.cuda.is_available():
        mixed_precision = "fp16"
    else:
        mixed_precision = "no"

self.accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision=mixed_precision,
)
```

**Problem:** Doesn't check if model is 4-bit quantized. 4-bit + FP16 mixed precision = GradScaler failure.

**Fix:**
```python
mixed_precision = "no"
if use_mixed_precision:
    if torch.cuda.is_available():
        # Check if model is 4-bit quantized
        is_4bit = getattr(model, "is_loaded_in_4bit", False) or \
                  getattr(model, "hf_quantizer", None) is not None
        
        if is_4bit:
            # 4-bit models compute in FP16 already; gradient scaling is incompatible
            # Use "no" to disable GradScaler (model already benefits from 4-bit quantization)
            mixed_precision = "no"
            print("ℹ️ 4-bit model detected: disabling gradient scaling (GradScaler incompatible with 4-bit)")
        else:
            # Use fp16 for T4 GPUs, bf16 for newer GPUs
            mixed_precision = "fp16"
    else:
        mixed_precision = "no"
```

**Risk:** Low — 4-bit models already benefit from quantization, disabling gradient scaling won't hurt training quality.

---

### Fix #2: Handle clip_grad_norm_ with Unprepared Models

**File:** `fastvla/training.py`  
**Lines:** 174-178  
**Severity:** 🟡 Medium — may cause issues with dispatched models

**Current code:**
```python
if self.accelerator.sync_gradients:
    self.accelerator.clip_grad_norm_(
        self.model.parameters(), self.max_grad_norm
    )
```

**Problem:** When model is NOT prepared by Accelerator (has `hf_device_map`), calling `accelerator.clip_grad_norm_()` still tries to unscale gradients which may not exist.

**Fix:**
```python
if self.accelerator.sync_gradients:
    # Skip gradient clipping for 4-bit models (no gradient scaling active)
    is_4bit = getattr(self.model, "is_loaded_in_4bit", False)
    
    if is_4bit:
        # 4-bit models: clip gradients directly without unscale
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
    else:
        # Standard models: use Accelerator's clip_grad_norm_ (handles unscale)
        self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
```

**Risk:** Low — direct clip_grad_norm_ is safe when no gradient scaling is active.

---

### Fix #3: Validate Model Compatibility at Trainer Init

**File:** `fastvla/training.py`  
**Lines:** After Accelerator initialization (around line 85)  
**Severity:** 🟡 Medium — fails silently until runtime

**Add validation:**
```python
# ── Validate model/precision compatibility ──────────────────────────
is_4bit = getattr(model, "is_loaded_in_4bit", False)
if is_4bit and use_mixed_precision and torch.cuda.is_available():
    print(
        "⚠️ Warning: 4-bit model with mixed precision enabled.\n"
        "   Gradient scaling will be disabled (incompatible with 4-bit quantization).\n"
        "   This is expected behavior — 4-bit quantization already reduces memory.\n"
        "   To suppress this warning, set use_mixed_precision=False."
    )
```

**Risk:** None — informational only.

---

### Fix #4: Ensure labels dtype matches action_preds

**File:** `fastvla/model.py`  
**Lines:** ~385-388 (loss computation)  
**Severity:** 🟠 High — may cause silent dtype mismatch in loss

**Current code:**
```python
if labels is not None:
    labels = labels.to(head_device)
```

**Problem:** Only moves labels to device, doesn't match dtype with action_preds. If action_preds is FP16 (from 4-bit model), loss computation mixes FP16/FP32.

**Fix:**
```python
if labels is not None:
    labels = labels.to(device=head_device, dtype=action_preds.dtype)
```

**Risk:** Low — ensures dtype consistency.

---

### Fix #5: Fix 8-bit Optimizer + 4-bit Model Compatibility

**File:** `fastvla/optimization.py`  
**Lines:** 68-82  
**Severity:** 🟡 Medium — 8-bit optimizer may not work correctly with 4-bit gradients

**Current code:**
```python
if BNB_AVAILABLE and torch.cuda.is_available():
    optimizer = bnb.optim.AdamW8bit(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
```

**Problem:** 8-bit optimizer quantizes optimizer states. Combined with 4-bit model + FP16 gradients, this is triple quantization.

**Fix:**
```python
# Check if model is 4-bit quantized
is_4bit = any(
    hasattr(p, "quant_state") or p.dtype == torch.float16
    for p in model.parameters()
)

if BNB_AVAILABLE and torch.cuda.is_available() and not is_4bit:
    # Use 8-bit optimizer only for non-4-bit models
    optimizer = bnb.optim.AdamW8bit(...)
elif BNB_AVAILABLE and torch.cuda.is_available() and is_4bit:
    # 4-bit models: use standard AdamW (4-bit already provides memory savings)
    print("ℹ️ 4-bit model detected: using standard AdamW (8-bit optimizer skipped)")
    optimizer = torch.optim.AdamW(...)
else:
    optimizer = torch.optim.AdamW(...)
```

**Risk:** Low — 4-bit models don't need 8-bit optimizer; memory savings already achieved.

---

## Proactive Analysis: What Else Could Break

### Potential Issue #6: Multi-Device Model Preparation
**File:** `fastvla/training.py` lines 105-109  
**Scenario:** Model with `device_map="auto"` has parameters on multiple devices  
**Risk:** Accelerator may not handle multi-device models correctly when calling `prepare()`  
**Status:** Already handled by the `hasattr(model, "hf_device_map")` check, but should be documented

### Potential Issue #7: save_pretrained with Dispatched Model
**File:** `fastvla/training.py` lines 212-215  
**Scenario:** Saving a model that was loaded with `device_map="auto"`  
**Risk:** `save_pretrained()` may fail if model is dispatched across devices  
**Fix needed:** Add try/except and fallback to `torch.save(model.state_dict(), ...)`

### Potential Issue #8: Vision Encoder on Different Device
**File:** `fastvla/model.py` lines 305-320  
**Scenario:** Vision encoder loaded on GPU, but pooled output on CPU  
**Risk:** `pooled.to(head_device)` may fail if head_device doesn't match  
**Status:** Already handled by device transfer logic, but dtype fix is critical

---

## Implementation Priority

| Priority | Fix | File | Impact | Complexity |
|----------|-----|------|--------|------------|
| **P0** | Fix #1: Disable gradient scaling for 4-bit | `training.py:73-85` | Blocks all 4-bit training | Low |
| **P0** | Fix #2: Handle clip_grad_norm_ for 4-bit | `training.py:174-178` | Crashes during training | Low |
| **P1** | Fix #4: Labels dtype matching | `model.py:385-388` | Silent loss corruption | Low |
| **P1** | Fix #5: 8-bit optimizer + 4-bit | `optimization.py:68-82` | Triple quantization | Medium |
| **P2** | Fix #3: Validation at init | `training.py:~85` | Better error messages | Trivial |
| **P2** | Fix #7: Checkpoint save for dispatched | `training.py:212-215` | Can't save checkpoints | Low |

---

## Test Suite Coverage

The test suite (`tests/test_e2e_training.py`) covers:

| Test | Catches Error # |
|------|-----------------|
| `test_01_basic_train_step` | Baseline |
| `test_02_mixed_precision_train_step` | Dtype issues |
| `test_03_full_train_loop_two_steps` | End-to-end |
| `test_04_4bit_model_mixed_precision` | **Error #2** (FP16 unscale) |
| `test_05_model_with_device_map_auto` | Device map issues |
| `test_06_gradient_accumulation` | Sync logic |
| `test_07_checkpoint_save_and_restore` | Fix #7 |
| `test_08_evaluation_loop` | Eval path |
| `test_missing_images_raises_error` | Collator validation |
| `test_missing_instructions_uses_fallback` | Fix from previous commit |
| `test_triton_action_head_cpu_fp16_forward` | **Error #1** (already fixed) |
| `test_action_decode_backward_fp16` | Backward path dtype |
| `test_standard_adamw_with_fp16_model` | FP16 model basics |
| `test_8bit_optimizer_with_standard_model` | Fix #5 |

---

## Execution Plan

1. **Apply Fix #1** (training.py mixed precision logic)
2. **Apply Fix #2** (training.py clip_grad_norm_ handling)
3. **Apply Fix #4** (model.py labels dtype)
4. **Apply Fix #5** (optimization.py 8-bit optimizer check)
5. **Apply Fix #3** (validation message)
6. **Apply Fix #7** (checkpoint save fallback)
7. **Run test suite on Kaggle**
8. **Verify no regressions**
9. **Commit all changes**
