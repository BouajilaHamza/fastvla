# Mixed Precision & Dtype Compatibility Fixes

## Summary

Fixed **6 critical bugs** that caused dtype mismatches during mixed precision training with the `Accelerate` library. These bugs manifested as:

```
RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::Half != float
```

## Root Cause

When `Accelerate` enables mixed precision training (`fp16`), it wraps the forward pass with `torch.autocast`, which automatically converts activations to `float16`. However, model weights remain in `float32`. The original code didn't handle this dtype mismatch, causing failures at matrix multiplication operations.

---

## Fixes Applied

### 1. `fastvla/kernels/action_head.py` (Line 53)
**Issue:** CPU fallback path in `TritonActionHead.forward()` didn't convert input dtype  
**Fix:** Added `x = x.to(self.weight1.dtype)` before matrix operations

```python
# BEFORE (BROKEN):
h = torch.nn.functional.relu(x @ self.weight1 + self.bias1)

# AFTER (FIXED):
x = x.to(self.weight1.dtype)
h = torch.nn.functional.relu(x @ self.weight1 + self.bias1)
```

### 2. `fastvla/model.py` (Line 381)
**Issue:** Only moved pooled activations to action head's device, didn't convert dtype  
**Fix:** Added dtype conversion alongside device transfer

```python
# BEFORE (BROKEN):
head_device = next(self.action_head.parameters()).device
action_preds = self.action_head(pooled.to(head_device))

# AFTER (FIXED):
head_device = next(self.action_head.parameters()).device
head_dtype = next(self.action_head.parameters()).dtype
action_preds = self.action_head(pooled.to(device=head_device, dtype=head_dtype))
```

### 3. `fastvla/adapters/action_head.py` - DiscreteActionHead (Line 59)
**Issue:** No dtype handling in `forward()` method  
**Fix:** Added dtype normalization at start of forward pass

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Ensure input dtype matches layer weights dtype for mixed precision compatibility
    hidden_states = hidden_states.to(self.fc1.weight.dtype)
    h = F.relu(self.fc1(hidden_states))
    ...
```

### 4. `fastvla/adapters/action_head.py` - ContinuousActionHead (Line 107)
**Issue:** Standard PyTorch path didn't handle dtype  
**Fix:** Added dtype normalization

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.triton_head is not None:
        return self.triton_head(hidden_states)
    
    # Ensure input dtype matches layer weights dtype for mixed precision compatibility
    hidden_states = hidden_states.to(self.fc1.weight.dtype)
    h = F.relu(self.fc1(hidden_states))
    return torch.tanh(self.fc2(h))
```

### 5. `fastvla/adapters/action_head.py` - FlowMatchingActionHead (Line 133)
**Issue:** Same dtype handling gap  
**Fix:** Added dtype normalization (same pattern as above)

### 6. `fastvla/kernels/action.py` - action_decode_backward (Line 166)
**Issue:** Backward pass didn't normalize dtypes, could fail under autocast  
**Fix:** Added dtype normalization at start of backward function

```python
def action_decode_backward(grad_output, hidden, weight1, bias1, weight2, bias2):
    # Normalize dtypes for all inputs to prevent mixed precision issues
    dtype = hidden.dtype
    if weight1.dtype != dtype: weight1 = weight1.to(dtype)
    if bias1.dtype != dtype: bias1 = bias1.to(dtype)
    if weight2.dtype != dtype: weight2 = weight2.to(dtype)
    if bias2.dtype != dtype: bias2 = bias2.to(dtype)
    ...
```

### 7. `fastvla/data/collator.py` (Lines 104-127)
**Issue:** Missing `instructions` field caused missing `input_ids` → model crash  
**Fix:** Added graceful fallback for missing text data and validation

```python
else:
    # Provide default empty text input if instructions are missing
    if hasattr(self.tokenizer, 'pad_token_id'):
        batch["input_ids"] = torch.full(
            (len(features), 1),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long
        )
    else:
        batch["input_ids"] = torch.zeros((len(features), 1), dtype=torch.long)
    batch["attention_mask"] = torch.ones((len(features), 1), dtype=torch.long)

# Validate required keys exist
required_keys = ["pixel_values", "input_ids", "labels"]
missing_keys = [k for k in required_keys if k not in batch]
if missing_keys:
    raise ValueError(f"Batch is missing required keys: {missing_keys}. ...")
```

---

## Files Changed

| File | Lines Changed | Severity |
|------|--------------|----------|
| `fastvla/kernels/action_head.py` | 53 | 🔴 Critical |
| `fastvla/model.py` | 378-381 | 🔴 Critical |
| `fastvla/adapters/action_head.py` | 59, 107, 133 | 🔴 Critical |
| `fastvla/kernels/action.py` | 166-172 | 🟡 High |
| `fastvla/data/collator.py` | 104-127 | 🟡 High |

---

## Test Suite

Created comprehensive test suite in:
- `tests/test_dtype_compatibility.py` - pytest format
- `tests/run_dtype_tests_standalone.py` - standalone runner

Tests cover:
- ✅ CPU float32 forward/backward
- ✅ CPU float16 forward/backward (mixed precision simulation)
- ✅ All action head variants (Triton, Discrete, Continuous, FlowMatching)
- ✅ Vision-language fusion with mixed dtypes
- ✅ Data collator with missing fields
- ✅ Dtype consistency between float16/float32 outputs

---

## Installation Instructions (Kaggle/Fresh Environment)

### Option 1: From Local Source
```bash
# In your Kaggle notebook cell or terminal
import subprocess
import sys

# Force reinstall with no cache
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "/path/to/FastVLA",
    "--force-reinstall", 
    "--no-cache-dir",
    "--no-deps"
])

# Clear module cache
for key in list(sys.modules.keys()):
    if 'fastvla' in key:
        del sys.modules[key]

# Now import fresh
from fastvla.model import FastVLAModel
```

### Option 2: Editable Install (Recommended for Development)
```bash
pip install -e /path/to/FastVLA --no-cache-dir
```

With editable install, code changes take effect immediately without reinstall.

### Option 3: From Git Repository
```bash
pip install git+https://github.com/YourOrg/FastVLA.git --no-cache-dir --force-reinstall
```

---

## Verification

After installation, verify the fixes are present:

```python
# Check action_head.py has the dtype fix
import inspect
from fastvla.kernels.action_head import TritonActionHead

source = inspect.getsource(TritonActionHead.forward)
assert "x.to(self.weight1.dtype)" in source, "Missing dtype fix!"
print("✅ Fix verified: action_head dtype handling present")

# Check model.py has the dtype fix
from fastvla.model import FastVLAModel
source = inspect.getsource(FastVLAModel.forward)
assert "dtype=head_dtype" in source, "Missing model dtype fix!"
print("✅ Fix verified: model dtype handling present")
```

---

## Why This Happened

Mixed precision training with `Accelerate`:
1. Wraps forward pass with `torch.autocast(dtype=torch.float16)`
2. Activations become `float16` automatically
3. Model weights stay in `float32`
4. Matrix operations (`@`, `torch.mm`, etc.) require **matching dtypes**
5. Original code didn't handle this mismatch → `RuntimeError`

The fix ensures all linear operations receive inputs with matching dtypes by converting the input to match the weight dtype before computation.

---

## Notes

- The GPU Triton kernel path (`ActionDecodeFunction.apply`) already handles dtypes correctly in `action_decode_forward()` (lines 108-112 normalize dtypes)
- The Triton kernel's backward pass now also normalizes dtypes (fix #6)
- All fixes are **backward compatible** - they don't change behavior for matched dtypes, only prevent crashes on mismatches
- Performance impact is **negligible** - `x.to(dtype)` is a no-op if dtypes already match
