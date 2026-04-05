# Quick Testing Guide

## Prerequisites

First, make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

## Option 1: Simple Test (Recommended First)

Run the simple test script that doesn't require actual models:

```bash
python test_simple.py
```

This will test:
- ✅ Module imports
- ✅ Configuration creation  
- ✅ Custom kernels (fusion, multi-cam)
- ✅ Optimization utilities
- ✅ Data collator
- ✅ Benchmarking utilities
- ✅ Training utilities

## Option 2: Unit Tests with Pytest

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_kernels.py -v
pytest tests/test_model.py -v

# With coverage
pytest tests/ -v --cov=fastvla --cov-report=term
```

## Option 3: Manual Testing

Test individual components:

### Test Kernels

```python
import torch
from fastvla.kernels import vision_language_fusion_forward

device = "cuda" if torch.cuda.is_available() else "cpu"
visual = torch.randn(2, 32, 64, device=device)
text = torch.randn(2, 32, 64, device=device)
fused = vision_language_fusion_forward(visual, text)
print(f"✅ Fusion kernel works! Shape: {fused.shape}")
```

### Test Configuration

```python
from fastvla import FastVLAConfig

config = FastVLAConfig(
    action_dim=7,
    load_in_4bit=False,
)
print(f"✅ Config works! Action dim: {config.action_dim}")
```

### Test Optimization Utils

```python
from fastvla import get_quantization_config, get_peft_config

qconfig = get_quantization_config(load_in_4bit=True)
peft_config = get_peft_config(r=16, lora_alpha=32)
print(f"✅ Optimization utils work!")
```

## Troubleshooting

### Import Errors

If you get import errors:
1. Make sure dependencies are installed: `pip install -r requirements.txt`
2. Check that you're in the correct directory
3. Try: `pip install transformers torch triton`

### CUDA Not Available

Tests will automatically fall back to CPU if CUDA is not available. Some tests may be slower but will still work.

### Model Download Issues

The simple test (`test_simple.py`) doesn't require actual models. For full model tests, you may need to:
- Have internet connection for model downloads
- Use smaller models (TinyLlama) for testing
- Set `load_in_4bit=False` to avoid quantization issues

## Expected Output

When running `python test_simple.py`, you should see:

```
============================================================
FastVLA Simple Test Suite
============================================================
Testing imports...
✅ All imports successful

Testing configuration...
✅ Configuration test passed

Testing kernels...
✅ Fusion kernel test passed
✅ Multi-cam kernel test passed

Testing optimization utilities...
✅ Quantization config test passed
✅ PEFT config test passed
✅ Memory estimation test passed

Testing data collator...
✅ Collator test passed

Testing benchmarking utilities...
✅ Benchmarking test passed

Testing training utilities...
✅ Training utilities test passed

============================================================
Test Summary
============================================================
Imports              ✅ PASSED
Configuration        ✅ PASSED
Kernels              ✅ PASSED
Optimization Utils   ✅ PASSED
Collator             ✅ PASSED
Benchmarking         ✅ PASSED
Training Utils       ✅ PASSED

Total: 7/7 tests passed

🎉 All tests passed!
```

## Next Steps

After tests pass:
1. Try the example scripts in `examples/`
2. Run benchmarks: `python examples/benchmark_example.py`
3. Train a model: `python examples/train_example.py`

For more detailed testing instructions, see [TESTING.md](TESTING.md).

