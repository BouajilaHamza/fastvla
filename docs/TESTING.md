# Testing Guide for FastVLA

This guide explains how to test the FastVLA implementation.

## Quick Test (No Models Required)

Run the simple test script that tests core functionality without requiring actual models:

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

## Running Unit Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test kernels only
pytest tests/test_kernels.py -v

# Test model integration
pytest tests/test_model.py -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=fastvla --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

## Testing Kernels

The kernels can be tested individually:

```python
import torch
from fastvla.kernels import vision_language_fusion_forward, multi_cam_pack_forward

# Test fusion kernel
device = "cuda" if torch.cuda.is_available() else "cpu"
visual = torch.randn(2, 32, 64, device=device)
text = torch.randn(2, 32, 64, device=device)
fused = vision_language_fusion_forward(visual, text)
print(f"Fused shape: {fused.shape}")  # Should be (2, 32, 64)

# Test multi-cam kernel
cams = torch.randn(2, 3, 3, 224, 224, device=device)
packed = multi_cam_pack_forward(cams)
print(f"Packed shape: {packed.shape}")  # Should be (2, 9, 224, 224)
```

## Testing Model (Without Loading Actual Models)

For testing without downloading large models, you can use the test fixtures:

```python
import pytest
from fastvla import FastVLAModel, FastVLAConfig

# Use pytest fixtures
def test_model(test_config, test_batch):
    model = FastVLAModel(test_config)
    outputs = model(**test_batch)
    action_preds, loss = outputs
    assert action_preds.shape == (2, 7)
    assert loss is not None
```

## Testing with Actual Models

If you want to test with actual models (requires GPU and model downloads):

```python
from fastvla import FastVLAModel

# Load model
model = FastVLAModel.from_pretrained(
    vision_encoder_name="google/vit-base-patch16-224",
    llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_4bit=False,  # Set to True for quantization
    gradient_checkpointing=True,
    use_peft=True,
)

# Create test batch
import torch
batch = {
    "pixel_values": torch.randn(2, 3, 3, 224, 224),  # 2 samples, 3 cameras
    "input_ids": torch.randint(0, 1000, (2, 32)),
    "attention_mask": torch.ones(2, 32),
    "labels": torch.randn(2, 7),
}

# Forward pass
action_preds, loss = model(**batch)
print(f"Action predictions shape: {action_preds.shape}")
print(f"Loss: {loss.item()}")
```

## Testing Training Loop

Test the training loop with dummy data:

```python
from fastvla import FastVLATrainer
from torch.utils.data import DataLoader, TensorDataset
import torch

# Create dummy model
class DummyModel(torch.nn.Module):
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        preds = torch.randn(pixel_values.size(0), 7)
        loss = torch.nn.functional.mse_loss(preds, labels) if labels is not None else None
        return preds, loss

model = DummyModel()

# Create dummy dataset
dataset = TensorDataset(
    torch.randn(10, 3, 3, 224, 224),
    torch.randint(0, 1000, (10, 32)),
    torch.ones(10, 32),
    torch.randn(10, 7),
)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b[0] for b in batch]),
        "input_ids": torch.stack([b[1] for b in batch]),
        "attention_mask": torch.stack([b[2] for b in batch]),
        "labels": torch.stack([b[3] for b in batch]),
    }

train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Create trainer
trainer = FastVLATrainer(
    model=model,
    train_dataloader=train_loader,
    use_8bit_optimizer=False,
    use_mixed_precision=False,
    device="cpu",
)

# Train for a few steps
trainer.train(num_epochs=1, max_steps=5)
```

## Testing Benchmarking

Test the benchmarking utilities:

```python
from fastvla import PerformanceProfiler
import torch

profiler = PerformanceProfiler(device="cpu")

# Profile an operation
with profiler.profile("matrix_multiply"):
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = x @ y

# Get summary
summary = profiler.get_summary()
print(f"Total time: {summary['total_time_sec']:.4f} seconds")
print(f"Max memory: {summary['max_memory_gb']:.4f} GB")
```

## Testing Optimization Utilities

Test optimization utilities:

```python
from fastvla import get_quantization_config, get_8bit_optimizer, get_peft_config
import torch.nn as nn

# Test quantization config
qconfig = get_quantization_config(load_in_4bit=True)
print(f"Quantization config: {qconfig}")

# Test PEFT config
peft_config = get_peft_config(r=16, lora_alpha=32)
print(f"PEFT config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

# Test 8-bit optimizer (requires model with parameters)
model = nn.Linear(10, 7)
optimizer = get_8bit_optimizer(model, learning_rate=1e-4)
print(f"Optimizer type: {type(optimizer)}")
```

## Continuous Integration

For CI/CD, you can run tests in a headless environment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=fastvla --cov-report=term

# Run simple test
python test_simple.py
```

## Troubleshooting

### CUDA Not Available

If CUDA is not available, tests will fall back to CPU. Some kernel tests may be skipped or run slower.

### Model Download Issues

If you encounter model download issues, use smaller models or set `load_in_4bit=False` and `use_peft=False` for testing.

### Memory Issues

If you run out of memory during testing:
- Use smaller batch sizes
- Disable quantization (`load_in_4bit=False`)
- Use CPU instead of GPU
- Reduce model size (use TinyLlama instead of Llama-2-7b)

## Next Steps

After running tests successfully:
1. Try the example scripts in `examples/`
2. Run benchmarks with `examples/benchmark_example.py`
3. Train a model with `examples/train_example.py`

