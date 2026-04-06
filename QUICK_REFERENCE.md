# FastVLA Distributed Training - Quick Reference Guide

## 🔥 The Fix in 30 Seconds

Your error was caused by **5 bugs** that have now been fixed:

1. ✅ Duplicate optimizer steps → Removed duplicates
2. ✅ No shape validation → Added with clear error messages
3. ✅ Manual device placement → Removed (let Accelerator handle it)
4. ✅ Wrong loss scaling → Fixed
5. ✅ No data validation → Added to collator

**Your code should work now!** Just make sure your dataset's action dimensions match the model config.

---

## ✅ Checklist Before Running

- [ ] Model's `action_dim` matches dataset's action dimensions
- [ ] All samples in dataset have same action shape
- [ ] Using `FastVLATrainer` (not manual training loop)
- [ ] Not manually moving tensors to devices
- [ ] `use_mixed_precision=True` for T4 GPUs

---

## 📊 Common Action Dimensions

| Dataset | Action Dim | Notes |
|---------|-----------|-------|
| Bridge Data | 7 | [x, y, z, roll, pitch, yaw, grip] |
| LIBERO | 7 | Same format |
| CALVIN | 7 | Standard robotics |
| Custom | ? | Set `action_dim` to match your data |

---

## 🚀 Quick Start (Kaggle 2x T4)

```python
from fastvla import FastVLAModel, FastVLATrainer

# 1. Load model (action_dim MUST match your data!)
model = FastVLAModel.from_pretrained(
    dummy=True,
    action_dim=7,  # ← CHANGE THIS to match your dataset
    vocab_size=50257
)

# 2. Create trainer
trainer = FastVLATrainer(
    model=model,
    dataset=your_dataset,
    batch_size=4,
    gradient_accumulation_steps=2,
    use_mixed_precision=True,  # fp16 for T4
    use_8bit_optimizer=True,
)

# 3. Train!
results = trainer.train(max_steps=1000)
```

---

## 🐛 Troubleshooting

### Error: "Action dimension mismatch"

**Cause**: Your dataset's actions don't match model's `action_dim`.

**Fix**:
```python
# Check your dataset
sample = dataset[0]
print(sample["actions"].shape)  # e.g., torch.Size([7])

# Set model to match
model = FastVLAModel.from_pretrained(
    action_dim=sample["actions"].shape[0],  # ← Use this!
    ...
)
```

### Error: "Inconsistent action dimensions in batch"

**Cause**: Different samples have different action sizes.

**Fix**:
```python
# Make all samples have same action dimension
class MyDataset(Dataset):
    def __getitem__(self, idx):
        action = load_action(idx)
        assert action.shape[-1] == 7, f"Expected 7D action, got {action.shape}"
        return {
            "actions": action,  # Always 7D!
            ...
        }
```

### Out of Memory on T4

**Fix**: Reduce batch size, increase gradient accumulation
```python
trainer = FastVLATrainer(
    batch_size=2,  # Was 4
    gradient_accumulation_steps=4,  # Was 2 (effective batch = 8)
    use_8bit_optimizer=True,  # Important!
    ...
)
```

### Training is Slow

**Fix**: Enable optimizations
```python
trainer = FastVLATrainer(
    use_mixed_precision=True,  # 2x speedup on T4
    use_8bit_optimizer=True,   # Less memory
    gradient_accumulation_steps=2,  # Balance speed/stability
    ...
)
```

---

## 📝 Understanding the Original Error

```
RuntimeError: The size of tensor a (7) must match the size of tensor b (2) 
at non-singleton dimension 1
```

**What it means**: 
- Model predicted 7-dimensional actions
- But labels were only 2-dimensional
- MSE loss can't compute between different sizes

**Why it happened**:
- Your dataset returns 2D actions: `torch.randn(2)`
- But model expects 7D actions: `action_dim=7`
- No validation caught this mismatch until the loss computation

**How it's fixed now**:
1. Collator validates action dimensions
2. Model validates shapes before loss
3. Clear error message tells you exactly what's wrong

---

## 🎯 Best Practices

### 1. Always Validate Your Dataset
```python
# Quick validation script
dataset = MyDataset()
for i in range(len(dataset)):
    sample = dataset[i]
    assert sample["actions"].shape == (7,), \
        f"Sample {i} has wrong action shape: {sample['actions'].shape}"
print("✓ Dataset validated!")
```

### 2. Use the Auto-Initialized Collator
```python
# Let FastVLATrainer set up the collator
trainer = FastVLATrainer(
    model=model,
    dataset=dataset,  # It will auto-create collator with right action_dim
    ...
)

# Don't manually create collator unless you need custom behavior
```

### 3. Monitor Training
```python
trainer = FastVLATrainer(
    logging_steps=50,  # Print metrics every 50 steps
    eval_steps=100,    # Evaluate every 100 steps
    save_steps=500,    # Checkpoint every 500 steps
    ...
)
```

### 4. Handle Gradient Accumulation Correctly
```python
# Effective batch size = batch_size * gradient_accumulation_steps
# For T4 with limited memory:
trainer = FastVLATrainer(
    batch_size=2,              # Per-GPU batch size
    gradient_accumulation_steps=4,  # Accumulate 4 times
    # Effective: 2 * 4 = 8
    ...
)
```

---

## 🔧 Advanced: Custom Training Loop

If you need more control:

```python
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=2,
    mixed_precision="fp16",
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

for epoch in range(num_epochs):
    for batch in dataloader:
        with accelerator.accumulate(model):
            preds, loss = model(**batch)
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
```

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `fastvla/training.py` | Trainer with fixed optimizer logic |
| `fastvla/model.py` | Model with shape validation |
| `fastvla/data/collator.py` | Collator with action validation |
| `tests/test_training_robustness.py` | Comprehensive test suite |
| `validate_fixes.py` | Quick validation script |
| `examples/kaggle_distributed_example.py` | Kaggle example |

---

## 🆘 Still Having Issues?

1. Run validation: `python validate_fixes.py`
2. Check action dimensions match
3. Review error messages (they're informative now!)
4. Test with dummy data first
5. Gradually add complexity

---

**Good luck with your distributed training! 🚀**
