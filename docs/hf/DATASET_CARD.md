---
language:
- ar
- en
multilinguality:
- bilingual
size_categories:
- 10K<n<100K
tags:
- robotics
- vla
- physical-ai
- arabic-robotics
- libero
- pusht
- flow-matching
license: apache-2.0
---

# Arabic Robotics Manipulation Datasets: Push-T & LIBERO-10

This dataset collection provides verified Modern Standard Arabic (MSA) instruction ground-truth annotations and continuous trajectory demonstrations for physical manipulation tasks, enabling bilingual and regional training for Vision-Language-Action (VLA) foundation models.

## Dataset Structure

1. **`hamzabouajila/ar-pusht-image`**:
   - Continuous 2D planar pushing task.
   - 205 demonstration episodes, 25,650 frames.
   - Verified command: `"ادفع الكتلة إلى الهدف"` (`push the block to the goal`).
   - Observations: 96x96 RGB camera views with 2-DoF continuous end-effector actions.

2. **`hamzabouajila/ar-libero-10-image`**:
   - 10 multi-stage compositional tabletop manipulation tasks for 7-DoF Franka Panda robots.
   - Multi-camera views: `agentview` and `wristview` (eye-in-hand).
   - Tasks include:
     - Stove & Moka Pot: `"شغّل الموقد وضع إبريق الموكا عليه"`
     - Cabinet & Bowl: `"ضع الوعاء الأسود في الدرج السفلي من الخزانة وأغلقه"`
     - Microwave & Mug: `"ضع الكوب الأصفر والأبيض في الميكروويف وأغلقه"`
     - Plate & Pudding: `"ضع الكوب الأبيض على الطبق وضع حلوى الشوكولاتة على يمين الطبق"`

## Standardization & Verification Process

- **Orthographic Hygiene:** Hand-verified Modern Standard Arabic grammar (فصحى معيارية سليمة), eliminating bidirectional rendering glitches, random machine-translation artifacts, and broken encoding.
- **Dialectal Benchmarking:** Cross-referenced against regional Arabic dialects (Egyptian, Levantine, Gulf) with synonyms and conversational command variations cataloged in `scripts/dataset/translate_arabic.py`.

## Loading with LeRobot or HuggingFace Datasets

```python
from datasets import load_dataset

# Load Arabic Push-T
ds = load_dataset("hamzabouajila/ar-pusht-image", split="train")
sample = ds[0]
print("Arabic Instruction:", sample["instruction"])
print("Action Coordinates:", sample["action"])
```
