---
language:
- ar
- en
license: mit
library_name: lerobot
tags:
- robotics
- vla
- imitation-learning
- arabic
- pusht
datasets:
- lerobot/pusht_image
---

# ar-pusht-image: Arabic-Native PushT Dataset for Robotics

This dataset is an **Arabic-localized version** of the standard `lerobot/pusht_image` dataset. It is specifically designed to enable the training and fine-tuning of **Vision-Language-Action (VLA)** models (like OpenVLA or FastVLA) using native Arabic instructions.

## Dataset Summary
- **Task:** Push-T (Robotic manipulation)
- **Localization:** 100% Arabic (Translated using NLLB-200)
- **Size:** ~48.3k frames across 206 episodes
- **Format:** LeRobot-compatible Parquet
- **Resolution:** 96x96 (Top-down view)

## Localization Process
The dataset was processed using the **FastVLA Arabic Dataset Factory**. 
1. **Instruction Mapping:** The implicit task instructions from the Push-T suite were extracted.
2. **Translation:** Instructions were translated from English to Arabic using the **NLLB-200-distilled-600M** model, known for its high-quality translation in low-resource and technical domains.
3. **Column Injection:** A literal `instruction` column was added to every row in the dataset to ensure direct compatibility with multi-modal training pipelines.

**Sample Instruction:**
- **English:** "push the block to the goal"
- **Arabic:** "إدفع الكتلة إلى الهدف"

## Data Structure
The dataset follows the `LeRobotDataset` schema with the following additions:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `instruction` | string | **Arabic natural language instruction.** |
| `observation.image` | image | Top-down camera observation. |
| `observation.state` | list | 2D position of the robot end-effector `[x, y]`. |
| `action` | list | 2D target position for the end-effector `[x, y]`. |

## Usage
You can load this dataset directly using the Hugging Face `datasets` library or the `lerobot` library.

### Using `datasets`
```python
from datasets import load_dataset

ds = load_dataset("hamzabouajila/ar-pusht-image", split="train")
print(ds[0]["instruction"]) # "إدفع الكتلة إلى الهدف"
```

### Using `LeRobot`
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("hamzabouajila/ar-pusht-image")
print(dataset[0]["instruction"])
```

## Citation & Credits
This dataset is a derivative work of `lerobot/pusht_image`. 
- **Original Authors:** Diffusion Policy authors (Chi et al.)
- **LeRobot Integration:** Hugging Face LeRobot Team
- **Arabic Localization:** FastVLA Project

## License
This dataset is released under the **MIT License**, matching the original PushT dataset.
