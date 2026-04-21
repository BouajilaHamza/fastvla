---
language:
- ar
- en
license: apache-2.0
library_name: lerobot
tags:
- robotics
- vla
- imitation-learning
- arabic
- libero-10
datasets:
- lerobot/libero_10_image
---

# ar-libero-10-image: Arabic-Native LIBERO-10 Dataset for Robotics

This dataset is an **Arabic-localized version** of the standard `lerobot/libero_10_image` dataset. It is specifically designed to enable the training and fine-tuning of **Vision-Language-Action (VLA)** models (like OpenVLA or FastVLA) to perform complex multi-task robotic manipulation using native Arabic instructions.

## Dataset Summary
- **Task:** LIBERO-10 (Long-horizon multi-task manipulation)
- **Localization:** 100% Arabic (Translated using NLLB-200)
- **Robot:** Franka Emika Panda
- **Size:** ~101.4k frames across 379 episodes
- **Tasks:** 10 distinct long-horizon tasks
- **Format:** LeRobot-compatible Parquet
- **Resolution:** 256x256 (Global & Wrist view)

## Localization Process
The dataset was processed using the **FastVLA Arabic Dataset Factory**. 
1. **Metadata-to-Data Mapping:** Since the original `libero_10_image` dataset uses `task_index` to reference instructions, we mapped every frame to its corresponding natural language instruction.
2. **Arabic Translation:** Instructions were translated from English to Arabic using the **NLLB-200-distilled-600M** model.
3. **Explicit Columns:** A literal `instruction` column was added to the dataset rows to ensure seamless integration with the FastVLA training pipeline.

### Task List (Arabic-English Mapping)

| Index | Arabic Instruction | English Original |
| :--- | :--- | :--- |
| **0** | قم بتشغيل الموقد وضع وعاء الموكا عليه. | Turn on the stove and put the moka pot on it. |
| **1** | أغلق الدرج السفلي للخزانة وضع الوعاء الأسود فوقه. | Close the bottom drawer of the cabinet and put the black bowl on top of it. |
| **2** | ضع الخوخ الأصفر في السلة وضع السلة على الرف. | Put the yellow peach in the basket and put the basket on the shelf. |
| **3** | ضع الوعاء الأبيض في وعاء الموكا وضع وعاء الموكا على الموقد. | Put the white bowl in the moka pot and put the moka pot on the stove. |
| **4** | ضع زجاجة النبيذ على رف النبيذ وضع رف النبيذ على الرف. | Put the wine bottle on the wine rack and put the wine rack on the shelf. |
| **5** | ضع حساء الحروف الأبجدية في السلة وحرك السلة إلى الجانب القريب من الطاولة. | Put the alphabet soup in the basket and move the basket to the near side of the table. |
| **6** | ضع الزبدة في السلة وحرك السلة إلى الجانب القريب من الطاولة. | Put the butter in the basket and move the basket to the near side of the table. |
| **7** | ضع الكوب الأبيض على الطبق وضع الطبق على الرف. | Put the white mug on the plate and put the plate on the shelf. |
| **8** | ضع الكأس الزجاجي على الطبق وضع الطبق على الرف. | Put the glass cup on the plate and put the plate on the shelf. |
| **9** | ضع الكوب الأبيض على الكتاب وضع الكتاب على الرف. | Put the white mug on the book and put the book on the shelf. |

## Data Structure
The dataset follows the `LeRobotDataset` schema with the following additions:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `instruction` | string | **Arabic natural language instruction.** |
| `observation.images.image` | image | Global camera view (256x256). |
| `observation.images.wrist_image` | image | Wrist-mounted camera view (256x256). |
| `observation.state` | list | Robot state (Joints/Gripper). |
| `action` | list | Target actions (EE/Gripper). |

## Citation & Credits
This dataset is a derivative work of `lerobot/libero_10_image`. 
- **Original Authors:** Bo Liu, et al. (LIBERO Benchmark)
- **LeRobot Integration:** Hugging Face LeRobot Team
- **Arabic Localization:** FastVLA Project

## License
This dataset is released under the **Apache 2.0 License**, matching the original LIBERO-10 dataset.
