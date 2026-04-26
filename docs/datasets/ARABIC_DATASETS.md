# Arabic-Native Datasets for Robotics

This document provides details on the Arabic-localized datasets used for training and fine-tuning FastVLA. These datasets enable Vision-Language-Action (VLA) models to perform robotic manipulation using native Arabic instructions.

---

## 1. ar-pusht-image: Arabic-Native PushT Dataset

This is an Arabic-localized version of the standard `lerobot/pusht_image` dataset.

### Dataset Summary
- **Task:** Push-T (2D Robotic manipulation)
- **Localization:** 100% Arabic (Translated using NLLB-200)
- **Size:** ~48.3k frames across 206 episodes
- **Format:** LeRobot-compatible Parquet
- **Resolution:** 96x96 (Top-down view)

### Sample Instruction
- **English:** "push the block to the goal"
- **Arabic:** "إدفع الكتلة إلى الهدف"

### Data Structure
| Feature | Type | Description |
| :--- | :--- | :--- |
| `instruction` | string | **Arabic natural language instruction.** |
| `observation.image` | image | Top-down camera observation. |
| `observation.state` | list | 2D position of the robot end-effector `[x, y]`. |
| `action` | list | 2D target position for the end-effector `[x, y]`. |

---

## 2. ar-libero-10-image: Arabic-Native LIBERO-10 Dataset

This is an Arabic-localized version of the standard `lerobot/libero_10_image` dataset.

### Dataset Summary
- **Task:** LIBERO-10 (Long-horizon multi-task manipulation)
- **Localization:** 100% Arabic (Translated using NLLB-200)
- **Robot:** Franka Emika Panda
- **Size:** ~101.4k frames across 379 episodes
- **Format:** LeRobot-compatible Parquet
- **Resolution:** 256x256 (Global & Wrist view)

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

---

## Localization Process
All datasets were processed using the **FastVLA Arabic Dataset Factory**:
1. **Instruction Mapping:** English instructions were extracted or mapped from task indices.
2. **Translation:** Translated to Arabic using the **NLLB-200-distilled-600M** model.
3. **Column Injection:** A literal `instruction` column was added to ensure direct compatibility with multi-modal training pipelines.

## Citation & Credits
These datasets are derivative works of the original PushT and LIBERO-10 datasets, integrated into the LeRobot format by the Hugging Face team and localized by the **FastVLA Project**.
