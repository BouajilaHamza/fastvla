🤗 PEFT
STATE-OF-THE-ART PARAMETER-EFFICIENT FINE-TUNING (PEFT) METHODS

Fine-tuning large pretrained models is often prohibitively costly due to their scale. Parameter-Efficient Fine-Tuning (PEFT) methods enable efficient adaptation of large pretrained models to various downstream applications by only fine-tuning a small number of (extra) model parameters instead of all the model's parameters. This significantly decreases the computational and storage costs.

PEFT is integrated with Transformers, Diffusers, and Accelerate.

QUICKSTART
pip install peft

from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
peft_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.save_pretrained("qwen2.5-3b-lora")

WHY YOU SHOULD USE PEFT
* High performance on consumer hardware (e.g., training 12B models on 80GB GPUs).
* Quantization support (QLoRA).
* Save compute and storage (checkpoints are often only a few MBs).

INTEGRATIONS
* Diffusers: Reduces memory for diffusion model training.
* Transformers: Direct integration via `add_adapter`, `load_adapter`, and `set_adapter`.
* Accelerate: Works out of the box for distributed training.
* TRL: Applied to RLHF components.

CITING 🤗 PEFT
@Misc{peft,
  title = {{PEFT}: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author = {Sourab Mangrulkar et al.},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year = {2022}
}
