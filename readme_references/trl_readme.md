TRL - TRANSFORMERS REINFORCEMENT LEARNING
A COMPREHENSIVE LIBRARY TO POST-TRAIN FOUNDATION MODELS

🎉 WHAT'S NEW
TRL v1: A major milestone marking a shift in what TRL is.

OVERVIEW
TRL is a cutting-edge library designed for post-training foundation models using advanced techniques like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Built on top of the 🤗 Transformers ecosystem.

HIGHLIGHTS
* Trainers: SFTTrainer, GRPOTrainer, DPOTrainer, RewardTrainer.
* Efficient and scalable: Leverages Accelerate and PEFT.
* CLI: Fine-tune models without writing code.

INSTALLATION
pip install trl

QUICK START
SFTTRAINER:
from trl import SFTTrainer
from datasets import load_dataset
dataset = load_dataset("trl-lib/Capybara", split="train")
trainer = SFTTrainer(model="Qwen/Qwen2.5-0.5B", train_dataset=dataset)
trainer.train()

GRPOTRAINER:
(Used for Deepseek R1 style training).
from trl import GRPOTrainer
trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=accuracy_reward, train_dataset=dataset)

CLI USAGE:
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name trl-lib/Capybara --output_dir Qwen2.5-0.5B-SFT

CITATION
@software{vonwerra2020trl,
  title = {{TRL: Transformers Reinforcement Learning}},
  author = {von Werra, Leandro et al.},
  year = {2020}
}
