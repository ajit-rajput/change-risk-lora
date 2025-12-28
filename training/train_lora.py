# At a high level, the script does 5 things:
# 	1.	Loads your dataset
# 	2.	Converts each sample into a training prompt
# 	3.	Loads a base LLM
# 	4.	Attaches LoRA adapters
# 	5.	Trains only the LoRA weights

import json
import torch
from datasets import Dataset
	# •	Hugging Face’s Dataset object
	# •	Makes batching, mapping, and tokenization easier
	# •	Much safer than manual lists during training

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
# AutoModelForCausalLM : Loads a text-generation model (GPT-style)
# AutoTokenizer : Converts text → tokens the model understands
# TrainingArguments : Central place for all training knobs
# Trainer : High-level training loop (handles backprop, saving, logging)
# DataCollatorForLanguageModeling : Prepares inputs/labels for causal LM training

from peft import LoraConfig, get_peft_model
# LoraConfig → defines how LoRA behaves
# •get_peft_model → injects LoRA layers into the base model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" ###"mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "dataset/security_change_risk.json"
OUTPUT_DIR = "training/lora_adapter"


def load_dataset(path):
    with open(path, "r") as f:
        raw = json.load(f)

    samples = []

## constructing instruction-tuning style prompts
# Instruction → what to do
# Input       → change description
# Response    → ideal output

    for item in raw:
        prompt = (
            "### Instruction:\n"
            f"{item['instruction']}\n\n"
            "### Input:\n"
            f"{item['input']}\n\n"
            "### Response:\n"
            f"{json.dumps(item['output'], indent=2)}"
        )
        samples.append({"text": prompt})

    return Dataset.from_list(samples)


def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure tokenizer has a pad token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )

    # 	r=8 -- Rank of low-rank matrices (capacity of adaptation)
    # 	lora_alpha=32 -- Scaling factor (controls impact strength)
    # 	target_modules -- Only apply LoRA to attention layers (efficient + effective)
    # 	lora_dropout=0.05 -- Prevents overfitting on small dataset
    # 	bias=“none” -- Don’t add extra bias parameters
    # 	task_type=“CAUSAL_LM” -- Specifies model type for correct adaptation
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(DATASET_PATH)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

	# •	batch_size=1 → minimal memory
	# •	grad_accumulation=4 → effective batch size = 4
	# •	learning_rate=2e-4 → typical for LoRA
	# •	3 epochs → small dataset, avoid overfitting
	# •	fp16=False → CPU stability
	# •	no logging services → offline PoC

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        report_to="none"
    )

    # Trainer handles:
    # 	Forward pass
    # 	Backprop
    # 	Optimizer
    # 	Checkpoint saving

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    ### Train and save the LoRA-adapted model. Updates only LoRA weights
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()