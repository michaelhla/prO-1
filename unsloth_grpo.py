import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, prepare_model_for_kbit_training
import re
import json
import os
import random
import wandb
from dotenv import load_dotenv
from datasets import Dataset
import time
from unsloth import FastLanguageModel, is_bfloat16_supported
from accelerate import PartialState

from stability_reward import StabilityRewardCalculator

NUM_EPOCHS = 2
MAX_LENGTH = 128000

# ... existing code for construct_prompt, validate_and_construct_prompt ...

print("\nLoading and processing BRENDA data...")
# ... existing data loading code ...

print("\nInitializing wandb...")
# ... existing wandb init code ...

# Get the device for this process
device_string = PartialState().process_index

# Initialize model with unsloth
print("Loading base model...")
model_load_start = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-70b-bnb-4bit",
    max_seq_length=MAX_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
    device_map={'': device_string},
)
print(f"Base model loaded in {time.time() - model_load_start:.2f} seconds")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407
)

# Enable gradient computation for LoRA parameters
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

# ... existing stability calculator initialization code ...

# Create training arguments
training_args = GRPOConfig(
    output_dir="./unsloth_grpo_output",
    run_name="unsloth_grpo_training_run",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1.41e-5,
    logging_steps=1,
    num_generations=2,
    max_prompt_length=512,
    max_completion_length=512,
    temperature=0.7,
    beta=0.04,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
)

# ... existing WandBLoggingCallback class ...

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=stability_reward_func,
    processing_class=tokenizer,
    callbacks=[WandBLoggingCallback()], 
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("./unsloth_grpo_output/final_model")

# ... existing code for saving stability cache and closing wandb ... 