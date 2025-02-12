import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import json
from typing import Dict, List
import wandb
import os
from dotenv import load_dotenv
import pynvml
from torch.cuda import memory_summary
from unsloth import FastLanguageModel, is_bfloat16_supported

# Load environment variables
load_dotenv()

MAX_LENGTH = 128000

# GPU Memory monitoring class
class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_gpu_utilization(self):
        return pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
    
    def get_gpu_memory_usage(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            'total': info.total / 1024**2,  # MB
            'used': info.used / 1024**2,    # MB
            'free': info.free / 1024**2     # MB
        }

def load_sft_data(data_path: str, tokenizer) -> Dataset:
    """Load and format SFT data from JSON file"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for trace in data["traces"]:
        text = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively.<|eot_id|><|start_header_id|>user<|end_header_id|>
{trace['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{trace['reasoning']}<|eot_id|>"""

        text += tokenizer.eos_token
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def train_model():
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        # Initialize GPU monitoring
        gpu_monitor = GPUMonitor()
        print("Initial GPU Memory Usage:")
        print(gpu_monitor.get_gpu_memory_usage())
        print(f"Initial GPU Utilization: {gpu_monitor.get_gpu_utilization()}%")
    
    if device == "cpu":
        print("Warning: Training on CPU will be very slow. GPU is recommended.")
    
    # Login to wandb using API key from .env
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project="protein-sft", name="llama-70b-4bit-sft-lora")

    # Initialize model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-70b-bnb-4bit",
        max_seq_length=MAX_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True
    )

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

    # Define the templates for completion-only training
    instruction_template = "<|start_header_id|>user<|end_header_id|>"
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    
    # Create the completion-only collator
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Load and process dataset
    train_dataset = load_sft_data("data/mega_cot.json", tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./sft_llama_70b_4bit_lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb",
        seed=3407,
    )

    # Initialize trainer with the collator
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Must be False for completion-only training
        data_collator=collator,
        args=training_args
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Start training
    trainer.train()
    
    # Save final model
    model.save_pretrained("llama_70b_4bit_sft_lora_model")
    tokenizer.save_pretrained("llama_70b_4bit_sft_lora_model")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    train_model()
