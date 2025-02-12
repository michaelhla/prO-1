import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import json
from typing import Dict, List
import wandb
import os
from dotenv import load_dotenv
import pynvml
from torch.cuda import memory_summary

# Load environment variables
load_dotenv()

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
        enzyme_prompt = trace['prompt']
        text = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. The assistant pays close attention to the user's instructions. <|User|>: {enzyme_prompt}. <|Assistant|>:"""
        text += trace['reasoning']

        
        # Tokenize the text
        tokenized = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
        
        # Add labels (same as input_ids for language modeling)
        formatted_data.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()  # Add labels
        })
        
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
    
    # Initialize wandb
    wandb.init(project="protein-sft", name="deepseek-mega-sft-lora")

    # Initialize model and tokenizer
    model_name = 'unsloth/DeepSeek-V3'
    
    # Create a BitsAndBytesConfig object for 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA for 670B model
    lora_config = LoraConfig(
        r=64,                     # Increased rank for 670B model (from 16)
        lora_alpha=128,          # Usually 2x the rank
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters info

    # Set model to training mode
    model.train()
    
    # Disable caching (required for gradient checkpointing)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load and process dataset with tokenizer
    train_dataset = load_sft_data("data/cot_mutation_traces.json", tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./sft_lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Can increase this now
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # Slightly higher learning rate for LoRA
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        remove_unused_columns=True,
    )

    # After model initialization, print memory usage
    if device == "cuda":
        print("\nGPU Memory Usage after model loading:")
        print(gpu_monitor.get_gpu_memory_usage())
        print(torch.cuda.memory_summary())
    
    # Modify the trainer initialization to include callbacks for monitoring
    class GPUMetricsCallback(TrainerCallback):
        def __init__(self, gpu_monitor):
            self.gpu_monitor = gpu_monitor
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero:
                memory_usage = self.gpu_monitor.get_gpu_memory_usage()
                gpu_util = self.gpu_monitor.get_gpu_utilization()
                wandb.log({
                    "gpu_memory_used_MB": memory_usage['used'],
                    "gpu_utilization": gpu_util
                }, commit=False)

    callbacks = []
    if device == "cuda":
        callbacks.append(GPUMetricsCallback(gpu_monitor))

    # Initialize trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks  # Add callbacks here
    )

    # Start training
    trainer.train()
    
    # Save final model
    try:
        trainer.save_model("/workspace/sft_lora_final")
    except:
        trainer.save_model("./sft_lora_final")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    train_model()
