import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import json
from typing import Dict, List
import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_sft_data(data_path: str, tokenizer) -> Dataset:
    """Load and format SFT data from JSON file"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for trace in data["traces"]:
        # Format conversation with roles
        text = "<SYSTEM>: You are an expert protein engineer with years of experience optimizing protein stability with rational design.\n"
        text += f"<USER>: {trace['prompt']}\n"
        text += """****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTIFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** 
At the end of your response, copy the final sequence, in the format below, using $$ to enclose the sequence:
%%FINAL_SEQUENCE%%: $$_______$$"""
        text += f"<ASSISTANT>: {trace['reasoning']}\n"
        
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
    
    if device == "cpu":
        print("Warning: Training on CPU will be very slow. GPU is recommended.")
    
    # Login to wandb using API key from .env
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    
    # Initialize wandb
    wandb.init(project="protein-sft", name="deepseek-sft-lora")

    # Initialize model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
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
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,                     # Rank of the update matrices
        lora_alpha=32,           # Alpha scaling
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Layers to apply LoRA to
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

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
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
