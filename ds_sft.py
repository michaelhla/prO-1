import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json
from typing import Dict, List

def load_sft_data(data_path: str) -> Dataset:
    """Load and format SFT data from JSON file"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for trace in data["traces"]:
        # Format: SEQUENCE: {seq}\nMUTATION: {mut}\nREASONING: {reasoning}
        text = f"SEQUENCE: {trace['sequence']}\n"
        text += f"MUTATION: {trace['mutation']}\n" 
        text += f"REASONING: {trace['reasoning']}\n"
        formatted_data.append({"text": text})
        
    return Dataset.from_list(formatted_data)

def train_model():
    # Initialize model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and process dataset
    train_dataset = load_sft_data("mutation_traces.json")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        fp16=True,
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
    trainer.save_model("./sft_final")

if __name__ == "__main__":
    train_model()
