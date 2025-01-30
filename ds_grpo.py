import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.core import LengthSampler
from peft import PeftModel, prepare_model_for_kbit_training
import re
import json
import os
import random
import wandb
from dotenv import load_dotenv
from datasets import Dataset

from stability_reward import StabilityRewardCalculator

NUM_EPOCHS = 10

# Load the base model & LoRA adapter exactly as in ds_sft.py, then convert to ValueHead

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
peft_lora_path = "./sft_lora_output/checkpoint-27"  # Changed to use output_dir's final checkpoint

# Load environment variables (for WANDB_API_KEY)
load_dotenv()

def construct_prompt(enzyme_data, sequence):
    """Construct prompt for a single enzyme"""
    # Get reaction, substrates and products from first reaction if available
    if enzyme_data.get('reaction'):
        reaction = random.choice(enzyme_data['reaction'])
        substrates = reaction['substrates'] if reaction else ['Unknown']
        products = reaction['products'] if reaction else ['Unknown']
    else:
        substrates = ['Unknown']
        products = ['Unknown']

    # Get metals/ions if available
    metal_ions = enzyme_data.get('metal_ions', ['None'])
    if not metal_ions:
        metal_ions = ['None']

    # Format known mutations text
    known_mutations_text = ""
    if enzyme_data.get('engineering'):
        known_mutations_text = "KNOWN MUTATIONS AND EFFECTS:\n" + ''.join([
            f"- {mut['mutation']}: {mut['effect']}\n" 
            for mut in enzyme_data.get('engineering', [])
        ])

    # Construct the prompt
    enzyme_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data.get('name', 'Unknown')}
EC NUMBER: {enzyme_data.get('ec_number', 'Unknown')}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data.get('general_information', 'No additional information available')}
SUBSTRATES: {', '.join(substrates)}
PRODUCTS: {', '.join(products)}
METALS/IONS: {', '.join(metal_ions)}
{known_mutations_text}

Propose mutations to optimize the stability of the enzyme given the information above. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects (or does not affect) protein structure
2. How the mutation affects (or does not affect) protein function
3. The chemical properties of the amino acids and substrates/products

****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTIFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** 

Copy the final sequence in the brackets of \\boxed{{}} to enclose the sequence:"""

    whole_prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. If there is a single final answer, wrap it in \\boxed{{}}. User: {prompt}. Assistant:"""



    return whole_prompt


# Initialize wandb
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(
    project="protein-rl",
    name="deepseek-grpo",
    config={
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "peft_lora_path": "./sft_lora_output/checkpoint-27",  # Updated config to match
        "num_epochs": NUM_EPOCHS,
        "batch_size": 2,
        "learning_rate": 1.41e-5,
        "num_generations": 4,
    }
)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

# Prepare base model for k-bit training (to keep it consistent with your ds_sft.py approach)
base_model = prepare_model_for_kbit_training(base_model)

# Load LoRA adapters
print(f"Loading LoRA adapter from {peft_lora_path}")
model = PeftModel.from_pretrained(base_model, peft_lora_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Disable grad for all non‐LoRA parameters if desired (so only LoRA params update)

for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

model.train()

calculator = StabilityRewardCalculator()

# Try to load existing stability cache, create new if not found
try:
    with open('stability_cache.json', 'r') as f:
        stability_cache = json.load(f)
except FileNotFoundError:
    stability_cache = {}

def calculate_relative_stability(original_seq, modified_seq, calculator):
    """Calculate percentage difference between original and modified sequence stability"""
    if original_seq in stability_cache:
        original_score = stability_cache[original_seq]
    else:
        original_score = calculator.get_stability_score(original_seq)
        stability_cache[original_seq] = original_score
        
    modified_score = calculator.get_stability_score(modified_seq)
    
    # Calculate percentage difference
    reward = -((modified_score - original_score) / abs(original_score)) * 100
    return reward

def stability_reward_func(prompts, completions, sequences, **kwargs):
    """Custom reward function for stability optimization"""
    rewards = []
    
    for prompt, completion, sequence in zip(prompts, completions, sequences):
        try:
            # Extract modified sequence from completion
            sequence_match = re.search(r'\\boxed{(.*?)}', completion)
            if not sequence_match:
                rewards.append(-100.0)
                continue
                
            modified_sequence = sequence_match.group(1).strip()
            
            # Calculate reward using the original sequence passed in via dataset
            reward = calculate_relative_stability(
                original_seq=sequence,
                modified_seq=modified_sequence,
                calculator=calculator
            )
            rewards.append(reward)
            
        except Exception as e:
            print(f"Error calculating stability score: {e}")
            rewards.append(-100.0)
            
    return rewards

# Create training arguments
training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1.41e-5,
    logging_steps=1,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=512,
    temperature=0.7,
    beta=0.04,
    remove_unused_columns=False,
)

# Create dataset from BRENDA data
train_dataset = Dataset.from_json("data/brenda_data.json").map(
    lambda x: {"prompt": construct_prompt(x, x['sequence']), "sequences": x['sequence']}
)

class WandBLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log all metrics from the trainer
            wandb.log(logs, step=state.global_step)
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Log evaluation metrics
            wandb.log({"eval/" + k: v for k, v in metrics.items()}, step=state.global_step)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=stability_reward_func,
    tokenizer=tokenizer,
    callbacks=[WandBLoggingCallback()]
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("./grpo_output/final_model")

# Save the stability cache
with open('stability_cache.json', 'w') as f:
    json.dump(stability_cache, f)

# Close wandb
wandb.finish()
