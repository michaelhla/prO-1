import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from peft import PeftModel, prepare_model_for_kbit_training
import re
import json
import os
import random
import wandb
from dotenv import load_dotenv

from stability_reward import StabilityRewardCalculator

NUM_EPOCHS = 10

# Load the base model & LoRA adapter exactly as in ds_sft.py, then convert to ValueHead

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
peft_lora_path = "./sft_lora_output/checkpoint-27"  # Changed to use output_dir's final checkpoint

# Load environment variables (for WANDB_API_KEY)
load_dotenv()

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
        "num_responses_per_prompt": 4,
    }
)

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

Copy the final sequence in the brackets of \\boxed{{}} to enclose the sequence:
"""

system_prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. If there is a single final answer, wrap it in \\boxed{{}}. User: {prompt}. Assistant:"""




# Optional: 8-bit quantization config (adjust as needed)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
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
base_model = PeftModel.from_pretrained(base_model, peft_lora_path)

# Convert to a Value Head model for PPO
print("Converting loaded model to a ValueHead for PPO...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, low_cpu_mem_usage=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Disable grad for all non‐LoRA parameters if desired (so only LoRA params update)

for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

model.train()

# Define PPO config & trainer

ppo_config = PPOConfig(
    batch_size=8,
    learning_rate=1.41e-5,
    mini_batch_size=4,
    optimize_cuda_cache=True,
    gradient_accumulation_steps=1
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)


class StabilityScoreCache:
    def __init__(self, cache_file="stability_scores_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
        self.brenda_data = load_brenda_data()
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_stability_score(self, sequence, calculator):
        """Calculate stability score with caching"""
        if sequence in self.cache:
            return self.cache[sequence]
        
        # prob should reformat at some point for efficiency
        sequence_in_brenda = any(
            enzyme_data.get('sequence') == sequence 
            for enzyme_data in self.brenda_data.values()
        )
        
        score = calculator.calculate_stability(sequence)
        
        # Only cache if sequence is in BRENDA dataset
        if sequence_in_brenda:
            self.cache[sequence] = score            
        return score

def calculate_relative_stability(original_seq, modified_seq, cache, calculator):
    """Calculate percentage difference between original and modified sequence stability"""
    original_score = cache.get_stability_score(original_seq, calculator)
    modified_score = cache.get_stability_score(modified_seq, calculator)
    
    # Calculate percentage difference
    reward = -((modified_score - original_score) / abs(original_score)) * 100
    return reward

# Initialize cache and calculator at the start of your script
stability_cache = StabilityScoreCache()
calculator = StabilityRewardCalculator()

# Main GRPO loop

num_responses_per_prompt = 4  # Example group size

# Load transformed BRENDA data
def load_brenda_data():
    with open('data/transformed_brenda.json', 'r') as f:
        brenda_data = json.load(f)
    return brenda_data

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

    return enzyme_prompt


for epoch in range(NUM_EPOCHS):
    all_query_tensors = []
    all_response_tensors = []
    all_rewards = []
    
    # Sample batch_size number of enzymes for this epoch
    batch_enzymes = random.sample(list(stability_cache.brenda_data.items()), k=2)  # or whatever batch size you want
    
    epoch_stats = {
        "mean_reward": 0,
        "max_reward": float('-inf'),
        "min_reward": float('inf'),
        "num_invalid_sequences": 0,
        "num_valid_sequences": 0
    }
    
    for enzyme_id, enzyme_data in batch_enzymes:
        query_tensors_group = []
        response_tensors_group = []
        stability_scores_group = []
        
        # Construct prompt for this enzyme
        prompt = construct_prompt(enzyme_data, enzyme_data['sequence'])
        query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Generate multiple responses for the current prompt
        for _ in range(num_responses_per_prompt):
            response = ppo_trainer.generate(
                query_tensor,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0
            )
            
            response_tensor = response[:, query_tensor.shape[1]:]
            decoded_response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
            
            # Extract sequence from \boxed{}
            sequence_match = re.search(r'\\boxed{(.*?)}', decoded_response)
            if sequence_match:
                modified_sequence = sequence_match.group(1).strip()
                try:
                    stability_score = calculate_relative_stability(
                        original_seq=enzyme_data['sequence'],
                        modified_seq=modified_sequence,
                        cache=stability_cache,
                        calculator=calculator
                    )
                    epoch_stats["num_valid_sequences"] += 1
                    epoch_stats["max_reward"] = max(epoch_stats["max_reward"], stability_score)
                    epoch_stats["min_reward"] = min(epoch_stats["min_reward"], stability_score)
                except Exception as e:
                    print(f"Error calculating stability score: {e}")
                    stability_score = -100.0
                    epoch_stats["num_invalid_sequences"] += 1
            else:
                print(f"Warning: No sequence found in response: {decoded_response[:100]}...")
                stability_score = -100.0
                epoch_stats["num_invalid_sequences"] += 1
            
            query_tensors_group.append(query_tensor)
            response_tensors_group.append(response_tensor)
            stability_scores_group.append(stability_score)
        
        # Calculate group-level advantages
        group_rewards = calculate_group_rewards(stability_scores_group)
        
        all_query_tensors.extend(query_tensors_group)
        all_response_tensors.extend(response_tensors_group)
        all_rewards.extend(group_rewards)
    
    # Convert rewards to tensor and run PPO step
    rewards = torch.stack(all_rewards).to(model.device)
    stats = ppo_trainer.step(all_query_tensors, all_response_tensors, rewards)
    
    epoch_stats["mean_reward"] = rewards.mean().item()
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "mean_reward": epoch_stats["mean_reward"],
        "max_reward": epoch_stats["max_reward"],
        "min_reward": epoch_stats["min_reward"],
        "num_valid_sequences": epoch_stats["num_valid_sequences"],
        "num_invalid_sequences": epoch_stats["num_invalid_sequences"],
        "valid_sequence_ratio": epoch_stats["num_valid_sequences"] / (epoch_stats["num_valid_sequences"] + epoch_stats["num_invalid_sequences"]),
        **stats  # Include PPO training stats
    })
    
    print(f"Epoch {epoch}: Mean reward = {epoch_stats['mean_reward']:.4f}, "
          f"Valid sequences = {epoch_stats['num_valid_sequences']}, "
          f"Invalid sequences = {epoch_stats['num_invalid_sequences']}, "
          f"Std reward = {rewards.std().item():.4f}")

# Don't forget to close wandb at the end
wandb.finish()
