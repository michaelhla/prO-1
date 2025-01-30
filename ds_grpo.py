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
# Check if local path exists, otherwise download from HF hub
if os.path.exists("./sft_lora_output/checkpoint-27"):
    peft_lora_path = "./sft_lora_output/checkpoint-27"
else:
    from hf_util import sync_with_hf_hub
    sync_with_hf_hub(
        local_path="./sft_lora_output/checkpoint-27",
        repo_id="mhla/prO-1",
        upload=False,
        subfolder="sft_lora_output/checkpoint-27"
    )
    peft_lora_path = "./sft_lora_output/checkpoint-27"

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
<answer> answer here </answer>. If there is a single final answer, wrap it in \\boxed{{}}. User: {enzyme_prompt}. Assistant:"""



    return whole_prompt

def validate_and_construct_prompt(x):
    """Wrapper function to validate data before constructing prompt"""
    try:

        # Ensure required fields exist and are of correct type
        if 'sequence' not in x:
            print(f"Warning: Missing required field 'sequence'")
            return None
            
        # Convert all fields to strings where appropriate
        safe_data = {
            'name': str(x.get('name', 'Unknown')),
            'ec_number': str(x.get('ec_number', 'Unknown')),
            'sequence': str(x['sequence']),
            'general_information': str(x.get('general_information', 'No additional information available')),
            'reaction': [],
            'metal_ions': [],
            'engineering': []
        }
        
        # Handle reaction data
        if 'reaction' in x and x['reaction']:
            safe_data['reaction'] = []
            for reaction in x['reaction']:
                if isinstance(reaction, dict):
                    safe_reaction = {
                        'substrates': [str(s) for s in reaction.get('substrates', [])],
                        'products': [str(p) for p in reaction.get('products', [])]
                    }
                    safe_data['reaction'].append(safe_reaction)
        
        # Handle metal ions
        if 'metal_ions' in x and x['metal_ions']:
            safe_data['metal_ions'] = [str(ion) for ion in x['metal_ions']]
            
        # Handle engineering data
        if 'engineering' in x and x['engineering']:
            safe_data['engineering'] = []
            for mut in x['engineering']:
                if isinstance(mut, dict):
                    safe_mut = {
                        'mutation': str(mut.get('mutation', '')),
                        'effect': str(mut.get('effect', ''))
                    }
                    safe_data['engineering'].append(safe_mut)
        
        # Create the dataset record
        result = {
            "prompt": construct_prompt(safe_data, safe_data['sequence']), 
            "sequences": safe_data['sequence']
        }
        
        # Verify the output is valid
        if not isinstance(result['prompt'], str) or not isinstance(result['sequences'], str):
            print(f"Warning: Invalid output types - prompt: {type(result['prompt'])}, sequences: {type(result['sequences'])}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error processing record: {str(e)}")
        print(f"Problematic record: {json.dumps(x, default=str)}")
        return None

# Create dataset from BRENDA data
with open("data/transformed_brenda.json", 'r') as f:
    data_dict = json.load(f)
    # Convert dictionary values to list of records
    data_list = list(data_dict.values())

# Create the dataset with strict validation
valid_data_list = []
for item in data_list:
    processed = validate_and_construct_prompt(item)
    if processed is not None:
        valid_data_list.append(processed)

# Create dataset from validated records
train_dataset = Dataset.from_list(valid_data_list)
print(f"Dataset size: {len(train_dataset)}")

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
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    use_cache=False  # Disable KV cache to work with gradient checkpointing
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare base model for k-bit training
base_model = prepare_model_for_kbit_training(base_model)

# Load LoRA adapters
print(f"Loading LoRA adapter from {peft_lora_path}")
model = PeftModel.from_pretrained(base_model, peft_lora_path)

# Enable gradient computation for LoRA parameters
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

# Verify some parameters require gradients
trainable_params = [p for p in model.parameters() if p.requires_grad]
if not trainable_params:
    raise ValueError("No parameters have requires_grad=True. Training will not work!")

print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

model.train()  # Ensure model is in training mode

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
        original_score = calculator.calculate_stability(original_seq)
        stability_cache[original_seq] = original_score
        
    modified_score = calculator.calculate_stability(modified_seq)
    
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
    run_name="grpo_training_run",  # Add distinct run name
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
    gradient_checkpointing_kwargs={"use_reentrant": False}  # Explicitly set use_reentrant
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
    processing_class=tokenizer,
    callbacks=[WandBLoggingCallback()], 
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
