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
MAX_LENGTH = 2048

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

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions****

COPY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE:"""

    whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively.<|eot_id|><|start_header_id|>user<|end_header_id|>
{enzyme_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

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

# Data loading section
data_load_start = time.time()

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
print(f"Data loading and processing completed in {time.time() - data_load_start:.2f} seconds")

# Initialize wandb
wandb_start = time.time()
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(
    project="protein-rl",
    name="unsloth-grpo",
    config={
        "model_name": "unsloth/llama-3-70b-bnb-4bit",
        "num_epochs": NUM_EPOCHS,
        "batch_size": 1,
        "learning_rate": 1.41e-5,
        "num_generations": 2,
    }
)

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

# Print model size information
def get_model_size_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convert to GB
    param_size = total_params * 4 / (1024 ** 3)  # Assuming float32
    param_size_4bit = total_params * 0.5 / (1024 ** 3)  # 4-bit quantized
    
    print(f"\nModel Size Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Theoretical size in float32: {param_size:.2f} GB")
    print(f"Approximate size in 4-bit: {param_size_4bit:.2f} GB")

# Print GPU memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"\nGPU {i} Memory Usage:")
            print(f"Total: {total_memory:.2f} GB")
            print(f"Allocated: {allocated_memory:.2f} GB")
            print(f"Cached: {cached_memory:.2f} GB")
            print(f"Free: {total_memory - allocated_memory:.2f} GB")

print("\nChecking initial model size and memory usage...")
get_model_size_info(model)
print_gpu_memory()

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

print("\nChecking model size and memory usage after adding LoRA...")
get_model_size_info(model)
print_gpu_memory()

# Enable gradient computation for LoRA parameters
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False


def calculate_relative_stability(original_seq, modified_seq, calculator):
    """Calculate percentage difference between original and modified sequence stability"""
    if original_seq in stability_cache:
        original_score = stability_cache[original_seq]
    else:
        print(f"Calculating stability for {original_seq}")
        stability_calc_start = time.time()
        original_score = calculator.calculate_stability(original_seq)
        stability_cache[original_seq] = original_score
        print(f"Stability calculation completed in {time.time() - stability_calc_start:.2f} seconds")
        
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

class WandBLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log all metrics from the trainer
            wandb.log(logs, step=state.global_step)
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Log evaluation metrics
            wandb.log({"eval/" + k: v for k, v in metrics.items()}, step=state.global_step) 


# Create training arguments
training_args = GRPOConfig(
    output_dir="./unsloth_grpo_output",
    run_name="unsloth_grpo_training_run",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1.41e-5,
    logging_steps=1,
    num_generations=1,
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

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=stability_reward_func,
    processing_class=tokenizer,
    callbacks=[WandBLoggingCallback()], 
)

# Add memory monitoring before training
print("\nMemory usage before training:")
print_gpu_memory()

# Train the model
trainer.train()

# Print final memory usage
print("\nFinal memory usage:")
print_gpu_memory()

# Save the final model
trainer.save_model("./unsloth_grpo_output/final_model")