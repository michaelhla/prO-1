import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import re
import json
import os
import random
import wandb
from dotenv import load_dotenv
from datasets import Dataset
import time
from accelerate import PartialState
from huggingface_hub import login
from bitsandbytes.optim import PagedAdamW32bit
import torch.nn as nn
from pathlib import Path
import shutil
from datetime import datetime
import math

from stability_reward import StabilityRewardCalculator

# accelerate launch fsdp_grpo.py
load_dotenv()

NUM_EPOCHS = 5
MAX_INPUT_LENGTH = 6000
MAX_OUTPUT_LENGTH = 4096
THINK_LENGTH = 3000
DEVICE = "cuda"
RUN_NAME = "unsloth-grpo-mega-run"

# Print model size information
def get_model_size_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get actual memory usage
    memory_params = sum(p.nelement() * p.element_size() for p in model.parameters())
    memory_buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_memory = memory_params + memory_buffers  # in bytes
    
    # Convert to more readable formats
    def bytes_to_mb(bytes_val): return bytes_val / (1024 * 1024)
    def bytes_to_gb(bytes_val): return bytes_val / (1024 * 1024 * 1024)
    
    print(f"\nModel Size Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Actual model size in memory: {bytes_to_gb(total_memory):.2f} GB")
    
    # If using CUDA, also show GPU memory usage
    if next(model.parameters()).is_cuda:
        print("\nGPU Memory Usage:")
        print(f"Allocated: {bytes_to_gb(torch.cuda.memory_allocated()):.2f} GB")
        print(f"Cached: {bytes_to_gb(torch.cuda.memory_reserved()):.2f} GB")
        
        # Show per-tensor memory usage (top 5 largest tensors)
        print("\nLargest Model Tensors:")
        tensor_sizes = [(name, p.nelement() * p.element_size()) 
                       for name, p in model.named_parameters()]
        tensor_sizes.sort(key=lambda x: x[1], reverse=True)
        for name, size in tensor_sizes[:5]:
            print(f"{name}: {bytes_to_mb(size):.2f} MB")


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

COPY THE FINAL SEQUENCE AND ONLY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. DO NOT INCLUDE ANY OTHER TEXT OR FORMATTING WITHIN THE BRACKETS."""

    whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 tokens. |eot_id|><|start_header_id|>user<|end_header_id|>
{enzyme_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return whole_prompt

def validate_and_construct_prompt(x):
    """Wrapper function to validate data before constructing prompt"""
    try:
        # Ensure required fields exist and are of correct type
        if 'sequence' not in x or 'orig_stab' not in x:
            print(f"Warning: Missing required field 'sequence' or 'orig_stab'")
            return None
            
        # Convert all fields to strings where appropriate
        safe_data = {
            'name': str(x.get('name', 'Unknown')),
            'ec_number': str(x.get('ec_number', 'Unknown')),
            'sequence': str(x['sequence']),
            'general_information': str(x.get('general_information', 'No additional information available')),
            'reaction': [],
            'metal_ions': [],
            'engineering': [],
            'orig_stab': x['orig_stab']  # Keep as number, don't convert to string
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
            "sequences": safe_data['sequence'],
            "orig_stabs": safe_data['orig_stab']  # Changed from orig_stab to orig_stabs to match usage
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

# Load dataset directly from train_dataset.json
with open("data/train_dataset.json", 'r') as f:
    valid_data_list = json.load(f)

# Create dataset from records
train_dataset = Dataset.from_list(valid_data_list)
print(f"Dataset size: {len(train_dataset)}")
print(f"Data loading completed in {time.time() - data_load_start:.2f} seconds")

# Calculate and print dataset statistics for prompt lengths
prompt_lengths = [len(example['prompt']) for example in valid_data_list]
print("\nPrompt Length Statistics:")
print(f"Mean length: {sum(prompt_lengths) / len(prompt_lengths):.2f}")
print(f"Median length: {sorted(prompt_lengths)[len(prompt_lengths)//2]}")
print(f"Max length: {max(prompt_lengths)}")
print(f"Min length: {min(prompt_lengths)}")


# Initialize wandb only on main process
proc_state = PartialState()
if proc_state.is_main_process:
    wandb_start = time.time()
    try:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(
            project="protein-rl",
            name=RUN_NAME,
            config={
                "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "num_epochs": NUM_EPOCHS,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "num_generations": 4,
            }
        )
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        # Login to Hugging Face before any model loading
    try:
        huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
        if not huggingface_token:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        login(token=huggingface_token)
        print("Successfully logged into Hugging Face")
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")
        raise  # Add this to stop execution if login fails

# Before training, set up the reward calculator on the dedicated GPU
def setup_reward_calculator():

    # Use the last GPU for reward calculation
    reward_device = torch.device(DEVICE)
    
    calculator = StabilityRewardCalculator(device=DEVICE)  # Initialize your calculator here
    
    return calculator



def calculate_relative_stability(original_seq, modified_seq, calculator, orig_stab):
    """Calculate percentage difference between original and modified sequence stability"""
    # Move calculations to a dedicated GPU (e.g., last available GPU)
    with torch.cuda.device(DEVICE):
        modified_score = calculator.calculate_stability(modified_seq)
    
    # Calculate percentage difference
    reward = -((modified_score - orig_stab) / abs(orig_stab)) * 100
    return reward

def stability_reward_func(prompts, completions, sequences, orig_stabs, **kwargs):
    """Custom reward function for stability optimization"""
    rewards = []
    
    for i, (prompt, completion, sequence, orig_stab) in enumerate(zip(prompts, completions, sequences, orig_stabs)):
        try:
            reward = 0.0
            print(f"COMPLETION {i}")
            print(completion)
            print('-'*100)

            # Calculate reward for length of thinking section
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            if think_match:
                think_text = think_match.group(1)
                # Count tokens in thinking section
                think_tokens = len(think_text.split())
                # Gaussian reward centered at THINK_LENGTH tokens with std dev of 1000
                token_reward = math.exp(-((think_tokens - THINK_LENGTH)**2)/(2*1000**2))
                reward += 0.2*token_reward

            # Extract modified sequence from completion
            sequence_match = re.search(r'\\boxed{(.*?)}', completion)
            if not sequence_match:
                rewards.append(reward)
                continue
            modified_sequence = sequence_match.group(1).strip()

            # Calculate Levenshtein edit distance between sequences
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]

            # Calculate edit distance and give reward if within 10 modifications
            edit_dist = levenshtein_distance(sequence, modified_sequence)
            if edit_dist <= 10:
                reward += 0.3
            
            # Calculate reward using the original sequence passed in via dataset
            stab_calc = calculate_relative_stability(
                original_seq=sequence,
                modified_seq=modified_sequence,
                calculator=calculator,
                orig_stab=orig_stab
            )

            if stab_calc: 
                reward+=0.3

            if stab_calc > 0.0:
                reward += 1.0
            
            rewards.append(reward)
                        
        except Exception as e:
            print(f"Error calculating stability score: {e}")
            rewards.append(reward)
            
    return rewards

class WandBLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        proc_state = PartialState()
        if proc_state.is_main_process and logs:  # Only log on main process
            # Log all metrics from the trainer
            wandb.log(logs, step=state.global_step)
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        proc_state = PartialState()
        if proc_state.is_main_process and metrics:  # Only log on main process
            # Log evaluation metrics
            wandb.log({"eval/" + k: v for k, v in metrics.items()}, step=state.global_step) 

class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir="checkpoints", checkpoint_freq=100, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        """Save checkpoint every checkpoint_freq steps"""
        if state.global_step % self.checkpoint_freq == 0:
            self._save_checkpoint(args, state)
            
    def _save_checkpoint(self, args, state):
        """Save LoRA checkpoint and maintain max number of checkpoints"""
        proc_state = PartialState()
        if not proc_state.is_main_process:
            return
            
        # Create checkpoint name with timestamp and step
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"checkpoint-{timestamp}-step{state.global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            # Save LoRA weights and config
            state.model.save_pretrained(checkpoint_path)  # This saves LoRA weights
            
            # Save additional training state
            training_state = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "training_args": args.to_dict(),
            }
            torch.save(training_state, checkpoint_path / "trainer_state.pt")
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_path)
            
            # Maintain only max_checkpoints number of checkpoints
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
            if len(checkpoints) > self.max_checkpoints:
                for checkpoint in checkpoints[:-self.max_checkpoints]:
                    shutil.rmtree(checkpoint)
                    
            print(f"Saved LoRA checkpoint: {checkpoint_path}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

def load_from_checkpoint(checkpoint_path, model, trainer):
    """Load LoRA weights and training state from checkpoint"""
    try:
        checkpoint_path = Path(checkpoint_path)
        
        # Load LoRA weights
        model.load_adapter(checkpoint_path, "default")  # Load LoRA weights
        
        # Load training state
        training_state = torch.load(checkpoint_path / "trainer_state.pt")
        trainer.state.global_step = training_state["global_step"]
        trainer.state.epoch = training_state["epoch"]
        trainer.state.best_metric = training_state["best_metric"]
        
        print(f"Loaded LoRA checkpoint from {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

calculator = setup_reward_calculator()

# Initialize Unsloth's FastLanguageModel with GRPO patch
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

# Model initialization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/meta-Llama-3.1-8B-Instruct",
    max_seq_length=MAX_INPUT_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=32,  # Adjust based on your needs
    gpu_memory_utilization=0.6,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,

)

# Modify training arguments for Unsloth compatibility
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=1e-3,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=MAX_INPUT_LENGTH,
    max_completion_length=MAX_OUTPUT_LENGTH,
    num_train_epochs=NUM_EPOCHS,
    max_grad_norm=0.1,
    output_dir=f"./{RUN_NAME}",
)

# Initialize GRPO trainer with Unsloth configuration
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[stability_reward_func],  # Keep your existing reward function
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[WandBLoggingCallback(),CheckpointCallback(
        checkpoint_dir=f"./{RUN_NAME}/checkpoints",
        checkpoint_freq=8, 
        max_checkpoints=5     # Keep last 5 checkpoints
    )]
)

# Train the model
trainer.train()

# Save the model - Unsloth style
model.save_pretrained(f"./{RUN_NAME}/final_model")

if proc_state.is_main_process:
    wandb.finish()
