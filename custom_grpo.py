import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback, AutoModelForCausalLM
from datasets import Dataset
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil
from debug import GRPOTrainer
from debug import GRPOConfig
from accelerate import PartialState
import re
import random
import math
import time
from torch import nn
from huggingface_hub import login
from bitsandbytes.optim import PagedAdamW32bit
import wandb
from dotenv import load_dotenv
from torch.distributed.fsdp import StateDictType

from stability_reward import StabilityRewardCalculator

RUN_NAME = 'debugging-custom-grpo'
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 6000
MAX_OUTPUT_LENGTH = 4096


##################################################################
# 1. Initialize Distributed Process
##################################################################
def init_distributed(rank, world_size, backend="nccl"):
    """
    Initializes the default process group for distributed training.
    Make sure to set the appropriate MASTER_ADDR and MASTER_PORT
    environment variables or pass them via your launching script.
    """
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # For reproducibility
    torch.manual_seed(42)


##################################################################
# 2. Build Model
##################################################################
def build_model(model_name_or_path, device):
    """
    Loads the LLaMA model with 4-bit quantization and LoRA, then wraps with FSDP.
    """
    # Add memory debugging at start
    proc_state = PartialState()
    
    # Initialize process state
    local_rank = proc_state.local_process_index
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    # Print DeepSpeed status and ZeRO stage
    if proc_state.is_main_process:
        print("\nDistributed setup:")
        print(f"- Local Rank: {local_rank}")
        print(f"- World Size: {torch.distributed.get_world_size()}")
        print(f"- Distributed Type: {proc_state.distributed_type}")
        print(f"- Device: {device}")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load model with quantization
    print(f"Loading model from: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # use_cache=True,  # Important: Set this to False for training
        # device_map="auto",  # Let the model handle device placement
    )

    # Prepare for kbit training - WITHOUT use_reentrant
    model = prepare_model_for_kbit_training(
        model, 
    )

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=[],
    )
    model = get_peft_model(model, peft_config)

    # model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    # model.enable_input_require_grads()  # Enable input gradients

    # Debug: Check trainable parameters
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()*4

    if proc_state.is_main_process:
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
    return model


##################################################################
# Callbacks
##################################################################
class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir="checkpoints", checkpoint_freq=100, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.checkpoint_freq == 0:
            self._save_checkpoint(args, state)
            
    def _save_checkpoint(self, args, state):
        proc_state = PartialState()
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"checkpoint-{timestamp}-step{state.global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            if proc_state.is_main_process:
                print("Saving checkpoint, step:", state.global_step)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Save only the LoRA adapter weights
                state.model.module.save_pretrained(checkpoint_path)
                
                # Save training state
                training_state = {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "best_metric": state.best_metric,
                    "training_args": args.to_dict(),
                }
                torch.save(training_state, checkpoint_path / "trainer_state.pt")
                
                # Maintain max_checkpoints
                checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
                if len(checkpoints) > self.max_checkpoints:
                    for checkpoint in checkpoints[:-self.max_checkpoints]:
                        shutil.rmtree(checkpoint)
                        
                print(f"Saved LoRA checkpoint: {checkpoint_path}")
            
        except Exception as e:
            if proc_state.is_main_process:
                print(f"Error saving checkpoint: {e}")
        
        torch.distributed.barrier()

def load_from_checkpoint(checkpoint_path, model, trainer):
    """Load LoRA weights and training state from checkpoint"""
    try:
        checkpoint_path = Path(checkpoint_path)
        
        # Load only LoRA weights
        model.module.load_adapter(checkpoint_path, "default")
        
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


##################################################################
# Reward Function
##################################################################
def levenshtein_distance(s1, s2):
    """Calculate Levenshtein edit distance between sequences"""
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


def calculate_stability_reward(calculator, prompts, completions, sequences, orig_stabs, device=None):
    """
    Calculate stability rewards for each completion in a distributed setting.
    
    Args:
        prompts (list): List of prompts
        completions (list): List of model completions
        sequences (list): Original sequences
        orig_stabs (list): Original stability scores
        device (str, optional): Device to use for calculations
    """
    
    rewards = []
    for completion, sequence, orig_stab in zip(completions, sequences, orig_stabs):
        try:
            reward = 0.0
            
            # Extract modified sequence from completion
            sequence_match = re.search(r'\\boxed{(.*?)}', completion)
            if not sequence_match:
                rewards.append(reward)
                continue
            
            modified_sequence = sequence_match.group(1).strip()

            # Calculate edit distance reward
            edit_dist = levenshtein_distance(sequence, modified_sequence)
            if edit_dist <= 10:
                reward += 0.3

            # Calculate stability with calculator
            modified_score = calculator.calculate_stability(modified_sequence)
            stab_calc = -((modified_score - orig_stab) / abs(orig_stab)) * 100

            # Add rewards for valid sequence and improved stability
            if stab_calc:
                reward += 0.3
            if stab_calc > 0.0:
                reward += 1.0

            rewards.append(reward)

        except Exception as e:
            print(f"Error calculating stability score: {e}")
            rewards.append(0.0)

    return rewards


##################################################################
# 5. Main Function
##################################################################
def main(rank, world_size, model_path, epochs=1):
    # Remove the init_distributed call since PartialState already handled it
    # init_distributed(rank, world_size)  # Remove this line

    device = torch.device(f"cuda:{rank}")
    calculator = StabilityRewardCalculator(device=device)

    # Build the model (FSDP or DDP)
    model = build_model(model_path, device)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct", 
        trust_remote_code=True, 
        model_max_length=MAX_INPUT_LENGTH
    )

    # Instead of adding a new token, use an existing one (typically eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    print(f"Pad token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, id: {tokenizer.eos_token_id}")

    # Load dataset directly from train_dataset.json
    with open("data/train_dataset.json", 'r') as f:
        valid_data_list = json.load(f)

    # Create dataset from records
    train_dataset = Dataset.from_list(valid_data_list)
    print(f"Dataset size: {len(train_dataset)}")

    # Configure training arguments with DeepSpeed
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=MAX_INPUT_LENGTH,
        max_completion_length=MAX_OUTPUT_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        max_grad_norm=0.1,
        output_dir=f"./{RUN_NAME}",
        do_sample=True,
        use_cache=True
    )

                            
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=calculate_stability_reward,
        calculator=calculator,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[WandBLoggingCallback(),CheckpointCallback(
            checkpoint_dir=f"./{RUN_NAME}/checkpoints",
            checkpoint_freq=8, 
            max_checkpoints=5     # Keep last 5 checkpoints
        )]
    )

    trainer.train()

    wandb.finish()

    # Cleanup
    dist.destroy_process_group()    
    return trainer


if __name__ == "__main__":
    """
    Example usage:
    torchrun --nproc_per_node=2 train.py --model_path=./llama_weights --epochs=1
    """


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    # Use accelerate's PartialState to get rank information
    proc_state = PartialState()
    rank = proc_state.local_process_index
    world_size = proc_state.num_processes

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    load_dotenv()

     # Login to Hugging Face using API key from .env
    try:
        huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
        login(token=huggingface_token)
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")

    if proc_state.is_main_process:
        try:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            wandb.init(project="protein-cluster", name=f"{RUN_NAME}")
        except Exception as e:
            print(f"Error initializing wandb: {e}")

    main(
        rank=rank,
        world_size=world_size,
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        epochs=args.epochs,
    )