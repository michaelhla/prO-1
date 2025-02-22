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
# 2. Build Model and Wrap with FSDP
##################################################################
def build_fsdp_model(model_name_or_path, device, use_fsdp=True):
    """
    Loads the LLaMA model with 4-bit quantization and LoRA, then wraps with FSDP.
    """
    # Initialize process state
    proc_state = PartialState()
    local_rank = proc_state.local_process_index
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

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
        device_map={"": local_rank},
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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

    # Convert any remaining parameters to bfloat16
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    # Disable model caching
    model.config.use_cache = False

    if use_fsdp:
        # # Define mixed precision policy
        # mixed_precision_policy = MixedPrecision(
        #     param_dtype=torch.bfloat16,
        #     reduce_dtype=torch.bfloat16,
        #     buffer_dtype=torch.bfloat16
        # )

        # Configure auto wrapping policy
        # auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={LlamaDecoderLayer})

        # Wrap with FSDP

        def _is_peft_model(model):
            classes_to_check = (PeftModel,)
            # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
            return isinstance(model, classes_to_check)

        # Check if model is PeftModel before FSDP wrapping
        if _is_peft_model(model):
            print("Base model is correctly wrapped with PEFT")
        else:
            print("Warning: Base model is not a PeftModel")

        # Wrap with FSDP first
        wrapped_model = FSDP(
            model,
            # auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
        )
        if _is_peft_model(wrapped_model):
            print("Wrapped model is correctly wrapped with PEFT")
            print(f"Wrapped model class: {wrapped_model.__class__.__name__}")
            print(f"Full class hierarchy: {type(wrapped_model).mro()}")
        else:
            print("Warning: Wrapped model is not a PeftModel")
            print(f"Wrapped model class: {wrapped_model.__class__.__name__}")
            print(f"Full class hierarchy: {type(wrapped_model).mro()}")
        
        print("\nFSDP Sharding Information:")
        print("=" * 50)
        
        def print_sharding_info(module, prefix=""):
            if isinstance(module, FSDP):
                print(f"{prefix}FSDP Wrapped: {module._get_name()}")
                print(f"{prefix}├── Rank: {proc_state.local_process_index}")
                
                # Count trainable (LoRA) parameters (these are in bf16)
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Count frozen parameters (these are in 4-bit)
                frozen_params = sum(p.numel() * 4 if hasattr(p, 'quant_state') else p.numel() 
                                  for p in module.parameters() if not p.requires_grad)
                
                print(f"{prefix}├── Trainable (LoRA) parameters: {trainable_params:,}")
                print(f"{prefix}├── Frozen parameters (unpacked): {frozen_params:,}")
                
                # Get GPU memory usage
                gpu_memory = torch.cuda.memory_allocated(device=proc_state.local_process_index)
                gpu_memory_reserved = torch.cuda.memory_reserved(device=proc_state.local_process_index)
                print(f"{prefix}└── GPU Memory: {gpu_memory/1e9:.2f}GB allocated, {gpu_memory_reserved/1e9:.2f}GB reserved")
            
            for name, child in module.named_children():
                print_sharding_info(child, prefix + "    ")
        
        print_sharding_info(wrapped_model)
        
        # Separate total counts for trainable and frozen parameters
        total_trainable = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in wrapped_model.parameters() if not p.requires_grad)
        
        if proc_state.is_main_process:
            print(f"\nTotal trainable (LoRA) parameters: {total_trainable:,}")
            print(f"Total frozen (base) parameters: {total_frozen:,}")
            print(f"Expected trainable parameters per rank: ~{total_trainable // world_size:,}")
        print("=" * 50)
    
    else:
        print('FSDP SETUP FAILED')
    
    return wrapped_model


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
        """Save checkpoint every checkpoint_freq steps"""
        if state.global_step % self.checkpoint_freq == 0:
            self._save_checkpoint(args, state)
            
    def _save_checkpoint(self, args, state):
        """Save LoRA checkpoint and maintain max number of checkpoints"""
        proc_state = PartialState()
        
        # Create checkpoint name with timestamp and step
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"checkpoint-{timestamp}-step{state.global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            # All processes need to participate in the state dict gathering
            with FSDP.state_dict_type(state.model, StateDictType.FULL_STATE_DICT):
                full_state_dict = state.model.state_dict()
            
            # Only main process saves the files
            if proc_state.is_main_process:
                print("Saving checkpoint, step:", state.global_step)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Save the gathered state dict
                torch.save(full_state_dict, checkpoint_path / "pytorch_model.bin")
                
                # Save the model config separately
                state.model.config.save_pretrained(checkpoint_path)
                
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
            if proc_state.is_main_process:
                print(f"Error saving checkpoint: {e}")
        
        # Make sure all processes sync up before continuing
        torch.distributed.barrier()

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
class DistributedStabilityRewardFunc:
    def __init__(self, think_length=1000):
        # Initialize calculator on each GPU
        self.device = f"cuda:{torch.distributed.get_rank()}"
        self.calculator = StabilityRewardCalculator(device=self.device)
        self.think_length = think_length
        self.__name__ = "stability_reward"  # Required for GRPO logging

    def __call__(self, prompts, completions, sequences, orig_stabs, **kwargs):
        """Distributed reward calculation logic"""
        rewards = []
        for i, (prompt, completion, sequence, orig_stab) in enumerate(zip(prompts, completions, sequences, orig_stabs)):
            try:
                reward = 0.0
                print(f"COMPLETION {i}")
                print(completion)
                print('-'*100)

                # # Calculate reward for length of thinking section
                # think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
                # if think_match:
                #     think_text = think_match.group(1)
                #     # Count tokens in thinking section
                #     think_tokens = len(think_text.split())
                #     # Gaussian reward centered at THINK_LENGTH tokens with std dev of 1000
                #     token_reward = math.exp(-((think_tokens - self.think_length)**2)/(2*1000**2))
                #     reward += 0.2 * token_reward

                # Extract modified sequence from completion
                sequence_match = re.search(r'\\boxed{(.*?)}', completion)
                if not sequence_match:
                    rewards.append(reward)
                    continue
                
                modified_sequence = sequence_match.group(1).strip()

                # Calculate edit distance and give reward if within 10 modifications
                edit_dist = self.levenshtein_distance(sequence, modified_sequence)
                if edit_dist <= 10:
                    reward += 0.3

                # Calculate stability with local calculator
                stab_calc = self.calculate_relative_stability(
                    sequence, modified_sequence, orig_stab
                )

                ## reward for valid sequence in \boxed{}
                if stab_calc:
                    reward += 0.3

                if stab_calc > 0.0:
                    reward += 1.0

                rewards.append(reward)

            except Exception as e:
                print(f"Error calculating stability score: {e}")
                rewards.append(reward)

        return rewards

    def calculate_relative_stability(self, original_seq, modified_seq, orig_stab):
        """Calculate stability using local calculator"""
        modified_score = self.calculator.calculate_stability(modified_seq)
        reward = -((modified_score - orig_stab) / abs(orig_stab)) * 100
        return reward

    @staticmethod
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

##################################################################
# 5. Main Function
##################################################################
def main(rank, world_size, model_path, epochs=1, use_fsdp=True):
    # Remove the init_distributed call since PartialState already handled it
    # init_distributed(rank, world_size)  # Remove this line

    device = torch.device(f"cuda:{rank}")

    # Build the model (FSDP or DDP)
    model = build_fsdp_model(model_path, device, use_fsdp=use_fsdp)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", trust_remote_code=True, model_max_length=MAX_INPUT_LENGTH)
    
    # Add padding token before model creation
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Ensure padding token id is properly set
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')



    # Load dataset directly from train_dataset.json
    with open("data/train_dataset.json", 'r') as f:
        valid_data_list = json.load(f)

    # Create dataset from records
    train_dataset = Dataset.from_list(valid_data_list)
    print(f"Dataset size: {len(train_dataset)}")

    # Create reward function instance
    reward_func = DistributedStabilityRewardFunc()

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
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],  # Keep your existing reward function
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[WandBLoggingCallback(),CheckpointCallback(
            checkpoint_dir=f"./{RUN_NAME}/checkpoints",
            checkpoint_freq=8, 
            max_checkpoints=5     # Keep last 5 checkpoints
        )]
    )

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
        use_fsdp=True
    )