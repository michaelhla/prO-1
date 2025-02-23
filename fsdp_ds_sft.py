import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import json
from typing import Dict, List
import wandb
import os
from dotenv import load_dotenv
import pynvml
from torch.cuda import memory_summary
from accelerate import PartialState
from huggingface_hub import login
from accelerate.utils import set_seed
from torch.distributed.checkpoint import save_state_dict
from torch.distributed.checkpoint.state_dict import get_state_dict
from datetime import datetime
import shutil
from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


# accelerate launch --multi_gpu --num_processes=5 ds_sft.py

# Load environment variables
load_dotenv()


MAX_LENGTH = 8192

# Login to Hugging Face using API key from .env
try:
    huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
    login(token=huggingface_token)
except Exception as e:
    print(f"Error logging into Hugging Face: {e}")


# GPU Memory monitoring class
class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
    
    def get_gpu_utilization(self, device_id=None):
        if device_id is not None:
            return pynvml.nvmlDeviceGetUtilizationRates(self.handles[device_id]).gpu
        return [pynvml.nvmlDeviceGetUtilizationRates(handle).gpu for handle in self.handles]
    
    def get_gpu_memory_usage(self, device_id=None):
        def get_memory_info(handle):
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'total': info.total / 1024**2,  # MB
                'used': info.used / 1024**2,    # MB
                'free': info.free / 1024**2     # MB
            }
        
        if device_id is not None:
            return get_memory_info(self.handles[device_id])
        return [get_memory_info(handle) for handle in self.handles]
    
    def get_gpu_names(self):
        return [pynvml.nvmlDeviceGetName(handle).decode('utf-8') for handle in self.handles]
    
    def print_gpu_info(self):
        for i in range(self.num_gpus):
            name = self.get_gpu_names()[i]
            util = self.get_gpu_utilization(i)
            mem = self.get_gpu_memory_usage(i)
            print(f"GPU {i} ({name}):")
            print(f"  Utilization: {util}%")
            print(f"  Memory: {mem['used']:.0f}MB / {mem['total']:.0f}MB ({(mem['used']/mem['total']*100):.1f}%)")

def load_sft_data(data_path: str, tokenizer) -> Dataset:
    """Load and format SFT data from JSON file"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for trace in data["traces"]:
        # Skip traces with no completions
        if not trace.get("completions"):
            print(f"Warning: Found trace with no completions, skipping...")
            continue
            
        try:
            # Remove completion with lowest reward
            trace["completions"].remove(min(trace["completions"], key=lambda x: x["reward"]))

            # Concatenate prompt with all remaining completions
            for completion in trace["completions"]:
                text = trace["prompt"] + completion["completion"]
            
                # Tokenize with explicit max length
                encoded = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                
                formatted_data.append({"text": text})
        except Exception as e:
            print(f"Warning: Error processing trace: {e}")
            continue
    
    if not formatted_data:
        raise ValueError("No valid data was processed. Check your input file.")
        
    return Dataset.from_list(formatted_data)

def train_model():
    state = PartialState()
    local_rank = state.local_process_index
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
    # Configure model loading with explicit dtype
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,  # More memory efficient storage
    )

    # Load model with explicit dtype
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure PEFT with explicit dtype
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
    
    # Print any remaining float32 parameters
    print("\nChecking for any remaining float32 parameters:")
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            print(f"Parameter {name} is still in float32")

    # After loading the model, explicitly set use_cache to False
    model.config.use_cache = False
    

    try: 
        gpu_monitor = GPUMonitor()
        print("Initial GPU Memory Usage:")
        gpu_monitor.print_gpu_info()
    except Exception as e:
        print(f"Error initializing GPU monitor: {e}")
    
    # Login to wandb using API key from .env only on main process
    if state.is_main_process:
        try:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            wandb.init(project="protein-sft", name="8h100-sft")
        except Exception as e:
            print(f"Error initializing wandb: {e}")
    
    # Initialize tokenizer first
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", trust_remote_code=True, model_max_length=MAX_LENGTH)
    
    # Add padding token before model creation
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Ensure padding token id is properly set
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Load and process dataset
    train_dataset = load_sft_data("data/r1-gen-sft-final.json", tokenizer)
    # Filter out entries that exceed max length
    def filter_by_length(example):
        # Tokenize the full text to check length
        tokenized = tokenizer(
            example["text"],
            truncation=False,
            padding=False,
        )
        # Get length from input_ids instead of expecting a "length" key
        return len(tokenized["input_ids"]) <= MAX_LENGTH

    # Apply the filter and print stats
    original_size = len(train_dataset)
    train_dataset = train_dataset.filter(filter_by_length)
    filtered_size = len(train_dataset)
    print(f"Filtered dataset from {original_size} to {filtered_size} examples")
    print(f"Removed {original_size - filtered_size} examples that exceeded max length of {MAX_LENGTH}")

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


    # Set tokenizer parallelism explicitly before any tokenizer operations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Update training arguments with modern FSDP state dict handling
    training_args = TrainingArguments(
        output_dir="./sft_llama_70b_4bit_lora_output",
        num_train_epochs=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        bf16=True,
        fp16=False,
        optim="paged_adamw_32bit",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "activation_checkpointing": True,
        },
    )


    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        callbacks=[WandBLoggingCallback(), CheckpointCallback(
            checkpoint_dir="./r1_llama_70b_sft_output/checkpoints",
            checkpoint_freq=65,  # CHANGE ONLY FOR TESTING 
            max_checkpoints=1000     # Keep last 5 checkpoints
        )]) 

    

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Start training
    trainer.train()
    
    # Save final model
    if state.is_main_process:
        final_model_path = Path("r1_llama_70b_4bit_sft_lora_model")
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # Get and save the full state dict
        full_state_dict = model.state_dict()
        torch.save(full_state_dict, final_model_path / "pytorch_model.bin")
        
        # Save the model config
        model.config.save_pretrained(final_model_path)
        
        # Save the tokenizer
        tokenizer.save_pretrained(final_model_path)
        
        print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    train_model()