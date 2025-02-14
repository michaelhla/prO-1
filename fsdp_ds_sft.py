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

# accelerate launch --multi_gpu --num_processes=5 ds_sft.py

# Load environment variables
load_dotenv()

MAX_LENGTH = 8192

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
        text = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively.<|eot_id|><|start_header_id|>user<|end_header_id|>
{trace['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{trace['reasoning']}<|eot_id|>"""
        
        # Tokenize with explicit max length
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        formatted_data.append({"text": text})
    
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

    
    # Update training arguments with modern FSDP state dict handling
    training_args = TrainingArguments(
        output_dir="./sft_llama_70b_4bit_lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        optim="paged_adamw_32bit",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
        # Add explicit gradient checkpointing settings
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # After loading the model, explicitly set use_cache to False
    model.config.use_cache = False
    
    # Login to Hugging Face using API key from .env
    try:
        huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
        login(token=huggingface_token)
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")
    
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
            wandb.init(project="protein-sft", name="4096_seq_len")
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

    # Define the templates for completion-only training
    instruction_template = "<|start_header_id|>system<|end_header_id|>"
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    
    # Create the completion-only collator
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Load and process dataset
    train_dataset = load_sft_data("data/mega_cot.json", tokenizer)

    # Initialize trainer with the collator
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=training_args,
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Start training
    trainer.train()
    
    # Save final model
    model.save_pretrained("llama_70b_4bit_sft_lora_model")
    tokenizer.save_pretrained("llama_70b_4bit_sft_lora_model")
    
    # At the end of training, close wandb only on main process
    if state.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    train_model()