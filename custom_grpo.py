# Python standard library imports
import os
import json
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import time

# PyTorch imports
import torch
from accelerate import PartialState
from accelerate.utils import set_seed

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType
)

# Other imports
import wandb
from dotenv import load_dotenv

NUM_EPOCHS = 5
MAX_INPUT_LENGTH = 8000
MAX_OUTPUT_LENGTH = 2
RUN_NAME = "grpo-full-model"

def build_model(model_name_or_path, tokenizer, device):
    """
    Loads the full model on each GPU with 4-bit quantization and LoRA, wrapped in DDP
    """
    def print_gpu_memory(step_name):
        if torch.cuda.is_available():
            gpu_idx = proc_state.local_process_index
            allocated = torch.cuda.memory_allocated(gpu_idx) / (1024 * 1024 * 1024)
            reserved = torch.cuda.memory_reserved(gpu_idx) / (1024 * 1024 * 1024)
            max_reserved = torch.cuda.max_memory_reserved(gpu_idx) / (1024 * 1024 * 1024)
            print(f"\nGPU {gpu_idx} Memory at {step_name}:")
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Reserved:  {reserved:.2f} GB")
            print(f"Max Reserved: {max_reserved:.2f} GB")
            
    proc_state = PartialState()
    local_rank = proc_state.local_process_index
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    print_gpu_memory("start")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load full model on each GPU
    print(f"Loading model from: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    print_gpu_memory("after model load")

    model = prepare_model_for_kbit_training(model)
    print_gpu_memory("after prepare_model_for_kbit_training")

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
    print_gpu_memory("after LoRA setup")
    
    # Wrap model in DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    print_gpu_memory("after DDP wrap")

    # Try to force garbage collection
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print_gpu_memory("after garbage collection")

    return model

class CustomGRPOTrainer:
    def __init__(self, model, tokenizer, beta=0.1, lr=1e-4, num_generations=2):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.num_generations = num_generations
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.proc_state = PartialState()

    ## TODO make completions much faster, vllm or kv cache
    def generate_completions(self, prompts, max_new_tokens=MAX_OUTPUT_LENGTH):
        """Generate multiple completions for each prompt and return sequences and logits"""
        with torch.no_grad():
            # Move input tensors to the same device as the model
            device = next(self.model.parameters()).device
            input_ids = prompts['input_ids'].to(device)
            attention_mask = prompts['attention_mask'].to(device)
            
            # Generate with model and get logits directly
            outputs = self.model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=self.num_generations,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Concatenate logits from the generation output
            logits = torch.cat(outputs.scores, dim=1)  # Concatenate along the sequence dimension
            
            # Ensure input_ids are included in outputs
            outputs.input_ids = input_ids
            outputs.logits = logits  # Assign logits directly to outputs
            
        return outputs

    ## TODO: implement the stability calc reward 
    def compute_reward(self, outputs):
        """Compute reward for each generated sequence"""
        # Assuming outputs.sequences is a tensor with shape (batch_size * num_generations, sequence_length)
        print(outputs)
        rewards = torch.ones(outputs.sequences.size(0), device=self.model.device)  # Placeholder: all rewards set to 1
        return rewards
    
    def compute_advantage(self, original_scores, generated_scores): 
        """Compute normalized advantages within each group of generations"""
        if generated_scores.dim() == 1:
            generated_scores = generated_scores.view(-1, self.num_generations)
            
        advantages = []
        for prompt_scores in generated_scores:
            mean_score = prompt_scores.mean()
            std_score = prompt_scores.std()
            prompt_advantages = (prompt_scores - mean_score) / (std_score + 1e-8)
            advantages.append(prompt_advantages)
            
        return torch.stack(advantages)


    ## TODO: fix the kl divergence calculation, step by step walking through, batch issue
    def compute_kl_divergence(self, outputs, logits, input_ids, attention_mask):
        """Compute KL divergence between current LoRA and base model"""
        # Store current LoRA weights
        original_weights = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                original_weights[name] = param.data.clone()
                param.data.zero_()
        
        # Get base model logits (with LoRA zeroed)
        with torch.no_grad():
            base_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            ).logits
        
        # Restore LoRA weights
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.data.copy_(original_weights[name])
        
        # Ensure logits have the same shape
        if logits.size() != base_logits.size():
            raise ValueError(f"Logits shape mismatch: lora_probs {logits.size()}, base_probs {base_logits.size()}")
        
        # Compute KL divergence
        lora_probs = torch.softmax(logits, dim=-1)
        base_probs = torch.softmax(base_logits, dim=-1)
        kl = torch.sum(lora_probs * (torch.log(lora_probs + 1e-10) - torch.log(base_probs + 1e-10)), dim=-1)
        
        return kl

    def train_step(self, batch):
        """Single training step implementing GRPO"""
        start_time = time.time()
        
        def log_step(step_name, start_t):
            gpu_idx = self.proc_state.local_process_index
            elapsed = time.time() - start_t
            allocated = torch.cuda.memory_allocated(gpu_idx) / (1024 * 1024 * 1024)
            print(f"GPU {gpu_idx} | {step_name}: {elapsed:.2f}s | Memory: {allocated:.2f}GB")
            return time.time()
        
        # Update metrics
        def update_metrics(metrics_dict, new_metrics, step_name, start_t):
            metrics_dict.update(new_metrics)
            current_t = time.time()
            print(f"GPU {self.proc_state.local_process_index} | Metrics update {step_name}: {current_t - start_t:.2f}s")
            return current_t

        self.model.train()
        current_time = log_step("Model to train mode", start_time)
        
        # Get completions and logits in one step
        outputs = self.generate_completions(batch)
        current_time = log_step("Generation", current_time)
        
        # Get original scores from batch
        original_scores = batch['orig_stabs']
        current_time = log_step("Score extraction", current_time)
        
        # Compute advantages
        generated_scores = self.compute_reward(outputs)
        advantages = self.compute_advantage(original_scores, generated_scores)
        current_time = log_step("Reward computation", current_time)
        
        # Compute KL divergence
        attention_mask = batch['attention_mask']
        kl = self.compute_kl_divergence(outputs, outputs.logits, outputs.input_ids, attention_mask)
        current_time = log_step("KL divergence", current_time)
        
        # Compute policy ratio and loss
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        with torch.no_grad():
            log_probs_no_grad = torch.log_softmax(outputs.logits, dim=-1)
        
        policy_ratio = torch.exp(log_probs - log_probs_no_grad)
        current_time = log_step("Policy ratio", current_time)
        
        # Compute loss
        ## TODO: fix the grpo calculation that claude fucked 
        sequence_length = outputs.logits.size(1)
        token_losses = -(
            policy_ratio * advantages.unsqueeze(-1).expand(-1, sequence_length, -1) - 
            self.beta * kl.unsqueeze(-1).expand(-1, sequence_length, -1)
        )
        
        loss = token_losses.mean()
        current_time = log_step("Loss computation", current_time)
        
        # Update metrics
        metrics = defaultdict(float)
        current_time = update_metrics(metrics, {
            'reward_mean': generated_scores.mean().item(),
            'reward_std': generated_scores.std().item(),
            'kl': kl.mean().item(),
            'loss': loss.item()
        }, "core metrics", current_time)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        current_time = log_step("Backward pass", current_time)
        
        self.optimizer.step()
        current_time = log_step("Optimizer step", current_time)
        
        total_time = time.time() - start_time
        metrics['step_time'] = total_time
        print(f"GPU {self.proc_state.local_process_index} | Total step time: {total_time:.2f}s")
        
        return metrics

    def fit(self, dataloader, epochs=1):
        """Training loop"""
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_metrics = defaultdict(float)
            
            for step, batch in enumerate(dataloader):
                batch_start = time.time()
                print(f"GPU {self.proc_state.local_process_index} | Starting batch load: {time.time() - batch_start:.2f}s")
                
                step_metrics = self.train_step(batch)
                step_time = time.time() - batch_start
                print(f"GPU {self.proc_state.local_process_index} | Full batch processing time: {step_time:.2f}s")
                
                # Update epoch metrics
                metrics_start = time.time()
                for k, v in step_metrics.items():
                    epoch_metrics[k] += v
                metrics_time = time.time() - metrics_start
                print(f"GPU {self.proc_state.local_process_index} | Epoch metrics update time: {metrics_time:.2f}s")
                
                if step % 10 == 0:
                    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in step_metrics.items())
                    print(f"Epoch {epoch} | Step {step} | {metrics_str}")
            
            # Average epoch metrics
            avg_start = time.time()
            for k in epoch_metrics:
                epoch_metrics[k] /= len(dataloader)
            avg_time = time.time() - avg_start
            
            epoch_time = time.time() - epoch_start
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
            print(f"GPU {self.proc_state.local_process_index} | Epoch {epoch} completed in {epoch_time:.2f}s | Metrics averaging time: {avg_time:.2f}s | {metrics_str}")

def create_rl_dataloader(train_dataset_path, tokenizer, batch_size=2):
    """Creates a DataLoader for RL training"""
    # Load the JSON dataset
    with open(train_dataset_path, 'r') as f:
        dataset = json.load(f)

    class RLDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            prompt = item['prompt']
            original_score = item['orig_stabs']  # Original reward signal

            # Tokenize the prompt
            encodings = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            )

            return {
                'input_ids': encodings['input_ids'].squeeze(0),
                'attention_mask': encodings['attention_mask'].squeeze(0),
                'orig_stabs': torch.tensor(original_score, dtype=torch.float),
                'prompt': prompt  # Keep original prompt for reference
            }

    def collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item['attention_mask'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        original_scores = torch.stack([item['orig_stabs'] for item in batch])
        prompts = [item['prompt'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_stabs': original_scores,
            'prompts': prompts
        }

    dataset = RLDataset(dataset, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    
    return dataloader

##################################################################
# 5. Main Function
##################################################################
def main(rank, world_size, model_path, epochs=1):
    device = torch.device(f"cuda:{rank}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", trust_remote_code=True, model_max_length=MAX_INPUT_LENGTH)
    
    # Use EOS token for padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build the model
    model = build_model(model_path, tokenizer, device)

    # Create trainer
    trainer = CustomGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=1e-4
    )

    # Create a dummy DataLoader (replace with real dataset)
    dataloader = create_rl_dataloader('data/train_dataset.json', tokenizer, batch_size=2)

    # Train
    trainer.fit(dataloader, epochs=epochs)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Example usage:
    torchrun --nnodes 1 --nproc_per_node 8 custom_grpo.py
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    proc_state = PartialState()
    rank = proc_state.local_process_index
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if proc_state.is_main_process:
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

    main(
        rank=rank,
        world_size=world_size,
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        epochs=NUM_EPOCHS
    )