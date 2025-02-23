import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
import json
from collections import defaultdict

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

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

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
        model = FSDP(
            model,
            # auto_wrap_policy=auto_wrap_policy,
        )
        
        # Print FSDP sharding info
        if proc_state.is_main_process:
            print("\nFSDP Sharding Information:")
            print("=" * 50)
            
            def print_sharding_info(module, prefix=""):
                if isinstance(module, FSDP):
                    print(f"{prefix}FSDP Wrapped: {module._get_name()}")
                    print(f"{prefix}├── Rank: {proc_state.local_process_index}")
                    print(f"{prefix}├── World Size: {proc_state.world_size}")
                    # Get number of parameters in this shard
                    num_params = sum(p.numel() for p in module.parameters())
                    print(f"{prefix}└── Parameters in shard: {num_params:,}")
                
                for name, child in module.named_children():
                    print_sharding_info(child, prefix + "    ")
            
            print_sharding_info(model)
            
            # Print total parameters across all ranks
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nTotal parameters across all ranks: {total_params:,}")
            print("=" * 50)
    
    else:
        print('FSDP SETUP FAILED')
    
    return model



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
# 3. A Simple Custom Trainer Class
##################################################################
class CustomGRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta=0.1, lr=1e-4, num_generations=4):
        """
        Args:
            model: The model to be trained
            ref_model: Reference model (frozen) for KL divergence calculation
            tokenizer: Tokenizer for text generation
            beta: KL penalty coefficient
            lr: Learning rate
            num_generations: Number of generations per prompt (G in the paper)
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.num_generations = num_generations
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    def generate_completions(self, prompts, max_length=128):
        """Generate multiple completions for each prompt"""
        all_completions = []
        
        for _ in range(self.num_generations):
            outputs = self.model.generate(
                input_ids=prompts['input_ids'],
                attention_mask=prompts['attention_mask'],
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )
            all_completions.append(outputs)
        
        return all_completions
    
    def compute_reward(self, completion):
        """
        Compute reward for a single completion
        """
        # Placeholder for reward computation
        return torch.randn(completion.sequences.size(0))
    
    def compute_advantage(self, original_scores, generated_scores):
        """
        Compute normalized advantages within each group of generations for a prompt
        Args:
            original_scores: shape [batch_size]
            generated_scores: shape [batch_size, num_generations]
        """
        # Reshape to [batch_size, num_generations]
        if generated_scores.dim() == 1:
            generated_scores = generated_scores.view(-1, self.num_generations)
            
        advantages = []
        for prompt_scores in generated_scores:
            # Normalize within this prompt's generations
            mean_score = prompt_scores.mean()
            std_score = prompt_scores.std()
            prompt_advantages = (prompt_scores - mean_score) / (std_score + 1e-8)
            advantages.append(prompt_advantages)
            
        return torch.stack(advantages)  # Returns [batch_size, num_generations]
    
    def compute_kl_divergence(self, model_logits, ref_logits):
        """Compute KL divergence between model and reference policy"""
        # Get probabilities from logits
        model_probs = torch.softmax(model_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        
        # Compute ratio π_ref/π_θ
        ratio = ref_probs / (model_probs + 1e-8)  # Add small epsilon to prevent division by zero
        
        # Compute KL divergence according to the formula
        kl = ratio - torch.log(ratio) - 1
        
        return kl.sum(dim=-1)  # Sum over vocabulary dimension
    
    def train_step(self, batch):
        """Single training step implementing GRPO"""
        self.model.train()
        self.ref_model.eval()
        
        # Generate completions
        completions = self.generate_completions(batch)
        
        # Get original scores from batch
        original_scores = batch['original_scores']
        
        # Forward pass through model and reference model
        total_loss = 0
        metrics = {
            'reward_mean': 0,
            'reward_std': 0,
            'kl': 0,
            'loss': 0
        }
        
        for completion in completions:
            # Get logits from both models
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=completion.sequences,
                    attention_mask=torch.ones_like(completion.sequences)
                )
                ref_logits = ref_outputs.logits
            
            model_outputs = self.model(
                input_ids=completion.sequences,
                attention_mask=torch.ones_like(completion.sequences)
            )
            model_logits = model_outputs.logits
            
            # Compute advantages
            generated_scores = self.compute_reward(completion)  # You'll need to implement this
            advantages = self.compute_advantage(original_scores, generated_scores)
            
            # Compute KL divergence
            kl = self.compute_kl_divergence(model_logits, ref_logits)
            
            # Compute GRPO loss according to the equation
            # L_GRPO(θ) = -(1/G) Σ (1/|o_i|) Σ [π_θ(o_i,t|q,o_i,<t) / [π_θ(o_i,t|q,o_i,<t)]_no_grad * A_i,t - β*D_KL]
            
            # Get log probabilities
            log_probs = torch.log_softmax(model_logits, dim=-1)
            
            # Compute the denominator (no_grad term)
            with torch.no_grad():
                log_probs_no_grad = torch.log_softmax(ref_logits, dim=-1)
            
            # Compute policy ratio (π_θ / [π_θ]_no_grad)
            policy_ratio = torch.exp(log_probs - log_probs_no_grad.detach())
            
            # Compute loss for each token
            sequence_length = model_logits.size(1)
            
            # For each token position, compute policy ratio * advantage - β * KL
            token_losses = -(
                policy_ratio * advantages.unsqueeze(-1).expand(-1, sequence_length, -1) - 
                self.beta * kl.unsqueeze(-1).expand(-1, sequence_length, -1)
            )
            
            # Average over sequence length (1/|o_i|) and batch (1/G)
            sequence_loss = token_losses.mean(dim=1)  # Average over sequence length
            batch_loss = sequence_loss.mean(dim=0)    # Average over batch (1/G)
            
            loss = batch_loss
            
            # Update metrics
            metrics['reward_mean'] += generated_scores.mean().item()
            metrics['reward_std'] += generated_scores.std().item()
            metrics['kl'] += kl.mean().item()
            metrics['loss'] += loss.item()
            
            total_loss += loss
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(completions)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def fit(self, dataloader, epochs=1):
        """Training loop"""
        for epoch in range(epochs):
            epoch_metrics = defaultdict(float)
            
            for step, batch in enumerate(dataloader):
                step_metrics = self.train_step(batch)
                
                # Update epoch metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k] += v
                
                if step % 10 == 0:
                    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in step_metrics.items())
                    print(f"Epoch {epoch} | Step {step} | {metrics_str}")
            
            # Average epoch metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= len(dataloader)
            
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
            print(f"Epoch {epoch} completed | {metrics_str}")


def create_rl_dataloader(train_dataset_path, tokenizer, batch_size=2):
    """
    Creates a DataLoader for RL training using prompts and scores from train_dataset.json
    """
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
            original_score = item['score']  # Original reward signal

            # Tokenize the prompt
            encodings = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            return {
                'input_ids': encodings['input_ids'].squeeze(0),
                'attention_mask': encodings['attention_mask'].squeeze(0),
                'original_score': torch.tensor(original_score, dtype=torch.float),
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
        original_scores = torch.stack([item['original_score'] for item in batch])
        prompts = [item['prompt'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'original_scores': original_scores,
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
def main(rank, world_size, model_path, epochs=1, use_fsdp=True):
    init_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Load the tokenizer
    print("Loading LLaMA tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Build the model (FSDP or DDP)
    model = build_fsdp_model(model_path, device, use_fsdp=use_fsdp)
    exit()

    # Create a dummy DataLoader (replace with real dataset)
    dataloader = create_rl_dataloader(tokenizer, batch_size=2, total_samples=16)

    # Build your custom trainer
    trainer = CustomGRPOTrainer(model, tokenizer, lr=1e-4)

    # Train
    trainer.fit(dataloader, epochs=epochs)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Example usage:
    torchrun --nproc_per_node=2 train.py --model_path=./llama_weights --epochs=1
    """
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=8)
    # parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_fsdp", action="store_true")
    args = parser.parse_args()

    # Note: Torch automatically sets LOCAL_RANK when using torchrun.
    # If you want to manually control ranks, you'd parse them differently.
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    main(
        rank=rank,
        world_size=world_size,
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        epochs=args.epochs,
        use_fsdp=args.use_fsdp
    )