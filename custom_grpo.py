import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaForCausalLM, LlamaTokenizer

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
    Loads the LLaMA model from a local directory, then optionally
    wraps it with FSDP for memory-efficient distributed training.
    """
    # Load model
    print(f"Loading model from: {model_name_or_path}")
    model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    model = model.to(device)

    if use_fsdp:
        # Wrap with FSDP. You can choose advanced configs here.
        # If your model is huge, consider using sharding strategies,
        # mixed precision, etc.
        model = FSDP(model)
        # Alternatively, you can do more advanced per-module wrapping:
        # model = FSDP(
        #     wrap(model),
        #     sharding_strategy=ShardingStrategy.FULL_SHARD,
        #     cpu_offload=CPUOffload(offload_params=False),
        #     auto_wrap_policy=some_custom_wrap_policy,
        #     mixed_precision=some_mixed_precision_config,
        # )
    else:
        # Or use standard DistributedDataParallel
        model = DDP(model, device_ids=[device])
    
    return model


##################################################################
# 3. A Simple Custom Trainer Class
##################################################################
class CustomGRPOTrainer:
    """
    A stand-in class to demonstrate a custom training loop for your
    "GRPO" method. The logic below is a stub; adapt as needed.
    """
    def __init__(self, model, tokenizer, lr=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, batch):
        """
        A single step of training. In a real setting, you'd:
        1. Prepare the input batch (tokens, masks, etc.).
        2. Forward through the model.
        3. Compute loss.
        4. Backprop + optimizer step.
        """
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, dataloader, epochs=1):
        """
        Main training loop for demonstration.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumerate(dataloader):
                loss_val = self.train_step(batch)
                total_loss += loss_val
                if step % 10 == 0:
                    # Print average loss every few steps
                    avg_loss = total_loss / (step + 1)
                    print(f"Epoch {epoch} | Step {step} | Avg Loss: {avg_loss:.4f}")

            print(f"Epoch {epoch} completed. Avg Loss: {total_loss / len(dataloader):.4f}")


##################################################################
# 4. Example DataLoader
##################################################################
def create_dummy_dataloader(tokenizer, batch_size=2, total_samples=100):
    """
    Creates a toy DataLoader to demonstrate the training loop.
    In practice, replace with a real dataset + sampler.
    """
    # We'll just generate random tokens for demonstration.
    input_texts = ["Hello world!"] * total_samples

    # Tokenize
    encodings = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    # We pretend the next token is always the label
    encodings["labels"] = encodings["input_ids"].clone()

    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings["labels"],
    )

    def collate_fn(batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

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

    # Create a dummy DataLoader (replace with real dataset)
    dataloader = create_dummy_dataloader(tokenizer, batch_size=2, total_samples=16)

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
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--model_path", type=str, required=True)
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
        model_path=args.model_path,
        epochs=args.epochs,
        use_fsdp=args.use_fsdp
    )