import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import numpy as np
from stability_reward import StabilityRewardCalculator
# Initialize model and tokenizer
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# PPO configuration
ppo_config = PPOConfig(
    batch_size=8,
    learning_rate=1.41e-5,
    mini_batch_size=4,
    optimize_cuda_cache=True,
    gradient_accumulation_steps=1
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# Length sampler for response generation
length_sampler = LengthSampler(32, 128)

def get_stability_score(response):
    calculator = StabilityRewardCalculator()
    return calculator.calculate_stability(response)

def calculate_group_rewards(scores):
    """Calculate rewards based on group statistics"""
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores) + 1e-6  # Add small epsilon to avoid division by zero
    
    # Normalize scores relative to group statistics
    normalized_rewards = (scores - mean) / std
    return torch.tensor(normalized_rewards)

# Training loop
for epoch in range(100):
    # Generate multiple responses per prompt
    query_tensors = []
    response_tensors = []
    all_stability_scores = []
    
    prompts = ["Write a function to sort a list", "Implement binary search"]  # Example prompts
    num_responses_per_prompt = 4  # Generate multiple responses for each prompt
    
    for prompt in prompts:
        query_tensor = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate multiple responses for the same prompt
        for _ in range(num_responses_per_prompt):
            response_length = length_sampler()
            response = ppo_trainer.generate(
                query_tensor,
                max_new_tokens=response_length,
                do_sample=True,
                temperature=0.9
            )
            
            response_tensor = response[:, query_tensor.shape[1]:]
            decoded_response = tokenizer.decode(response_tensor[0])
            stability_score = get_stability_score(decoded_response)
            
            query_tensors.append(query_tensor)
            response_tensors.append(response_tensor)
            all_stability_scores.append(stability_score)
    
    # Calculate group-based rewards
    rewards = calculate_group_rewards(all_stability_scores)
    
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    print(f"Epoch {epoch}: Mean reward = {rewards.mean().item()}, Std reward = {rewards.std().item()}")
