import json
import torch
from pathlib import Path
from tqdm import tqdm
from together import Together
from stability_reward import StabilityRewardCalculator
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Initialize Together client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Initialize stability calculator on last GPU if multiple GPUs available
num_gpus = torch.cuda.device_count()
reward_device = torch.device(f"cuda:{num_gpus - 1}" if num_gpus > 0 else "cpu")
stability_calculator = StabilityRewardCalculator(device=reward_device)

def calculate_reward(completion, sequence, orig_stab):
    """Calculate reward for a completion using stability calculator"""
    try:
        reward = 0.0
        
        # Calculate reward for thinking section length
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        if think_match:
            think_text = think_match.group(1)
            think_tokens = len(think_text.split())
            token_reward = torch.exp(-((think_tokens - 3000)**2)/(2*1000**2)).item()
            reward += token_reward

        # Extract modified sequence
        sequence_match = re.search(r'\\boxed{(.*?)}', completion)
        if not sequence_match:
            return reward
            
        modified_sequence = sequence_match.group(1).strip()

        # Calculate edit distance reward
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

        edit_dist = levenshtein_distance(sequence, modified_sequence)
        if edit_dist <= 10:
            reward += 0.3

        # Calculate stability reward
        with torch.cuda.device(reward_device):
            modified_score = stability_calculator.calculate_stability(modified_sequence)
            stab_calc = -((modified_score - orig_stab) / abs(orig_stab)) * 100

        if stab_calc:
            reward += 0.5
        if stab_calc > 0.0:
            reward += 1.0

        return reward, modified_score

    except Exception as e:
        print(f"Error calculating reward: {e}")
        return reward, None

def generate_completions(dataset_path="data/train_dataset.json", k=5, output_path="data/r1-gen-sft.json"):
    """Generate k completions for each prompt and calculate rewards"""
    
    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    augmented_data = []
    
    # Process each example
    for example in tqdm(data):
        prompt = example["prompt"]
        original_sequence = example.get("sequences", "")
        
        orig_stab = example.get("orig_stabs", 0.0)
        
        completions = []
        for _ in range(k):
            try:
                response = client.chat.completions.create(
                    model="deepseek-ai/deepseek-r1",
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                
                completion = response.choices[0].message.content
                reward, stability = calculate_reward(completion, original_sequence, orig_stab)
                
                completions.append({
                    "completion": completion,
                    "reward": reward,
                    "stability_score": stability if stability is not None else None
                })
                
            except Exception as e:
                print(f"Error generating completion: {e}")
                continue
        
        # Add to augmented dataset
        augmented_data.append({
            "prompt": prompt,
            "original_sequence": original_sequence,
            "original_stability": orig_stab,
            "completions": completions
        })
        
        # Periodically save progress
        if len(augmented_data) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump({"traces": augmented_data}, f, indent=2)
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump({"traces": augmented_data}, f, indent=2)

if __name__ == "__main__":
    generate_completions()
