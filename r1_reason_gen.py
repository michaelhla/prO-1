import json
import torch
from pathlib import Path
from tqdm import tqdm
from together import Together
from stability_reward import StabilityRewardCalculator
from dotenv import load_dotenv
import os
import re
import math
# Load environment variables
load_dotenv()

# Initialize Together client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Initialize stability calculator on last GPU if multiple GPUs available
num_gpus = torch.cuda.device_count()


def calculate_reward(completion, sequence, orig_stab, reward_device):
    """Calculate reward for a completion using stability calculator"""
    try:
        reward = 0.0
        
        # Calculate reward for thinking section length
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        if think_match:
            think_text = think_match.group(1)
            think_tokens = len(think_text.split())
            token_reward = math.exp(-((think_tokens - 3000)**2)/(2*1000**2))
            reward += token_reward

        # Extract modified sequence
        sequence_match = re.search(r'\\boxed{(.*?)}', completion)
        if not sequence_match:
            return reward, None
            
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
        return reward

def generate_completions(device, dataset_path="data/train_dataset.json", k=2, output_path="data/r1-gen-sft.json"):
    """Generate k completions for each prompt and calculate rewards"""
    cuda_device = torch.device(device)
    print(f"Using device: {cuda_device}")
    device_id = cuda_device.index if cuda_device.type == 'cuda' else 0
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Add device name to output path
    device_name = str(device).replace(':', '-')
    output_path = output_path.replace('.json', f'_{device_name}.json')
    
    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    # Calculate partition size and bounds
    start_idx = 60 if device_id == 0 else 60 + ((len(data) - 60) // num_devices) * device_id
    end_idx = 60 + ((len(data) - 60) // num_devices) * (device_id + 1) if device_id < num_devices - 1 else len(data)
    
    # Slice the dataset according to device partition
    data = data[start_idx:end_idx]
    
    augmented_data = []
    
    try:
        # Process each example
        for example in tqdm(data):
            prompt = example["prompt"]
            original_sequence = example.get("sequences", "")
            
            orig_stab = example.get("orig_stabs", 0.0)
            
            completions = []
            for _ in range(k):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-V3",
                        messages=[{"role": "user", "content": prompt}],
                        stream=False
                    )
                    
                    completion = response.choices[0].message.content
                    print(completion)
                    try:
                        reward, stability = calculate_reward(completion, original_sequence, orig_stab, reward_device=cuda_device)
                    except Exception as e:
                        print(f"Error calculating reward: {e}")
                        reward, stability = 0.0, None
                    
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
            
    except (KeyboardInterrupt, Exception) as e:
        print(f"\nInterruption detected ({type(e).__name__}). Saving current progress...")
        with open(output_path, 'w') as f:
            json.dump({"traces": augmented_data}, f, indent=2)
        print(f"Progress saved to {output_path}")
        raise

def merge_jsons(output_path="data/r1-gen-sft-final.json"):
    """Merge all JSON files with prefix 'r1-gen-sft' in the data directory"""
    base_dir = Path("data")
    merged_data = {}  # Use dict to track unique prompts
    
    # Find all files matching the pattern
    for file in base_dir.glob('r1-gen-sft*.json'):
        print(f"Processing {file}")
        with open(file, 'r') as f:
            data = json.load(f)
            for trace in data['traces']:
                prompt = trace['prompt']
                if prompt in merged_data:
                    # If prompt exists, extend the completions array
                    merged_data[prompt]['completions'].extend(trace['completions'])
                else:
                    # If new prompt, add entire trace
                    merged_data[prompt] = trace

    # Convert back to list format
    final_traces = list(merged_data.values())
    
    # Save merged result
    with open(output_path, 'w') as f:
        json.dump({"traces": final_traces}, f, indent=2)
    print(f"Merged data saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--merge', action='store_true', help='Whether to merge the results')

    args = parser.parse_args()
    if args.merge:
        merge_jsons() # TBD function to implement
        exit()

    reward_device = torch.device(f"cuda:{num_gpus - 1}" if num_gpus > 0 else "cpu")
    stability_calculator = StabilityRewardCalculator(device=args.device)
    
    generate_completions(args.device)
