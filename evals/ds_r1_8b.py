import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from stability_reward import StabilityRewardCalculator

load_dotenv()

# Initialize model and tokenizer
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Initialize stability calculator
stability_calculator = StabilityRewardCalculator()

def get_stability_score(sequence: str) -> float:
    """Calculate protein stability score using ESMFold and PyRosetta"""
    return stability_calculator.calculate_stability(sequence)

def propose_mutations(sequence: str) -> List[str]:
    """Get mutation proposals from DeepSeek"""
    prompt = f"""Given this enzyme sequence: {sequence}

Please propose 3-7 mutations that would increase the stability of this enzyme. 
Format your response as a list of mutations in the format 'X123Y' where:
- X is the original amino acid
- 123 is the position (1-indexed)
- Y is the proposed new amino acid

Copy the sequence with the mutations applied below. Wrap the sequence in <sequence> tags."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    mutations = response.strip().split('\n')
    return [m.strip() for m in mutations if m.strip() and len(m.strip()) > 0 and 'X' in m and 'Y' in m]

def apply_mutations(sequence: str, mutations: List[str]) -> str:
    """Apply a list of mutations to a sequence"""
    seq_list = list(sequence)
    for mutation in mutations:
        # Parse mutation format 'X123Y'
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        pos = int(mutation[1:-1]) - 1  # Convert to 0-indexed
        
        if seq_list[pos] != orig_aa:
            print(f"Warning: Expected {orig_aa} at position {pos+1} but found {seq_list[pos]}")
            continue
            
        seq_list[pos] = new_aa
    
    return ''.join(seq_list)

def main():
    # Load enzyme sequences
    with open('data/transformed_brenda.json', 'r') as f:
        enzymes = json.load(f)
    
    # Randomly select 100 enzymes
    selected_enzymes = random.sample(list(enzymes.items()), 100)
    
    results = []
    
    for enzyme_id, data in tqdm(selected_enzymes, desc="Processing enzymes"):
        sequence = data['sequence']
        original_stability = get_stability_score(sequence)
        
        try:
            # Get mutation proposals
            proposed_mutations = propose_mutations(sequence)
            
            # Apply mutations
            mutated_sequence = apply_mutations(sequence, proposed_mutations)
            
            # Calculate new stability
            new_stability = get_stability_score(mutated_sequence)
            
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'proposed_mutations': proposed_mutations,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability > original_stability
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            continue
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ds_r1_stability_mutations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nResults summary:")
    print(f"Number of enzymes processed: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Max stability improvement: {max(r['stability_change'] for r in results):.3f}")

if __name__ == "__main__":
    main()
