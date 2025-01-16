import torch
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
import random
from Bio.Substitution import BLOSUM62
import openai
import os
## use 4o mini to generate reasoning for each mutation step 
## generate mutations by sampling from the inverse BLOSUM matrix 
## create SFT dataset by sampling from the reasoning steps 
## get reward for each mutation and create preference dataset 
## 50 million training tokens

## EXAMPLE FORMAT: 
## SEQUENCE: {sequence} 
## MUTATION: K293A 
## REASONING: {reasoning}

## SFT dataset is set of responses given prior mutations steps and system prompt

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def sample_mutation(sequence: str, blosum: BLOSUM62) -> Tuple[str, int, str]:
    """Sample mutation position and new amino acid using inverse BLOSUM62"""
    pos = random.randint(0, len(sequence)-1)
    orig_aa = sequence[pos]
    
    # Get BLOSUM scores for this AA and invert them
    scores = np.array([blosum[orig_aa, aa] for aa in blosum.alphabet])
    inv_scores = 1.0 / (scores + 7) # Add offset to avoid division by zero
    probs = inv_scores / inv_scores.sum()
    
    new_aa = np.random.choice(list(blosum.alphabet), p=probs)
    return orig_aa, pos, new_aa

def generate_reasoning(sequence: str, mutation: Tuple[str, int, str]) -> str:
    """Generate reasoning for a mutation using OpenAI API"""
    orig_aa, pos, new_aa = mutation
    prompt = f"""Explain why mutating position {pos} from {orig_aa} to {new_aa} in the sequence {sequence} could be beneficial given the objective of improving activity. Keep your explanation concise and to the point."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert protein engineer with years of experience optimizing protein activity."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    reasoning = response.choices[0].message.content
    return reasoning
def create_datasets(sequences: List[str], num_mutations: int = 1000):
    """Create SFT and preference datasets in OpenAI format"""
    blosum = BLOSUM62()
    
    mutation_data = []
    
    for sequence in sequences:
        for _ in range(num_mutations):
            # Generate mutation and reasoning
            mutation = sample_mutation(sequence, blosum)
            reasoning = generate_reasoning(sequence, mutation)
            
            # Create mutated sequence
            orig_aa, pos, new_aa = mutation
            mutated_seq = sequence[:pos] + new_aa + sequence[pos+1:]
            
            # Store mutation data as tuple
            mutation_data.append((
                (orig_aa, pos, new_aa),  # mutation details
                reasoning,               # generated reasoning
                sequence,               # original sequence
                mutated_seq            # new sequence
            ))
    
    # Save mutation data
    Path("data").mkdir(exist_ok=True)
    with open("data/mutations.json", "w") as f:
        json.dump(mutation_data, f, indent=2)

if __name__ == "__main__":
    # Load initial sequences from BRENDA
    from brenda_clean import load_sequences
    sequences = load_sequences()
    
    # Generate datasets targeting 50M tokens
    create_datasets(sequences, num_mutations=25000) # Assuming ~2000 tokens per example

