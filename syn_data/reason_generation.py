import torch
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
import random
from Bio.Align import substitution_matrices
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
blosum62 = substitution_matrices.load("BLOSUM62")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

valid_aa = [aa for aa in blosum62.keys() if aa != '*']


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


def sample_mutation(sequence: str) -> Tuple[str, int, str]:
    """Sample mutation position and new amino acid using BLOSUM62 scores"""
    pos = random.randint(0, len(sequence)-1)
    orig_aa = sequence[pos]
    
    # Get BLOSUM scores for this AA
    scores = []
    valid_aas = []
    # Filter valid_aa to only include amino acids that start with orig_aa
    filtered_aa = [aa for aa in valid_aa if aa[0] == orig_aa]



    for mut_aa in filtered_aa:
        if mut_aa[1] != orig_aa:  # Skip the original AA
            scores.append(blosum62[orig_aa][mut_aa[1]])
            valid_aas.append(mut_aa[1])
            
    scores = np.array(scores, dtype=float).flatten()
    valid_aas = np.array(valid_aas).flatten()
    # Convert log-odds scores to probabilities using softmax
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum()
    
    # Ensure probs is 1D and sums to 1
    probs = probs.flatten()
    probs = probs / probs.sum()
    print(probs)
    
    # Sample new AA
    new_aa = np.random.choice(valid_aas, p=probs)
    return orig_aa, pos, new_aa

def generate_reasoning(sequence: str, mutation: Tuple[str, int, str]) -> str:
    """Generate reasoning for a mutation using OpenAI API"""
    orig_aa, pos, new_aa = mutation
    prompt = f"""An expert protein engineer selected this mutation to optimize the activity of this enzyme. Explain why mutating position {pos} from {orig_aa} to {new_aa} in the sequence {sequence} could be beneficial. Keep your explanation concise and to the point."""
    
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
def create_reasoning(sequences: List[str], num_mutations: int = 1000):
    """Create reasoning for each mutation"""    
    mutation_data = []
    
    for sequence in sequences:
        for _ in range(num_mutations):
            # Generate mutation and reasoning
            mutation = sample_mutation(sequence)
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
            print(mutation_data)
    
    # Save mutation data
    Path("data").mkdir(exist_ok=True)
    with open("data/mutations.json", "w") as f:
        json.dump(mutation_data, f, indent=2)

if __name__ == "__main__":
    # Load transformed BRENDA data
    with open('data/transformed_brenda.json', 'r') as f:
        transformed_data = json.load(f)
    
    # Extract sequences from transformed data
    sequences = [entry['sequence'] for entry in transformed_data.values() if 'sequence' in entry]
    
    sequences = [sequences[0]]

    # Generate datasets targeting 50M tokens
    create_reasoning(sequences, num_mutations=3) # Assuming ~2000 tokens per example

