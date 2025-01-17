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

def generate_reasoning(sequence: str, mutation: Tuple[str, int, str], enzyme_prompt: str, previous_steps: List[dict] = None) -> str:
    """Generate reasoning for a mutation using OpenAI API"""
    orig_aa, pos, new_aa = mutation
    
    # Build context from previous steps as conversation
    messages = [
        {"role": "system", "content": "You are an expert protein engineer with years of experience optimizing protein activity with rational design. You are helping guide a protein optimization process step by step."},
        {"role": "user", "content": enzyme_prompt}
    ]
    
    if previous_steps:
        for step in previous_steps:
            correct_mut = step["correct_mutation"]
            assistant_msg = f"For the next optimization, we should mutate position {correct_mut['position']} from {correct_mut['from_aa']} to {correct_mut['to_aa']}. {correct_mut['reasoning']}"
            user_msg = "Select the next mutation we should make and explain your reasoning. Keep your response to under 100 words. If there are no further optimizations to make, return <DONE> with no explanation."
            
            if step != previous_steps[-1]:
                messages.extend([
                    {"role": "assistant", "content": assistant_msg},
                    {"role": "user", "content": user_msg}
                ])  
            else:
                messages.extend([
                    {"role": "assistant", "content": assistant_msg},
                    {"role": "user",
                    "content": f"Given the modifications above and the respective reasoning, generate reasoning for the next mutation that has been selected: {orig_aa}{pos}{new_aa}"
                }])
    else: 
        messages.append({
            "role": "user",
            "content": f"Generate reasoning for the mutation that has been selected: {orig_aa}{pos}{new_aa}"
        })

    
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.9,
        max_tokens=100
    )
    
    reasoning = response.choices[0].message.content
    return reasoning

def generate_initial_mutations(sequence: str, n_mutations: int = None) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Generate n_mutations random mutations from the original sequence"""
    if n_mutations is None:
        n_mutations = random.randint(3, 5)
    
    mutations = []
    mutated_seq = sequence
    positions_used = set()
    
    while len(mutations) < n_mutations:
        pos = random.randint(0, len(sequence)-1)
        if pos in positions_used:
            continue
            
        orig_aa = sequence[pos]
        # Get valid mutations for this position
        scores = []
        valid_aas = []
        for aa in valid_aa:
            if aa != orig_aa:
                scores.append(blosum62[orig_aa][aa])
                valid_aas.append(aa)
                
        scores = np.array(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        
        new_aa = np.random.choice(valid_aas, p=probs)
        mutations.append((pos, orig_aa, new_aa))
        positions_used.add(pos)
        mutated_seq = mutated_seq[:pos] + new_aa + mutated_seq[pos+1:]
    
    return mutated_seq, mutations

def generate_incorrect_mutations(sequence: str, correct_pos: int, correct_aa: str) -> List[Tuple[int, str, str]]:
    """Generate incorrect mutations, both at the correct position and at other positions"""
    incorrect_mutations = []
    
    # Generate incorrect mutation at the correct position
    orig_aa = sequence[correct_pos]
    valid_incorrect = [aa for aa in valid_aa if aa != correct_aa and aa != orig_aa]
    incorrect_aa = random.choice(valid_incorrect)
    incorrect_mutations.append((correct_pos, orig_aa, incorrect_aa))
    
    # Generate 2-3 mutations at other positions
    n_other = random.randint(2, 3)
    positions_used = {correct_pos}
    
    while len(incorrect_mutations) < n_other + 1:
        pos = random.randint(0, len(sequence)-1)
        if pos in positions_used:
            continue
            
        orig_aa = sequence[pos]
        new_aa = random.choice([aa for aa in valid_aa if aa != orig_aa])
        incorrect_mutations.append((pos, orig_aa, new_aa))
        positions_used.add(pos)
    
    return incorrect_mutations

def create_mutation_traces(sequences: List[str], n_traces: int = 100):
    """Create dataset of mutation traces with correct and incorrect paths"""
    dataset = {
        "metadata": {
            "description": "Protein mutation traces dataset with correct and incorrect paths",
            "n_traces": n_traces
        },
        "traces": []
    }
    
    for sequence in sequences:
        for _ in range(n_traces):
            # Generate initial perturbed sequence
            perturbed_seq, mutations = generate_initial_mutations(sequence)
            
            # Generate a prompt for this enzyme trace
            enzyme_prompt = f"""You are an expert protein engineer. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

            ENZYME SEQUENCE: {sequence}
            GENERAL INFORMATION: {information}
            SUBSTRATES: {substrates}
            PRODUCTS: {products}
            
Your task is to analyze potential mutations and determine which ones will optimize activity given the substrates and products. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects protein structure and function
2. The chemical properties of the amino acids involved
3. The position's importance in the protein sequence

Provide clear, scientific reasoning for why each mutation would or would not be beneficial. Keep your response to under 100 words."""

            trace = {
                "original_sequence": sequence,
                "perturbed_sequence": perturbed_seq,
                "initial_mutations": mutations,
                "prompt": enzyme_prompt,
                "steps": []
            }
            
            # Create correct path back to original sequence
            current_seq = perturbed_seq
            previous_steps = []
            for pos, orig_target_aa, current_aa in mutations:
                # Generate reasoning for correct mutation
                mutation = (current_aa, pos, orig_target_aa)
                reasoning = generate_reasoning(current_seq, mutation, enzyme_prompt, previous_steps)
                
                # Generate incorrect mutations
                incorrect_mutations = generate_incorrect_mutations(current_seq, pos, orig_target_aa)
                
                step = {
                    "current_sequence": current_seq,
                    "correct_mutation": {
                        "position": pos,
                        "from_aa": current_aa,
                        "to_aa": orig_target_aa,
                        "reasoning": reasoning
                    },
                    "incorrect_mutations": [
                        {
                            "position": p,
                            "from_aa": orig,
                            "to_aa": new,
                            "reasoning": generate_reasoning(current_seq, (orig, p, new), enzyme_prompt, previous_steps)
                        }
                        for p, orig, new in incorrect_mutations
                    ]
                }
                
                previous_steps.append(step)
                trace["steps"].append(step)
                current_seq = current_seq[:pos] + orig_target_aa + current_seq[pos+1:]
            
            dataset["traces"].append(trace)
            
    # Save dataset
    Path("data").mkdir(exist_ok=True)
    with open("data/mutation_traces.json", "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    # Load transformed BRENDA data
    with open('data/transformed_brenda.json', 'r') as f:
        transformed_data = json.load(f)
    
    sequences = [entry['sequence'] for entry in transformed_data.values() if 'sequence' in entry]
    sequences = [sequences[0]]  # For testing, just use first sequence
    
    create_mutation_traces(sequences, n_traces=2)  # Adjust n_traces as needed

