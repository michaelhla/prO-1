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

    # Sample new AA
    new_aa = np.random.choice(valid_aas, p=probs)
    return orig_aa, pos, new_aa

def generate_reasoning(sequence: str, mutation: Tuple[str, int, str], enzyme_prompt: str, previous_steps: List[dict] = None) -> str:
    """Generate reasoning for a mutation using OpenAI API"""
    orig_aa, pos, new_aa = mutation
    
    # Build context from previous steps
    previous_context = ""
    if previous_steps:
        previous_context = "\nPrevious mutation steps:\n"
        for i, step in enumerate(previous_steps, 1):
            mut = step.get("correct_mutation")
            previous_context += f"%%MUTATION_{i}%%: {mut['from_aa']}{mut['position']}{mut['to_aa']}\n"
            previous_context += f" %%REASONING_{i}%%: {mut['reasoning']}\n"
    
    prompt = f"""Given the following task and prior modifications made by an expert protein engineer:

USER: {enzyme_prompt}

HISTORY: {previous_context}

Provide reasoning for why mutating position {pos} from {orig_aa} to {new_aa} next could be beneficial.  ****REMEMBER TO USE YOUR KNOWLEDGE OF THIS SPECIFIC ENZYME AND REACTION****. Keep your explanation concise and focused on the scientific reasoning. ONLY OUTPUT THE TEXT REASONING, NO OTHER TEXT OR LABELS."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert protein engineer with years of experience optimizing protein activity with rational design."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    reasoning = response.choices[0].message.content
    print(reasoning)
    return reasoning

def generate_initial_mutations(sequence: str, n_mutations: int = None) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Generate n_mutations random mutations from the original sequence"""
    if n_mutations is None:
        n_mutations = random.randint(3, 5)
    
    mutations = []
    mutated_seq = sequence
    positions_used = set()
    
    while len(mutations) < n_mutations:
        # Keep sampling until we get an unused position
        while True:
            orig_aa, pos, new_aa = sample_mutation(sequence)
            if pos not in positions_used:
                break
        
        positions_used.add(pos)
        mutations.append((pos, orig_aa, new_aa))
        mutated_seq = mutated_seq[:pos] + new_aa + mutated_seq[pos+1:]
    
    return mutated_seq, mutations

def generate_incorrect_mutations(sequence: str, correct_pos: int, correct_aa: str) -> List[Tuple[int, str, str]]:
    """Generate incorrect mutations, both at the correct position and at other positions"""
    incorrect_mutations = []
    positions_used = {correct_pos}
    
    # Generate incorrect mutation at the correct position
    while True:
        orig_aa, pos, new_aa = sample_mutation(sequence)
        if pos == correct_pos and new_aa != correct_aa:
            incorrect_mutations.append((pos, orig_aa, new_aa))
            break
    
    # Generate 2-3 mutations at other positions
    n_other = random.randint(2, 3)
    while len(incorrect_mutations) < n_other + 1:
        orig_aa, pos, new_aa = sample_mutation(sequence)
        if pos not in positions_used:
            positions_used.add(pos)
            incorrect_mutations.append((pos, orig_aa, new_aa))
    
    return incorrect_mutations

def create_mutation_traces(transformed_data: Dict[str, Dict], n_traces: int = 100, n_mutations: int = 5):
    """Create dataset of mutation traces with correct and incorrect paths"""
    dataset = {
        "metadata": {
            "description": "Protein mutation traces dataset with correct and incorrect paths",
            "n_traces": n_traces,
            "n_mutations": n_mutations
        },
        "traces": []
    }
    
    for enzyme_id, enzyme_data in transformed_data.items():
        sequence = enzyme_data.get('sequence')
        if not sequence:
            continue

        # Get reaction, substrates and products from first reaction if available
        if enzyme_data.get('reaction'):
            reaction = random.choice(enzyme_data['reaction'])
            substrates = reaction['substrates'] if reaction else ['Unknown']
            products = reaction['products'] if reaction else ['Unknown']
        else:
            reaction = 'Unknown'
            substrates = 'Unknown' 
            products = 'Unknown'

        # Get metals/ions if available, otherwise use Unknown
        if enzyme_data.get('metal_ions'):
            metal_ions = random.choice(enzyme_data['metal_ions'])
        else:
            metal_ions = ['None']

        # Format known mutations and effects section if available
        known_mutations_text = ""
        if enzyme_data.get('engineering'):
            known_mutations_text = "KNOWN MUTATIONS AND EFFECTS:\n" + ''.join([
                f"- {mut['mutation']}: {mut['effect']}\n" 
                for mut in enzyme_data.get('engineering', [])
            ])
            
        for _ in range(n_traces):
            # Generate initial perturbed sequence
            perturbed_seq, mutations = generate_initial_mutations(sequence, n_mutations)
            
            # Generate a prompt for this enzyme trace
            enzyme_prompt = f"""You are an expert protein engineer. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data.get('name', 'Unknown')}
EC NUMBER: {enzyme_data.get('ec_number', 'Unknown')}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data.get('general_information', 'No additional information available')}
SUBSTRATES: {', '.join(substrates)}
PRODUCTS: {', '.join(products)}
METALS/IONS: {', '.join(metal_ions)}
{known_mutations_text}


Propose a few mutations that will optimize enzymatic activity given the substrates and products above. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects protein structure and function
2. The chemical properties of the amino acids and substrates/products
3. The position's importance in the protein sequence

For each mutation you propose, provide clear, scientific reasoning for why the mutation would be beneficial, ****USE YOUR KNOWLEDGE OF THIS SPECIFIC ENZYME AND REACTION****. Keep your response to under 100 words."""

            trace = {
                "enzyme_id": enzyme_id,
                "original_sequence": sequence,
                "perturbed_sequence": perturbed_seq,
                "initial_mutations": mutations,
                "prompt": enzyme_prompt,
                "enzyme_data": {
                    "name": enzyme_data.get('enzyme_name'),
                    "ec_number": enzyme_data.get('ec_number'),
                    "organism": enzyme_data.get('organism'),
                    "substrates": substrates,
                    "products": products,
                    "metal_ions": metal_ions,
                    "engineering": enzyme_data.get('engineering', [])
                },
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
            # Exit after first sequence as example
            if len(dataset["traces"]) >= 1:
                break
            
    # Save dataset
    Path("data").mkdir(exist_ok=True)
    with open("data/mutation_traces.json", "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    # Load transformed BRENDA data
    with open('data/transformed_brenda.json', 'r') as f:
        transformed_data = json.load(f)

    
    create_mutation_traces(transformed_data, n_traces=2, n_mutations=3)  # n per sequence

