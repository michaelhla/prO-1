import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
import random
from Bio.Align import substitution_matrices
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig


load_dotenv()  # Load environment variables from .env file
blosum62 = substitution_matrices.load("BLOSUM62")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

valid_aa = [aa for aa in blosum62.keys() if aa != '*']


## use 4o to generate reasoning for each mutation step 
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

def generate_reasoning(perturbed_sequence: str, mutations: List[Tuple[str, int, str]], enzyme_prompt: str, initial_sequence: str) -> str:
    """Generate reasoning for all mutations using OpenAI API"""
    
    # Format mutations in standard notation
    formatted_mutations = []
    for mut in reversed(mutations):  # Reverse the mutations list
        pos = mut[0]
        if mut[1] == "ins":
            formatted_mutations.append(f"{perturbed_sequence[pos]}{pos+1}ins{mut[2]}")
        elif mut[1] == "del":
            start_pos = pos
            end_pos = mut[2]
            formatted_mutations.append(f"{perturbed_sequence[start_pos]}{start_pos+1}_{perturbed_sequence[end_pos]}{end_pos+1}del")
        else:
            orig_aa, new_aa = mut[1], mut[2]
            formatted_mutations.append(f"{orig_aa}{pos+1}{new_aa}")
    
    mutations_text = "\n".join(formatted_mutations)
    
    prompt = f"""
PREVIOUS TASK: {enzyme_prompt}

STARTING SEQUENCE: {perturbed_sequence}

MUTATIONS TO BE APPLIED TO GET TO THE FINAL SEQUENCE (in order):
{mutations_text}

FINAL SEQUENCE: {initial_sequence}

An expert protein engineer has selected these mutations to optimize the stability of the enzyme while keeping the function/activity of the enzyme unchanged. The mutations must be applied IN ORDER as listed above, as each mutation's position depends on the previous mutations being applied first. Generate reasoning in the style of a chain of thought that applies these mutations in the given order to reach the final sequence.

****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTIFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** 

Follow these formatting instructions exactly: 
Enclose the reasoning process within <think> </think> tags, i.e., <think> reasoning process here </think>. Then for your answer for the user, enclose the final sequence within <answer> </answer> tags, i.e., <answer> final sequence here and summary of reasoning </answer>.
COPY THE FINAL SEQUENCE WITH THE MUTATIONS APPLIED IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. YOU MUST FOLLOW THIS INSTRUCTION/FORMAT. EX; \\boxed{{MALWMTLLLLPVPDGPK...}} Do not use any coloring or other formatting within the boxed term, we only want the sequence in those brackets.
"""
    
    model = "gpt-4o"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert protein engineer with years of experience optimizing protein stability with rational design."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    reasoning = response.choices[0].message.content
    print(reasoning)
    return reasoning

def load_esm_model():
    # Initialize ESM3 model
    login(token=os.getenv("HUGGINGFACE_API_KEY"))
    esm_model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda" if torch.cuda.is_available() else "cpu")

    print(esm_model)
    
    return esm_model

def generate_insertion(sequence: str, esm_model: ESM3InferenceClient) -> Tuple[int, str]:
    """Generate a random insertion position and sequence using ESM3.
    Returns (position, new_aa_sequence)"""
    
    # Pick random position to insert
    pos = random.randint(0, len(sequence))
    
    # Generate random length of insertion (1-3 amino acids)
    insert_length = random.randint(1, 10)
    
    # Create sequence with blanks for ESM to fill
    blank_seq = sequence[:pos] + '_' * insert_length + sequence[pos:]
    print(blank_seq)
    
    # Create protein object and generate completion
    protein = ESMProtein(sequence=blank_seq)
    completed_protein = esm_model.generate(
        protein, 
        GenerationConfig(
            track="sequence", 
            num_steps=8, 
            temperature=0.7
        )
    )
    
    # Extract the generated amino acids
    completed_seq = completed_protein.sequence
    new_aas = completed_seq[pos:pos+insert_length]
    
    return pos, new_aas


def generate_initial_mutations(sequence: str, esm_model: ESM3InferenceClient, n_mutations: int = None) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Generate n_mutations random mutations from the original sequence.
    Returns (perturbed_sequence, mutations) where mutations describe how to go from
    perturbed sequence back to original sequence."""
    if n_mutations is None:
        n_mutations = random.randint(3, 5)
    
    mutations = []
    mutated_seq = sequence
    positions_used = set()
    
    # Store forward mutations temporarily
    forward_mutations = []
    
    while len(forward_mutations) < n_mutations:
        if random.random() < 0.99:
            if random.random() < 0.5:  # Insertion
                pos, new_aa = generate_insertion(mutated_seq, esm_model)
                # Store as forward mutation
                forward_mutations.append(("insertion", pos, new_aa))
                mutated_seq = mutated_seq[:pos] + new_aa + mutated_seq[pos:]
                positions_used.add(pos)
            else:  # Deletion
                deletion_length = random.randint(1, 5)
                while True:
                    pos = random.randint(0, len(mutated_seq)-deletion_length)
                    chunk = mutated_seq[pos:pos+deletion_length]
                    if not any(aa in chunk for aa in ['M', 'C']) and \
                       not any(p in positions_used for p in range(pos, pos+deletion_length)):
                        break
                
                # Store as forward mutation
                forward_mutations.append(("deletion", pos, chunk))
                mutated_seq = mutated_seq[:pos] + mutated_seq[pos+deletion_length:]
                for i in range(pos, pos+deletion_length):
                    positions_used.add(i)
        else:
            # Regular substitution
            while True:
                orig_aa, pos, new_aa = sample_mutation(sequence)
                if pos not in positions_used and orig_aa != 'M' and new_aa != 'M' and orig_aa != 'C':
                    break
            
            positions_used.add(pos)
            forward_mutations.append(("substitution", pos, (orig_aa, new_aa)))
            mutated_seq = mutated_seq[:pos] + new_aa + mutated_seq[pos+1:]
    
    # Convert forward mutations to reverse mutations
    offset = 0  # Track position changes due to insertions/deletions
    for mut_type, pos, info in forward_mutations:
        if mut_type == "deletion":
            # If we deleted sequence, we need to insert it back
            adjusted_pos = pos + offset
            # Store as (pos, "ins", sequence_to_insert)
            mutations.append((adjusted_pos, "ins", info))  # Format will be {pos}ins{info}
            offset -= len(info)  # Position adjustment decreases after deletion
        elif mut_type == "insertion":
            # If we inserted sequence, we need to delete it
            adjusted_pos = pos + offset
            # Store as (start_pos, "del", end_pos)
            end_pos = adjusted_pos + len(info) - 1
            mutations.append((adjusted_pos, "del", end_pos))  # Format will be {start_aa}{start_pos}_{end_aa}{end_pos}del
            offset += len(info)  # Position adjustment increases after insertion
        else:  # substitution
            # For a substitution, swap the amino acids
            orig_aa, new_aa = info
            adjusted_pos = pos + offset
            mutations.append((adjusted_pos, new_aa, orig_aa))  # Format remains {orig_aa}{pos}{new_aa}
    
    return mutated_seq, mutations

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def create_mutation_traces(transformed_data: Dict[str, Dict], n_traces: int = 100, n_mutations: int = 5):
    """Create dataset of mutation traces with reasoning paths"""
    dataset = {
        "metadata": {
            "description": "Protein mutation traces dataset with reasoning paths",
            "n_traces": n_traces,
            "n_mutations": n_mutations,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "estimated_cost": 0
        },
        "traces": []
    }

    esm_model = load_esm_model()
    
    # Calculate total iterations for the progress bar
    total_iterations = len(transformed_data) * n_traces
    
    with tqdm(total=total_iterations, desc="Generating mutation traces") as pbar:
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
                perturbed_seq, mutations = generate_initial_mutations(sequence, esm_model, n_mutations)
                
                # Generate a prompt for this enzyme trace
                enzyme_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data.get('name', 'Unknown')}
EC NUMBER: {enzyme_data.get('ec_number', 'Unknown')}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data.get('general_information', 'No additional information available')}
SUBSTRATES: {', '.join(substrates)}
PRODUCTS: {', '.join(products)}
METALS/IONS: {', '.join(metal_ions)}
{known_mutations_text}

Propose mutations to optimize the stability of the enzyme given the information above. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects (or does not affect) protein structure
2. How the mutation affects (or does not affect) protein function
3. The chemical properties of the amino acids and substrates/products

Remember to be specific to the enzyme and reaction specified in the prompt.
"""

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
                    }
                }
                
                # Count input tokens
                input_tokens = count_tokens(enzyme_prompt)
                dataset["metadata"]["total_input_tokens"] += input_tokens
                
                # Generate reasoning for all mutations
                trace["reasoning"] = generate_reasoning(perturbed_seq, mutations, enzyme_prompt, sequence)
                
                # Count output tokens
                output_tokens = count_tokens(trace["reasoning"])
                dataset["metadata"]["total_output_tokens"] += output_tokens
                
                # Calculate current cost
                input_cost = (dataset["metadata"]["total_input_tokens"] / 1_000_000) * 2.50
                output_cost = (dataset["metadata"]["total_output_tokens"] / 1_000_000) * 10.0
                total_cost = input_cost + output_cost
                dataset["metadata"]["estimated_cost"] = total_cost
                
                dataset["traces"].append(trace)
                pbar.update(1)
                pbar.set_postfix({'cost': f'${total_cost:.2f}'})
                
                # Check if we've exceeded cost threshold
                if total_cost >= 95.0:
                    print(f"\nReached cost threshold (${total_cost:.2f}). Saving and exiting...")
                    Path("data").mkdir(exist_ok=True)
                    with open("data/mega_cot.json", "w") as f:
                        json.dump(dataset, f, indent=2)
                    return
            
    # Save dataset if we haven't hit the cost threshold
    Path("data").mkdir(exist_ok=True)
    with open("data/mega_cot.json", "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    # Load transformed BRENDA data
    with open('data/transformed_brenda.json', 'r') as f:
        transformed_data = json.load(f)
    
    # Filter to keep only one protein per EC number while preserving data structure
    seen_ec_numbers = set()
    filtered_data = {}

    # First collect all proteins with engineering data
    proteins_with_engineering = {
        uniprot_id: data for uniprot_id, data in transformed_data.items() 
        if data.get("engineering") and len(data["engineering"]) > 0
        and isinstance(data.get("sequence"), str)  # Add check for string sequence
    }
    
    # Group by EC number
    ec_to_proteins = {}
    for uniprot_id, data in proteins_with_engineering.items():
        ec = data["ec_number"]
        if ec not in ec_to_proteins:
            ec_to_proteins[ec] = []
        ec_to_proteins[ec].append((uniprot_id, data))

    sampled_ec_numbers = random.sample(list(ec_to_proteins.keys()), 
                                     min(100, len(ec_to_proteins)))
    
    # Take one random protein from each sampled EC number
    filtered_data = {}
    for ec in sampled_ec_numbers:
        uniprot_id, data = random.choice(ec_to_proteins[ec])
        filtered_data[uniprot_id] = data


    length_threshold = 1024

    # Filter out sequences longer than threshold and ensure sequence is a string
    filtered_data = {
        uniprot_id: data for uniprot_id, data in filtered_data.items()
        if isinstance(data.get('sequence'), str) and len(data['sequence']) <= length_threshold
    }
    
    transformed_data = filtered_data

    create_mutation_traces(transformed_data, n_traces=2, n_mutations=3)  # n per sequence
