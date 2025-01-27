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
    for mut in mutations:
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

MUTATIONS TO BE APPLIED:
{mutations_text}

FINAL SEQUENCE: {initial_sequence}

An expert protein engineer has selected these mutations to optimize the stability of the enzyme while keeping the function/activity of the enzyme unchanged. Knowing that these are the mutations/operations that should be applied, generate a chain of reasoning that applies these mutations in a logical order resulting in the final sequence. Only generate reasoning using the mutations provided. For each mutation, explain the scientific rationale behind reverting it. ****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTIFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** 

At the end of your response, copy the final sequence, in the format below, using $$ to enclose the sequence:
%%FINAL_SEQUENCE%%: $$_______$$

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
    
    # Create protein object and generate completion
    protein = ESMProtein(sequence=blank_seq)
    seq_prompt = esm_model.encode(protein)
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
        if random.random() < 0.2:
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
        if mut_type == "insertion":
            # For an insertion, we need to delete in reverse
            adjusted_pos = pos + offset
            # Store as (pos, "ins", inserted_sequence)
            mutations.append((adjusted_pos, "ins", info))  # Format will be {pos}ins{info}
            offset += len(info)
        elif mut_type == "deletion":
            # For a deletion, we need to insert in reverse
            adjusted_pos = pos + offset
            # Store as (start_pos, "del", end_pos)
            end_pos = adjusted_pos + len(info) - 1
            mutations.append((adjusted_pos, "del", end_pos))  # Format will be {start_aa}{start_pos}_{end_aa}{end_pos}del
            offset -= len(info)
        else:  # substitution
            # For a substitution, swap the amino acids
            orig_aa, new_aa = info
            adjusted_pos = pos + offset
            mutations.append((adjusted_pos, new_aa, orig_aa))  # Format remains {orig_aa}{pos}{new_aa}
    
    return mutated_seq, mutations

def create_mutation_traces(transformed_data: Dict[str, Dict], n_traces: int = 100, n_mutations: int = 5):
    """Create dataset of mutation traces with reasoning paths"""
    dataset = {
        "metadata": {
            "description": "Protein mutation traces dataset with reasoning paths",
            "n_traces": n_traces,
            "n_mutations": n_mutations
        },
        "traces": []
    }

    esm_model = load_esm_model()
    
    for enzyme_id, enzyme_data in tqdm(transformed_data.items(), desc="Creating mutation traces"):
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
            enzyme_prompt = f"""You are an expert protein engineer. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

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
            
            # Generate reasoning for all mutations
            trace["reasoning"] = generate_reasoning(perturbed_seq, mutations, enzyme_prompt, sequence)
            
            dataset["traces"].append(trace)
            
    # Save dataset
    Path("data").mkdir(exist_ok=True)
    with open("data/cot_mutation_traces.json", "w") as f:
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
    }
    
    # Group by EC number
    ec_to_proteins = {}
    for uniprot_id, data in proteins_with_engineering.items():
        ec = data["ec_number"]
        if ec not in ec_to_proteins:
            ec_to_proteins[ec] = []
        ec_to_proteins[ec].append((uniprot_id, data))
    
    # Sample 100 EC numbers if available, otherwise take all
    sampled_ec_numbers = random.sample(list(ec_to_proteins.keys()), 
                                     min(100, len(ec_to_proteins)))
    
    # Take one random protein from each sampled EC number
    filtered_data = {}
    for ec in sampled_ec_numbers:
        uniprot_id, data = random.choice(ec_to_proteins[ec])
        filtered_data[uniprot_id] = data
    
    transformed_data = filtered_data

    create_mutation_traces(transformed_data, n_traces=2, n_mutations=3)  # n per sequence
