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

def generate_reasoning(sequence: str, mutations: List[Tuple[str, int, str]], enzyme_prompt: str) -> str:
    """Generate reasoning for all mutations using OpenAI API"""
    
    mutations_text = "\n".join([f"{orig_aa}{pos}{new_aa}" for orig_aa, pos, new_aa in mutations])
    
    prompt = f"""Given the following task and sequence modifications:

USER: {enzyme_prompt}

MUTATIONS TO REVERT:
{mutations_text}

Provide a detailed chain of reasoning for how to revert these mutations to restore optimal enzyme activity. For each mutation, explain the scientific rationale behind reverting it. ****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTIFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** 

At the end of your response, list the final sequence of mutations in order, in the format:
FINAL_SEQUENCE: mutation1,mutation2,mutation3

Keep your explanation focused on the scientific reasoning."""
    
    model = "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert protein engineer with years of experience optimizing protein activity with rational design."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    reasoning = response.choices[0].message.content
    print(reasoning)
    return reasoning

def generate_insertion(sequence: str) -> Tuple[int, str]:
    """Generate a random insertion position and sequence.
    Returns (position, new_aa_sequence)"""
    
    # Pick random position to insert
    pos = random.randint(0, len(sequence))
    
    # Generate random length of insertion (1-3 amino acids)
    insert_length = random.randint(1, 3)
    
    # Create sequence with blanks for ESM to fill
    blank_seq = sequence[:pos] + '_' * insert_length + sequence[pos:]
    
    # Load ESM model
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Tokenize sequence
    inputs = tokenizer(blank_seq, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Find positions of masked tokens
    mask_positions = [i for i, c in enumerate(blank_seq) if c == '_']
    
    # Sample amino acids for each masked position
    new_aas = ''
    for mask_pos in mask_positions:
        # Get probabilities for this position
        token_probs = torch.softmax(predictions[0, mask_pos], dim=0)
        
        # Sample from probability distribution
        token_id = torch.multinomial(token_probs, 1).item()
        new_aa = tokenizer.decode([token_id])
        
        new_aas += new_aa
        
    return pos, new_aas


def generate_initial_mutations(sequence: str, n_mutations: int = None) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Generate n_mutations random mutations from the original sequence"""
    if n_mutations is None:
        n_mutations = random.randint(3, 5)
    
    mutations = []
    mutated_seq = sequence
    positions_used = set()
    
    while len(mutations) < n_mutations:
        # With 20% probability, do an insertion or deletion instead of substitution
        if random.random() < 0.2:
            if random.random() < 0.5:  # 50-50 chance of insertion vs deletion
                pos, new_aa = generate_insertion(mutated_seq)
                mutations.append((pos, '', new_aa))  # Empty orig_aa indicates insertion
                mutated_seq = mutated_seq[:pos] + new_aa + mutated_seq[pos:]
                positions_used.add(pos)
            else:
                # Deletion: Sample a chunk to delete
                # Delete between 1-5 amino acids
                deletion_length = random.randint(1, 5)
                while True:
                    # Sample starting position
                    pos = random.randint(0, len(mutated_seq)-deletion_length)
                    # Get the chunk to delete
                    chunk = mutated_seq[pos:pos+deletion_length]
                    # Don't delete if chunk contains M or C
                    if not any(aa in chunk for aa in ['M', 'C']) and \
                       not any(p in positions_used for p in range(pos, pos+deletion_length)):
                        break
                
                # Add each position in the deletion to mutations list
                for i, aa in enumerate(chunk):
                    mutations.append((pos+i, aa, ''))  # Empty new_aa indicates deletion
                    positions_used.add(pos+i)
                
                # Delete the chunk
                mutated_seq = mutated_seq[:pos] + mutated_seq[pos+deletion_length:]
        else:
            # Regular substitution mutation
            while True:
                orig_aa, pos, new_aa = sample_mutation(sequence)
                # Skip if position contains methionine or would mutate to methionine or original is cysteine
                if pos not in positions_used and orig_aa != 'M' and new_aa != 'M' and orig_aa != 'C':
                    break
            
            positions_used.add(pos)
            mutations.append((pos, orig_aa, new_aa))
            mutated_seq = mutated_seq[:pos] + new_aa + mutated_seq[pos+1:]
    
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

Propose mutations to optimize enzymatic activity given the substrates and products above. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects protein structure and function
2. The chemical properties of the amino acids and substrates/products
3. The position's importance in the protein sequence"""

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
            trace["reasoning"] = generate_reasoning(perturbed_seq, mutations, enzyme_prompt)
            
            dataset["traces"].append(trace)
            
    # Save dataset
    Path("data").mkdir(exist_ok=True)
    with open("data/mutation_traces.json", "w") as f:
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
    
    # Sample 50 EC numbers if available, otherwise take all
    sampled_ec_numbers = random.sample(list(ec_to_proteins.keys()), 
                                     min(50, len(ec_to_proteins)))
    
    # Take one random protein from each sampled EC number
    filtered_data = {}
    for ec in sampled_ec_numbers:
        uniprot_id, data = random.choice(ec_to_proteins[ec])
        filtered_data[uniprot_id] = data
    
    transformed_data = filtered_data

    create_mutation_traces(transformed_data, n_traces=2, n_mutations=3)  # n per sequence
