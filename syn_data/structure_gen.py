import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import pyrosetta
from tqdm import tqdm
import json
import time
import random
import os
import sys 
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stability_reward import StabilityRewardCalculator

def sequence_to_pdb_string(positions, sequence):
    """Convert predicted positions to PDB format string."""
    io_string = io.StringIO()
    for i, (pos, aa) in enumerate(zip(positions, sequence)):
        record = (
            f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}"
            f"    {float(pos[0]):8.3f}{float(pos[1]):8.3f}{float(pos[2]):8.3f}"
            f"  1.00  0.00           C  \n"
        )
        io_string.write(record)
    io_string.write("END\n")
    return io_string.getvalue()

def filter_dataset(data_dict):
    """Apply multiple filters to the dataset."""
    print("Starting dataset filtering...")
    original_count = len(data_dict)
    
    # Load evaluation dataset to get excluded UniProt IDs
    print("Loading evaluation dataset...")
    with open("results/selected_enzymes.json", 'r') as f:
        eval_enzymes = json.load(f)
    eval_uniprot_ids = set(eval_enzymes.keys())
    
    # Filter out evaluation enzymes
    filtered_dict = {k: v for k, v in data_dict.items() if k not in eval_uniprot_ids}
    print(f"After removing eval enzymes: {len(filtered_dict)} entries (removed {original_count - len(filtered_dict)})")
    
    # Filter sequences longer than 1024 residues and ensure sequence is a string
    filtered_dict = {
        k: v for k, v in filtered_dict.items() 
        if isinstance(v.get('sequence'), str) and len(v['sequence']) <= 1024
    }
    print(f"After filtering by sequence length: {len(filtered_dict)} entries")

    # Filter out enzymes with unknown products (marked with '?')
    filtered_dict = {
        k: v for k, v in filtered_dict.items()
        if 'reaction' in v and all(
            all(product != '?' for product in reaction.get('products', []))
            for reaction in v['reaction']
        )
    }
    print(f"After filtering unknown products: {len(filtered_dict)} entries")

    # Group by EC number and select one entry per EC
    ec_groups = defaultdict(list)
    for uniprot_id, entry in filtered_dict.items():
        ec_groups[entry['ec_number']].append((uniprot_id, entry))
    
    # Select first entry for each EC number
    filtered_dict = {entries[0][0]: entries[0][1] for entries in ec_groups.values()}
    print(f"After selecting one per EC: {len(filtered_dict)} entries")


    # Filter based on general_information and engineering fields
    filtered_dict = {
        k: v for k, v in filtered_dict.items()
        if (('general_information' in v and v['general_information']) or
            ('engineering' in v and v['engineering']))
    }
    print(f"After filtering by information fields: {len(filtered_dict)} entries")
    
    # Filter out carbonic anhydrase
    filtered_dict = {
        k: v for k, v in filtered_dict.items()
        if 'carbonic anhydrase' not in v.get('name', '').lower()
    }
    
    return filtered_dict

def generate_structures(selected_samples, num_samples=1000, batch_size=4):
    """Generate protein structures for dataset samples using ESMFold."""
    # Initialize StabilityRewardCalculator
    stability_calculator = StabilityRewardCalculator()

    selected_samples = random.sample(selected_samples, num_samples)
    
    print(f"Generating structures for {num_samples} samples in batches of {batch_size}...")
    structures = []
    for i in tqdm(range(0, len(selected_samples), batch_size)):
        batch = selected_samples[i:i + batch_size]
        
        try:
            # Process each sequence in the batch
            for j, sample in enumerate(batch):
                sequence = sample['sequence']
                uniprot_id = list(filtered_dict.keys())[i+j]
                
                # Get structure and stability using StabilityRewardCalculator
                pdb_string = stability_calculator.predict_structure(sequence, uniprot_id=uniprot_id)
                stability_score = stability_calculator.calculate_stability(sequence)
                
                # Add structure and stability to the sample
                sample['structure_pdb'] = pdb_string
                sample['stability_score'] = float(stability_score)
                structures.append(sample)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            continue
    
    return structures

if __name__ == "__main__":
    # start_time = time.time()
    
    # # Load the dataset
    # print("Loading dataset...")
    # with open("data/transformed_brenda.json", 'r') as f:
    #     data_dict = json.load(f)
    
    # # Apply filters
    # filtered_dict = filter_dataset(data_dict)
    # data_list = list(filtered_dict.values())
    
    # print(f"Final dataset size after filtering: {len(data_list)} entries")
    # # Save filtered dataset
    # output_path = "data/brenda_structures.json"
    # print(f"Saving filtered dataset to {output_path}...")
    # with open(output_path, 'w') as f:
    #     json.dump(filtered_dict, f, indent=2)
    
    # # Generate structures and calculate stability
    # structured_samples = generate_structures(data_list, num_samples=1000)
    
    # # Create new dataset with structures and stability scores
    # print("Creating new dataset with generated structures and stability scores...")
    # structured_data = {}
    # for sample in structured_samples:
    #     key = sample['ec_number']  # adjust if your identifier is different
    #     if key in filtered_dict:
    #         # Copy over all existing data
    #         structured_data[key] = filtered_dict[key].copy()
    #         # Add new structure and stability data
    #         structured_data[key]['structure_pdb'] = sample['structure_pdb']
    #         structured_data[key]['stability_score'] = sample['stability_score']
    
    # # Save new dataset
    # output_path = "data/brenda_structures.json"
    # print(f"Saving new dataset to {output_path}...")
    # with open(output_path, 'w') as f:
    #     json.dump(structured_data, f, indent=2)
    
    # print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    # print(f"Successfully generated structures for {len(structured_samples)} samples")

    
