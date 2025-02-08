import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import pyrosetta
from tqdm import tqdm
import json
import time
import os
from multiprocessing import Pool

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

def calculate_stability(pdb_file):
    """Calculate stability score for a single PDB file using Rosetta."""
    if not pyrosetta.is_initialized():
        pyrosetta.init()
        
    pose = pyrosetta.pose_from_pdb(pdb_file)
    scorefxn = pyrosetta.get_fa_scorefxn()
    task = pyrosetta.standard_packer_task(pose)
    task.restrict_to_repacking()
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
    packer.apply(pose)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    min_mover.score_function(scorefxn)
    min_mover.apply(pose)
    return scorefxn(pose)

def generate_structures(dataset, num_samples=1000, batch_size=4):
    """Generate protein structures for dataset samples using ESMFold."""
    # Load model
    print("Loading ESMFold model...")
    local_path = '/root/prO-1/model_cache/'
    if os.path.exists(local_path):
        model = EsmForProteinFolding.from_pretrained(local_path)
    else:
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        model.save_pretrained(local_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    # Select samples
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    selected_samples = [dataset[i] for i in indices]
    
    # Create output directory for PDB files
    output_dir = "predicted_structures"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating structures for {num_samples} samples in batches of {batch_size}...")
    structures = []
    for i in tqdm(range(0, len(selected_samples), batch_size)):
        batch = selected_samples[i:i + batch_size]
        sequences = [item['sequence'] for item in batch]
        
        try:
            # Tokenize sequences
            tokenized_input = tokenizer(
                sequences,
                return_tensors="pt",
                add_special_tokens=False
            )['input_ids'].to(device)
            
            # Generate structures
            with torch.no_grad():
                output = model(tokenized_input)
            
            # Convert to PDB format
            final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
            output = {k: v.to("cpu").numpy() for k, v in output.items()}
            final_atom_positions = final_atom_positions.cpu().numpy()
            final_atom_mask = output["atom37_atom_exists"]
            
            # Process each structure in the batch
            for j, (sequence, sample) in enumerate(zip(sequences, batch)):
                aa = output["aatype"][j]
                pred_pos = final_atom_positions[j]
                mask = final_atom_mask[j]
                resid = output["residue_index"][j] + 1
                
                pred = OFProtein(
                    aatype=aa,
                    atom_positions=pred_pos,
                    atom_mask=mask,
                    residue_index=resid,
                    b_factors=output["plddt"][j],
                )
                
                # Save PDB file
                pdb_string = to_pdb(pred)
                pdb_path = os.path.join(output_dir, f"structure_{i+j}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(pdb_string)
                
                # Calculate stability
                stability_score = calculate_stability(pdb_path)
                
                # Add structure and stability to the sample
                sample['structure_pdb'] = pdb_string
                sample['stability_score'] = float(stability_score)
                structures.append(sample)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            continue
    
    return structures

if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize PyRosetta
    pyrosetta.init()
    
    # Load the dataset
    print("Loading dataset...")
    with open("data/transformed_brenda.json", 'r') as f:
        data_dict = json.load(f)
        data_list = list(data_dict.values())
    
    # Generate structures and calculate stability
    structured_samples = generate_structures(data_list, num_samples=1000)
    
    # Update the original data_dict with structures and stability scores
    print("Updating dataset with generated structures and stability scores...")
    for sample in structured_samples:
        key = sample['ec_number']  # adjust if your identifier is different
        if key in data_dict:
            data_dict[key]['structure_pdb'] = sample['structure_pdb']
            data_dict[key]['stability_score'] = sample['stability_score']
    
    # Save updated dataset
    output_path = "data/transformed_brenda_with_structures.json"
    print(f"Saving enhanced dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Successfully generated structures for {len(structured_samples)} samples")
