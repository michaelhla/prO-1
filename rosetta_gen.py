import json
import os
from stability_reward import StabilityRewardCalculator
import torch

# Set up stability calculator on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calculator = StabilityRewardCalculator(device=device)

# Load the transformed BRENDA data
with open("data/transformed_brenda.json", 'r') as f:
    data_dict = json.load(f)

# Get list of structure files
structure_files = os.listdir("predicted_structures")

# Calculate stability for each structure and add to data
from tqdm import tqdm
for pdb_file in tqdm(structure_files, desc="Calculating stability scores"):
    enzyme_id = pdb_file.replace(".pdb", "")
    
    if enzyme_id in data_dict:
        try:
            # Get sequence from data
            sequence = data_dict[enzyme_id]["sequence"]
            
            # Calculate stability using structure file
            stability = calculator.calculate_stability(
                sequence,
                pdb_file_path=f"predicted_structures/{pdb_file}"
            )
            
            # Add stability score to data
            data_dict[enzyme_id]["orig_stab"] = float(stability)
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            data_dict[enzyme_id]["orig_stab"] = None
# Save updated data back to file
try:
    with open("data/transformed_brenda.json", 'w') as f:
        json.dump(data_dict, f, indent=2)
except Exception as e:
    print(f"Error saving to original file: {e}")
    # Try saving to new file
    try:
        with open("data/transformed_brenda_with_stability.json", 'w') as f:
            json.dump(data_dict, f, indent=2)
        print("Saved data to transformed_brenda_with_stability.json")
    except Exception as e:
        print(f"Error saving to new file: {e}")

print("Finished calculating and storing stability scores")
