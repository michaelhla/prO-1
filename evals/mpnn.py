import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Install required packages
import subprocess
subprocess.run(["pip", "install", "git+https://github.com/dauparas/ProteinMPNN.git"], check=True)

import json
from pathlib import Path
from tqdm import tqdm
from colabdesign.mpnn import mk_mpnn_model
from stability_reward import StabilityRewardCalculator

# Initialize models
mpnn_model = mk_mpnn_model()
stability_calculator = StabilityRewardCalculator()

def main():
    # Load previous results
    with open('results/selected_enzymes.json', 'r') as f:
        selected_enzymes = json.load(f)
    
    results = []
    
    for enzyme_id, data in tqdm(selected_enzymes.items(), desc="Processing enzymes"):
        sequence = data['sequence']
        
        try:
            # Calculate original stability
            original_stability = stability_calculator.calculate_stability(sequence)
            
            # Get structure prediction for MPNN
            pdb_file = stability_calculator.predict_structure(sequence)
            
            # Generate new sequence with MPNN
            mpnn_model.prep_inputs(pdb_filename=pdb_file)
            mutated_sequence = mpnn_model.sample()
            
            # Calculate new stability
            new_stability = stability_calculator.calculate_stability(mutated_sequence)
            
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability < original_stability
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': None,
                'mutated_sequence': None,
                'new_stability': None,
                'stability_change': None,
                'is_improvement': False
            })
            continue
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'mpnn_stability_mutations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nResults summary:")
    print(f"Number of enzymes processed: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    
    stability_changes = [r['stability_change'] for r in results if r['stability_change'] is not None]
    if stability_changes:
        print(f"Max stability improvement: {min(stability_changes):.3f}")
    else:
        print("No valid stability improvements found")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error encountered: {str(e)}")
        # Save current results before exit
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'mpnn_stability_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e
    finally:
        # Uninstall ProteinMPNN after completion
        try:
            subprocess.run(["pip", "uninstall", "proteinmpnn", "-y"], check=True)
        except Exception as e:
            print(f"Warning: Failed to uninstall ProteinMPNN: {str(e)}")

