# gradient based sequence optimization using evoprotgrad. compare to critic with access to reward signal 

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Install required packages
import subprocess
subprocess.run(["pip", "install", "evo_prot_grad", "-q"], check=True)

import json
from pathlib import Path
from tqdm import tqdm
import evo_prot_grad
from transformers import AutoTokenizer, EsmForMaskedLM
from stability_reward import StabilityRewardCalculator

# Initialize models
esm2_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
stability_calculator = StabilityRewardCalculator()

def optimize_sequence(sequence):
    # Create temporary FASTA file
    fasta_format = f">Input_Sequence\n{sequence}"
    temp_fasta_path = "temp_input_sequence.fasta"
    with open(temp_fasta_path, "w") as f:
        f.write(fasta_format)

    # Initialize ESM expert
    esm2_expert = evo_prot_grad.get_expert(
        'esm',
        model=esm2_model,
        tokenizer=tokenizer,
        temperature=0.95,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run directed evolution
    evolution = evo_prot_grad.DirectedEvolution(
        wt_fasta=temp_fasta_path,
        output='best',
        experts=[esm2_expert],
        parallel_chains=1,
        n_steps=20,
        max_mutations=10,
        verbose=False
    )
    
    variants, scores = evolution()
    
    # Clean up temporary file
    os.remove(temp_fasta_path)
    
    return variants[0]  # Return the best variant

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
            
            # Generate new sequence with EvoProtGrad
            mutated_sequence = optimize_sequence(sequence)
            
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
    
    with open(output_dir / 'esm_stability_mutations.json', 'w') as f:
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
        with open(output_dir / 'esm_stability_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e
    finally:
        # Uninstall EvoProtGrad after completion
        try:
            subprocess.run(["pip", "uninstall", "evo_prot_grad", "-y"], check=True)
        except Exception as e:
            print(f"Warning: Failed to uninstall EvoProtGrad: {str(e)}")

