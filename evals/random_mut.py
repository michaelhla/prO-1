import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import random
from pathlib import Path
from tqdm import tqdm
from stability_reward import StabilityRewardCalculator

# Results summary:
# Number of enzymes processed: 29
# Number of successful improvements: 6
# Success rate: 20.7%

# Results summary (only most stable amino acids):
# Number of enzymes processed: 40
# Number of successful improvements: 15
# Success rate: 37.5%

# Initialize stability calculator
stability_calculator = StabilityRewardCalculator()

def get_stability_score(sequence: str) -> float:
    """Calculate protein stability score using ESMFold and PyRosetta"""
    return stability_calculator.calculate_stability(sequence)

def random_mutations(sequence: str, num_mutations: int) -> str:
    """Make random mutations in the sequence"""
    amino_acids = 'AGILPV'  # Testing only the most stable amino acids
    sequence = list(sequence)
    positions = random.sample(range(len(sequence)), num_mutations)
    
    for pos in positions:
        new_aa = random.choice(amino_acids.replace(sequence[pos], ''))
        sequence[pos] = new_aa
        
    return ''.join(sequence)

def main():
    # Load selected enzyme sequences
    with open('results/selected_enzymes.json', 'r') as f:
        selected_enzymes = json.load(f)
    
    # Try to load existing results to get original stability values
    existing_results = {}
    try:
        with open('results/ds_r1_stability_mutations.json', 'r') as f:
            for result in json.load(f):
                if result.get('original_stability') is not None:
                    existing_results[result['enzyme_id']] = result['original_stability']
    except Exception as e:
        print(f"Could not load existing results: {str(e)}")
    
    results = []
    
    for enzyme_id, data in tqdm(selected_enzymes.items(), desc="Processing enzymes"):
        sequence = data['sequence']
        
        # Try to get stability from existing results first
        original_stability = existing_results.get(enzyme_id)
        
        # If not found in existing results, calculate it
        if original_stability is None:
            try:
                original_stability = get_stability_score(sequence)
            except Exception as e:
                print(f"Error calculating stability for enzyme {enzyme_id}: {str(e)}")
                print('Sequence length: ', len(str(sequence)))
                continue
            
        try:
            # Make 3-7 random mutations
            num_mutations = random.randint(3, 7)
            mutated_sequence = random_mutations(sequence, num_mutations)
            
            # Calculate new stability
            new_stability = get_stability_score(mutated_sequence)
            
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability < original_stability,
                'num_mutations': num_mutations
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            continue
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'random_mutations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nResults summary:")
    print(f"Number of enzymes processed: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Fix max calculation to handle None values
    stability_changes = [r['stability_change'] for r in results if r['stability_change'] is not None]
    if stability_changes:
        print(f"Max stability improvement: {max(stability_changes):.3f}")
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
        with open(output_dir / 'random_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e
