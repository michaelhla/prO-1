import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict
import openai
from stability_reward import StabilityRewardCalculator

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Results summary:
# Number of enzymes processed: 32
# Number of successful improvements: 9
# Success rate: 28.1%
# Max stability improvement: -240.231

# Initialize stability calculator
stability_calculator = StabilityRewardCalculator()

def get_stability_score(sequence: str) -> float:
    """Calculate protein stability score using ESMFold and PyRosetta"""
    return stability_calculator.calculate_stability(sequence)

def propose_mutations(sequence: str, enzyme_data: Dict) -> str:
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data['name']}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data['general_information']}
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in enzyme_data['active_site_residues']])}

Propose 3-7 mutations to optimize the stability of the enzyme given the information above. YOUR MUTATIONS MUST BE POSITION SPECIFIC TO THE SEQUENCE. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning. 

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions.****

COPY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. YOU MUST FOLLOW THIS INSTRUCTION/FORMAT. EX; \\boxed{{MALWMTLLLLPVPDGPK...}} Do not use any coloring or other formatting within the boxed term, we only want the sequence in those brackets."""

    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": base_prompt}
        ],
    )
    
    return response.choices[0].message.content
def extract_sequence(response: str) -> str:
    """Extract sequence from model response"""
    try:
        # Look for sequence in \boxed{} format
        if '\\boxed{' in response and '}' in response:
            start = response.find('\\boxed{') + 7
            
            # Find matching closing bracket by counting brackets
            bracket_count = 1
            pos = start
            while bracket_count > 0 and pos < len(response):
                pos += 1
                if pos >= len(response):
                    return None
                if response[pos] == '{':
                    bracket_count += 1
                elif response[pos] == '}':
                    bracket_count -= 1
            
            if bracket_count == 0:
                end = pos
                sequence = response[start:end].strip()
                
                # Remove any LaTeX commands but keep their content
                while '\\' in sequence:
                    backslash_idx = sequence.find('\\')
                    if backslash_idx == -1:
                        break
                    
                    # Skip the backslash and find the opening brace
                    open_brace = sequence.find('{', backslash_idx)
                    if open_brace == -1:
                        break
                    
                    # Find matching closing brace
                    bracket_count = 1
                    pos = open_brace + 1
                    while bracket_count > 0 and pos < len(sequence):
                        if sequence[pos] == '{':
                            bracket_count += 1
                        elif sequence[pos] == '}':
                            bracket_count -= 1
                        pos += 1
                    
                    if bracket_count > 0:
                        break
                    
                    close_brace = pos - 1
                    
                    # Keep just what's inside the braces
                    inner_content = sequence[open_brace+1:close_brace]
                    sequence = sequence[:backslash_idx] + inner_content + sequence[close_brace+1:]
                
                # Remove any whitespace between characters
                sequence = ''.join(sequence.split())
                return sequence
        return None
    except Exception:
        return None

def main():
    # Load previous results instead of selected enzymes
    with open('results/o1_stability_mutations.json', 'r') as f:
        results = json.load(f)
    
    # Load selected enzyme sequences for reference
    with open('results/selected_enzymes.json', 'r') as f:
        selected_enzymes = json.load(f)
    
    for i, result in tqdm(enumerate(results), desc="Processing enzymes"):
        # Skip if already has valid mutated sequence
        if result['mutated_sequence'] is not None:
            continue
            
        enzyme_id = result['enzyme_id']
        data = selected_enzymes[enzyme_id]
        sequence = result['original_sequence']
        original_stability = result['original_stability']
        
        try:
            # Format enzyme data for the prompt
            enzyme_data = {
                "name": data.get('name', 'Unknown Enzyme'),
                "ec_number": data.get('ec_number', 'Unknown'),
                "general_information": data.get('description', 'No description available'),
                "reaction": [{
                    "substrates": data.get('substrates', ['Unknown']),
                    "products": data.get('products', ['Unknown'])
                }],
                "active_site_residues": data.get('active_site_residues', [])
            }
            
            # Get new model response
            response = propose_mutations(sequence, enzyme_data)
            # Extract mutated sequence
            mutated_sequence = extract_sequence(response)

            print(f'Mutated sequence {enzyme_id}: ', mutated_sequence)
            
            if mutated_sequence is None:
                print(f"Failed to extract sequence for enzyme {enzyme_id}")
                results[i].update({
                    'model_response': response,
                    'mutated_sequence': None,
                    'new_stability': None,
                    'stability_change': None,
                    'is_improvement': False,
                    'correct_format': False
                })
                continue
                
            # Calculate new stability
            new_stability = get_stability_score(mutated_sequence)
            
            results[i].update({
                'model_response': response,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability < original_stability,
                'correct_format': True
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            results[i].update({
                'model_response': response if 'response' in locals() else None,
                'mutated_sequence': None,
                'new_stability': None,
                'stability_change': None,
                'is_improvement': False,
                'correct_format': False
            })
            continue
    
    # Save updated results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'o1_stability_mutations.json', 'w') as f:
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
        with open(output_dir / 'o1_stability_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e
