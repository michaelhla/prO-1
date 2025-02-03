import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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

# Initialize stability calculator
stability_calculator = StabilityRewardCalculator()

def get_stability_score(sequence: str) -> float:
    """Calculate protein stability score using ESMFold and PyRosetta"""
    return stability_calculator.calculate_stability(sequence)

def propose_mutations(sequence: str, enzyme_data: Dict) -> str:
    """Get mutation proposals from GPT-4"""
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data['name']}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data['general_information']}
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in enzyme_data['active_site_residues']])}

Propose 3-7 mutations to optimize the stability of the enzyme given the information above. YOUR MUTATIONS MUST BE POSITION SPECIFIC TO THE SEQUENCE. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning. 

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions.****

COPY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. YOU MUST FOLLOW THIS INSTRUCTION/FORMAT. EX; \\boxed{{MALWMTLLLLPVPDGPK...}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert protein engineer focused on enzyme stabilization."},
            {"role": "user", "content": base_prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def extract_sequence(response: str) -> str:
    """Extract sequence from model response"""
    try:
        # Look for sequence in \boxed{} format
        if '\\boxed{' in response and '}' in response:
            start = response.find('\\boxed{') + 7
            end = response.find('}', start)
            return response[start:end].strip()
        return None
    except Exception:
        return None

def main():
    # Load selected enzyme sequences
    with open('results/selected_enzymes.json', 'r') as f:
        selected_enzymes = json.load(f)
    
    results = []
    
    for enzyme_id, data in tqdm(selected_enzymes.items(), desc="Processing enzymes"):
        sequence = data['sequence']
        original_stability = get_stability_score(sequence)
        
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
            
            # Get model response
            response = propose_mutations(sequence, enzyme_data)
            
            # Extract mutated sequence
            mutated_sequence = extract_sequence(response)
            
            if mutated_sequence is None:
                print(f"Failed to extract sequence for enzyme {enzyme_id}")
                continue
                
            # Calculate new stability
            new_stability = get_stability_score(mutated_sequence)
            
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'model_response': response,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability > original_stability
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            continue
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / '4o_stability_mutations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nResults summary:")
    print(f"Number of enzymes processed: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Max stability improvement: {max(r['stability_change'] for r in results):.3f}")

if __name__ == "__main__":
    main()
