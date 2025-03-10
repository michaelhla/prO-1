## what do we want to compare? 
# model size 
# openai vs deepseek 
# role of critic round
# role of reasoning model
# role of GRPO/SFT (without access to reward signal)

# speed to find a solution  (will have access to reward signal with critic)

# Results summary:
# Number of enzymes processed: 33
# Number of successful improvements: 8
# Success rate: 24.2%
# Max stability improvement: -536.414

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path
from tqdm import tqdm
from together import Together
from stability_reward import StabilityRewardCalculator
from dotenv import load_dotenv
from typing import List, Dict  # Add typing imports
from openai import OpenAI

load_dotenv()

# Initialize Together client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Initialize stability calculator
stability_calculator = StabilityRewardCalculator()



def get_stability_score(sequence: str) -> float:
    """Calculate protein stability score using ESMFold and PyRosetta"""
    return stability_calculator.calculate_stability(sequence)

def read_pdb_structure(pdb_path: str) -> str:
    """Read PDB file and return its contents as a string"""
    try:
        with open(pdb_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading PDB file {pdb_path}: {str(e)}")
        return "Structure information not available"

def calculate_and_save_structures(selected_enzymes: Dict) -> None:
    """Calculate and save structures for all enzymes"""
    print("Calculating structures for all enzymes...")
    for enzyme_id, data in tqdm(selected_enzymes.items(), desc="Calculating structures"):
        sequence = data['sequence']
        try:
            # Use the existing stability calculator to predict and save structure
            stability_calculator.predict_structure(sequence, uniprot_id=enzyme_id)
        except Exception as e:
            print(f"Error calculating structure for enzyme {enzyme_id}: {str(e)}")

def propose_mutations(sequence: str, enzyme_data: Dict) -> str:
    # Get the structure information from PDB file
    pdb_path = f"predicted_structures/{enzyme_data['enzyme_id']}.pdb"
    structure_info = read_pdb_structure(pdb_path)
    
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data['name']}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data['general_information']}
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in enzyme_data['active_site_residues']])}
PDB STRUCTURE: {structure_info}

Propose 3-7 mutations to optimize the stability of the enzyme given the information above. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning. 

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions.****

COPY THE FINAL SEQUENCE WITH THE MUTATIONS APPLIED IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. YOU MUST FOLLOW THIS INSTRUCTION/FORMAT. EX; \\boxed{{MALWMTLLLLPVPDGPK...}} Do not use any coloring or other formatting within the boxed term, we only want the sequence in those brackets."""

    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[
            {"role": "user", "content": base_prompt}
        ],
        stream=True
    )

    # Accumulate the streamed response
    full_response = ""
    for chunk in response:
        try: 
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        except: 
            print('Error in chunk: ', chunk)
    
    return full_response

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
    # Load selected enzyme sequences
    with open('results/selected_enzymes.json', 'r') as f:
        selected_enzymes = json.load(f)
    
    # Calculate and save structures for all enzymes first
    calculate_and_save_structures(selected_enzymes)
    
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
            # Format enzyme data for the prompt
            enzyme_data = {
                "enzyme_id": enzyme_id,  # Add enzyme_id to the data
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

            print(f'Mutated sequence {enzyme_id}: ', mutated_sequence)
            
            if mutated_sequence is None:
                print(f"Failed to extract sequence for enzyme {enzyme_id}")
                results.append({
                    'enzyme_id': enzyme_id,
                    'original_sequence': sequence,
                    'original_stability': original_stability,
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
            
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'model_response': response,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability < original_stability,
                'correct_format': True
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            print(f"Error occurred at line {sys.exc_info()[2].tb_lineno}")
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'model_response': response if 'response' in locals() else None,
                'mutated_sequence': None,
                'new_stability': None,
                'stability_change': None,
                'is_improvement': False,
                'correct_format': False
            })
            continue
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ds_r1_670_stability_mutations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nResults summary:")
    print(f"Number of enzymes processed: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Calculate max stability improvement
    stability_changes = [r['stability_change'] for r in results if r['stability_change'] is not None]
    if stability_changes:
        print(f"Max stability improvement: {min(stability_changes):.3f}")
    else:
        print("No valid stability improvements found")

def reprocess_failed_mutations():
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load results
    results_path = Path('results/ds_r1_670_stability_mutations.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Process entries with failed sequence extraction
    for entry in results:
        if entry['mutated_sequence'] is None and entry['model_response'] is not None:
            print(f"Reprocessing enzyme {entry['enzyme_id']}...")
            
            # Prompt for GPT-4 to identify mutations and format sequence
            prompt = f"""Given this protein engineering response, identify the mutations mentioned and apply them to the original sequence. 
            Output the final sequence in the format: \\boxed{{EXAMPLE MODIFIED SEQUENCE}}. YOU MUST FOLLOW THIS FORMATTING. 
            Do not use any formatting or other instructions. Just the sequence no color or other formatting.


            Original sequence: {entry['original_sequence']}
            Model response: {entry['model_response']}"""
            
            try:
                # Get GPT-4 response
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                
                gpt4_response = response.choices[0].message.content

                print('GPT-4 response: ', gpt4_response)
                
                # Extract sequence using existing function
                mutated_sequence = extract_sequence(gpt4_response)
                
                if mutated_sequence:
                    # Calculate new stability
                    new_stability = get_stability_score(mutated_sequence)
                    
                    # Update entry
                    entry.update({
                        'mutated_sequence': mutated_sequence,
                        'new_stability': new_stability,
                        'stability_change': new_stability - entry['original_stability'],
                        'is_improvement': new_stability < entry['original_stability'],
                        'correct_format': True
                    })
                    print(f"Successfully reprocessed enzyme {entry['enzyme_id']}")
                else:
                    print(f"Failed to extract sequence from GPT-4 response for enzyme {entry['enzyme_id']}")
            
            except Exception as e:
                print(f"Error reprocessing enzyme {entry['enzyme_id']}: {str(e)}")
                continue
    
    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Recalculate and print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nUpdated Results summary:")
    print(f"Number of enzymes processed: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Calculate max stability improvement
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
        # Update error state filename
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'ds_r1_stability_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e



