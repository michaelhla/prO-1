import os
import json
from datasets import Dataset
import random

# Get list of available structures
structure_files = set(os.listdir("predicted_structures"))


def construct_prompt(enzyme_data, sequence):
    """Construct prompt for a single enzyme"""
    # Get reaction, substrates and products from first reaction if available
    if enzyme_data.get('reaction'):
        reaction = random.choice(enzyme_data['reaction'])
        substrates = reaction['substrates'] if reaction else ['Unknown']
        products = reaction['products'] if reaction else ['Unknown']
    else:
        substrates = ['Unknown']
        products = ['Unknown']

    # Get metals/ions if available
    metal_ions = enzyme_data.get('metal_ions', ['None'])
    if not metal_ions:
        metal_ions = ['None']

    # Format known mutations text
    known_mutations_text = ""
    if enzyme_data.get('engineering'):
        known_mutations_text = "KNOWN MUTATIONS AND EFFECTS:\n" + ''.join([
            f"- {mut['mutation']}: {mut['effect']}\n" 
            for mut in enzyme_data.get('engineering', [])
        ])

    # Construct the prompt
    enzyme_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data.get('name', 'Unknown')}
EC NUMBER: {enzyme_data.get('ec_number', 'Unknown')}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data.get('general_information', 'No additional information available')}
SUBSTRATES: {', '.join(substrates)}
PRODUCTS: {', '.join(products)}
METALS/IONS: {', '.join(metal_ions)}
{known_mutations_text}

Propose mutations to optimize the stability of the enzyme given the information above. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects (or does not affect) protein structure
2. How the mutation affects (or does not affect) protein function
3. The chemical properties of the amino acids and substrates/products

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions****

COPY THE FINAL SEQUENCE AND ONLY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. DO NOT INCLUDE ANY OTHER TEXT OR FORMATTING WITHIN THE BRACKETS."""

    whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. Your thinking should be at least 4000 tokens. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively.<|eot_id|><|start_header_id|>user<|end_header_id|>
{enzyme_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return whole_prompt

def validate_and_construct_prompt(x):
    """Wrapper function to validate data before constructing prompt"""
    try:
        # Ensure required fields exist and are of correct type
        if 'sequence' not in x:
            print(f"Warning: Missing required field 'sequence'")
            return None
            
        # Convert all fields to strings where appropriate
        safe_data = {
            'name': str(x.get('name', 'Unknown')),
            'ec_number': str(x.get('ec_number', 'Unknown')),
            'sequence': str(x['sequence']),
            'general_information': str(x.get('general_information', 'No additional information available')),
            'reaction': [],
            'metal_ions': [],
            'engineering': [], 
            'orig_stab': x['orig_stab'] 
        }
        
        # Handle reaction data
        if 'reaction' in x and x['reaction']:
            safe_data['reaction'] = []
            for reaction in x['reaction']:
                if isinstance(reaction, dict):
                    safe_reaction = {
                        'substrates': [str(s) for s in reaction.get('substrates', [])],
                        'products': [str(p) for p in reaction.get('products', [])]
                    }
                    safe_data['reaction'].append(safe_reaction)
        
        # Handle metal ions
        if 'metal_ions' in x and x['metal_ions']:
            safe_data['metal_ions'] = [str(ion) for ion in x['metal_ions']]
            
        # Handle engineering data
        if 'engineering' in x and x['engineering']:
            safe_data['engineering'] = []
            for mut in x['engineering']:
                if isinstance(mut, dict):
                    safe_mut = {
                        'mutation': str(mut.get('mutation', '')),
                        'effect': str(mut.get('effect', ''))
                    }
                    safe_data['engineering'].append(safe_mut)
        
        # Create the dataset record
        result = {
            "prompt": construct_prompt(safe_data, safe_data['sequence']), 
            "sequences": safe_data['sequence'], 
            "orig_stabs": safe_data['orig_stab']
        }
        
        # Verify the output is valid
        if not isinstance(result['prompt'], str) or not isinstance(result['sequences'], str):
            print(f"Warning: Invalid output types - prompt: {type(result['prompt'])}, sequences: {type(result['sequences'])}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error processing record: {str(e)}")
        print(f"Problematic record: {json.dumps(x, default=str)}")
        return None

# Load and filter the transformed BRENDA data
with open("data/transformed_brenda.json", 'r') as f:
    data_dict = json.load(f)
    # Filter to only include enzymes with existing structures and keep track of keys
    data_list_with_ids = [
        (key, value) for key, value in data_dict.items()
        if f"{key}.pdb" in structure_files and value.get('orig_stab') is not None
    ]

# Create the dataset with strict validation
valid_count = 0
rejected_count = 0
valid_data_list = []
MAX_INPUT_LENGTH = 4096  # Define max length for prompts

for key, item in data_list_with_ids:
    processed = validate_and_construct_prompt(item)
    if processed is not None and len(processed['prompt']) < MAX_INPUT_LENGTH:
        valid_data_list.append(processed)
        valid_count += 1
    else:
        rejected_count += 1

# Create dataset from validated records
filtered_dataset = Dataset.from_list(valid_data_list)

# Save filtered data to JSON
with open("data/train_dataset.json", 'w') as f:
    json.dump(valid_data_list, f, indent=2)

print(f"\nProcessed {valid_count + rejected_count} total records")
print(f"Valid records: {valid_count}")
print(f"Rejected records: {rejected_count}")
print(f"Filtered dataset size: {len(filtered_dataset)}")

# Print prompt length statistics
prompt_lengths = [len(example['prompt']) for example in valid_data_list]
print("\nPrompt Length Statistics:")
print(f"Mean length: {sum(prompt_lengths) / len(prompt_lengths):.2f}")
print(f"Median length: {sorted(prompt_lengths)[len(prompt_lengths)//2]}")
print(f"Max length: {max(prompt_lengths)}")
print(f"Min length: {min(prompt_lengths)}")


