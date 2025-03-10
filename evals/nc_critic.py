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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
from stability_reward import StabilityRewardCalculator
import threading
from peft import PeftModel, get_peft_model, LoraConfig
from unsloth import FastLanguageModel
from pro1_inference import run_inference

# Results summary (pass@3):
# Number of enzymes processed: 38
# Number of successful improvements: 25
# Success rate: 65.8%

load_dotenv()

MAX_INPUT_LENGTH = 32768

# Initialize stability calculator
stability_calculator = StabilityRewardCalculator()

def get_stability_score(sequence: str) -> float:
    """Calculate protein stability score using ESMFold and PyRosetta"""
    return stability_calculator.calculate_stability(sequence)

def propose_mutations(sequence: str, enzyme_data: Dict, original_stability: float, model: FastLanguageModel, tokenizer: AutoTokenizer) -> str:
    """Get mutation proposals from DeepSeek"""
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data['name']}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data['general_information']}
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in enzyme_data['active_site_residues']])}

Propose mutations to optimize the stability of the enzyme given the information above. YOUR MUTATIONS MUST BE POSITION SPECIFIC TO THE SEQUENCE. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning. 

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions.****

COPY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. YOU MUST FOLLOW THIS INSTRUCTION/FORMAT. EX; \\boxed{{MALWMTLLLLPVPDGPK...}}"""

    response = run_inference(
        sequence=sequence, 
        enzyme_data=enzyme_data, 
        model=model, 
        tokenizer=tokenizer, 
        stability_calculator=stability_calculator, 
        max_iterations=1, 
        original_stability_score=original_stability, 
        )
    return response

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

def apply_mutations(sequence: str, mutations: List[str]) -> str:
    """Apply a list of mutations to a sequence"""
    seq_list = list(sequence)
    for mutation in mutations:
        # Parse mutation format 'X123Y'
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        pos = int(mutation[1:-1]) - 1  # Convert to 0-indexed
        
        if seq_list[pos] != orig_aa:
            print(f"Warning: Expected {orig_aa} at position {pos+1} but found {seq_list[pos]}")
            continue
            
        seq_list[pos] = new_aa
    
    return ''.join(seq_list)

def main():
    # Load enzyme sequences
    with open('data/transformed_brenda.json', 'r') as f:
        enzymes = json.load(f)
    # Load training dataset to exclude those enzymes
    with open('data/train_dataset.json', 'r') as f:
        train_dataset = json.load(f)
    
    # Filter out enzymes that are in the training set
    eligible_enzymes = {k:v for k,v in enzymes.items() if k not in train_dataset}
    
    selected_enzymes = random.sample(list(eligible_enzymes.items()), 40)
    # Save selected enzymes to a new dataset 
    selected_dataset = {enzyme_id: data for enzyme_id, data in selected_enzymes}
    
    # Create results directory if it doesn't exist
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save selected enzymes dataset
    with open(output_dir / 'selected_enzymes.json', 'w') as f:
        json.dump(selected_dataset, f, indent=2)
    
    results = []
    
    checkpoint_path = "all-lm-grpo-mega-run/checkpoints/checkpoint-20250225-025056-step40"
    
    # Initialize model using unsloth's FastLanguageModel
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/meta-Llama-3.1-8B-Instruct",
        max_seq_length=MAX_INPUT_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.6,
    )
    
    # Load the adapter weights
    print(f"Loading adapter from {checkpoint_path}...")
    model.load_adapter(checkpoint_path)
    
    # Set model to inference mode
    FastLanguageModel.for_inference(model)

    for enzyme_id, data in tqdm(selected_enzymes, desc="Processing enzymes"):
        sequence = data['sequence']
        try:
            original_stability = get_stability_score(sequence)
        except Exception as e:
            print(f"Error calculating stability for enzyme {enzyme_id}: {str(e)}")
            print('Sequence length: ', len(str(sequence)))
            continue
        
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
            response = propose_mutations(sequence, enzyme_data, original_stability, model, tokenizer)
            print(f"Generated response for enzyme {enzyme_id}: {response[2]}")  # Print the generation for each turn
            # Extract mutated sequence
            mutated_sequence = response[1]
            
            
            if mutated_sequence is None:
                print(f"Failed to extract sequence for enzyme {enzyme_id}")
                continue
                
            # Calculate new stability
            print('MUTATED SEQUENCE: ', mutated_sequence)
            new_stability = response[0]
            print('ddg: ', new_stability-original_stability)
            
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'model_response': response[2],
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability < original_stability,
                'correct_format': True
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            results.append({
                'enzyme_id': enzyme_id,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'model_response': 'ERROR',
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
    
    with open(output_dir / 'pro1_7b_stability_mutations.json', 'w') as f:
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
    try:
        torch.cuda.empty_cache()
        main()
    except Exception as e:
        print(f"Critical error encountered: {str(e)}")
        # Save current results before exit
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'pro1_7b_ reroll_stability_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e
