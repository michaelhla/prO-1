import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead
from stability_reward import StabilityRewardCalculator
import re
import threading
import json
from unsloth import FastLanguageModel
from pathlib import Path

MAX_LENGTH=40960

def get_critic_feedback(model, tokenizer, pdb_file, stability_score, base_output):
    """Get feedback from the critic model based on structural and stability analysis"""
    critic_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 tokens. |eot_id|><|start_header_id|>user<|end_header_id|>
As a protein structure expert, critique the following:

Scientist A's Suggestions: {base_output}
PDB Structure: {pdb_file}
Stability Score: {stability_score}

Provide specific feedback using the PDB structure and stability score. Focus on structural issues that might affect stability. If confident, suggest specific mutations that may improve stability while preserving function.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    inputs = tokenizer(critic_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    
    return generated_text

def extract_sequence_from_response(response):
    """Extract sequence from model response using regex pattern matching"""
    try:
        # Look for sequence in \boxed{...}
        sequence_match = re.search(r'\\boxed{(.*?)}', response)
        if sequence_match:
            return sequence_match.group(1).strip()
        
        # If no sequence found in expected format
        print("Warning: No sequence found in \\boxed{} format")
        return None
        
    except Exception as e:
        print(f"Error extracting sequence: {e}")
        return None

def run_inference(sequence, enzyme_data, model, tokenizer, stability_calculator, device="cuda", max_iterations=3, pdb_file=None, stability_score=None):
    """Run inference with the trained model and critic feedback loop"""

    # Update prompt to match the unsloth GRPO training prompt with modifications
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data.get('name', 'Unknown')}
EC NUMBER: {enzyme_data.get('ec_number', 'Unknown')}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data.get('general_information', 'No additional information available')}
SUBSTRATES: {', '.join(enzyme_data['reaction'][0]['substrates'])}
PRODUCTS: {', '.join(enzyme_data['reaction'][0]['products'])}
METALS/IONS: {', '.join(enzyme_data.get('metal_ions', ['None']))}

Propose mutations to optimize the stability of the enzyme given the information above. If applicable, be creative with your modifications, including insertions or deletions of sequences that may help improve stability (make sure to have good reasoning for these types of modifications). Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects (or does not affect) protein structure
2. How the mutation affects (or does not affect) protein function
3. The chemical properties of the amino acids and substrates/products

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions****

COPY THE FINAL SEQUENCE AND ONLY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. DO NOT INCLUDE ANY OTHER TEXT OR FORMATTING WITHIN THE BRACKETS."""

    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 tokens. |eot_id|><|start_header_id|>user<|end_header_id|>
{base_prompt} MAKE SURE YOU COPY THE ORIGINAL SEQUENCE CORRECTLY WITH THE MUTATIONS APPLIED!!!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    best_response = None
    best_score = float('inf')
    critic_feedback = None
    
    # Keep track of the last successful response and sequence
    last_successful_response = None
    last_successful_sequence = None
    
    iteration = 0
    while iteration < max_iterations:
        try:
            # Add critic's previous feedback to the prompt if available
            if iteration > 0:
                if critic_feedback is not None:
                    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Think deeply and logically. |eot_id|><|start_header_id|>user<|end_header_id|>
You have made some modifications to an enzyme sequence prior. Feedback on your sequence is below. 
ORIGINAL SEQUENCE: {sequence}
PREVIOUS ITERATION: {prev_sequence}
MODIFIED SEQUENCE: {last_successful_sequence}
CRITIC'S FEEDBACK ON YOUR MODIFICATIONS: {critic_feedback}
RECONSIDER YOUR DESIGNS BASED ON THE FEEDBACK. REMEMBER TO COPY THE FINAL SEQUENCE AND ONLY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. MAKE SURE YOU COPY THE ORIGINAL SEQUENCE CORRECTLY WITH THE MUTATIONS APPLIED!!!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                elif last_successful_response:
                    # If we have a previous successful response but no critic feedback,
                    # ask for the sequence to be properly formatted
                    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Think deeply and logically. |eot_id|><|start_header_id|>user<|end_header_id|>
COPY THE FINAL SEQUENCE AND ONLY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE. MAKE SURE YOU COPY THE ORIGINAL SEQUENCE CORRECTLY WITH THE MUTATIONS APPLIED!!!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                else:
                    # If something went wrong and we don't have a previous successful response,
                    # restart with the original prompt
                    print('ERROR: something went wrong, restarting from the beginning')
                    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 tokens. |eot_id|><|start_header_id|>user<|end_header_id|>
{base_prompt} MAKE SURE YOU COPY THE ORIGINAL SEQUENCE CORRECTLY WITH THE MUTATIONS APPLIED!!!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            # Generate response with the model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True
                )
            
            current_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(current_response)

            # Extract sequence from response
            modified_sequence = extract_sequence_from_response(current_response)
            if modified_sequence is None:
                print("No sequence found in response, trying again...")
                continue
                
            # Store this as the last successful response and sequence
            last_successful_response = current_response
            last_successful_sequence = modified_sequence
                
            # Generate structure and calculate stability
            path_to_pdb = stability_calculator.predict_structure(modified_sequence)
            stability_score = stability_calculator.calculate_stability(modified_sequence)
            # Convert PDB file path to string content
            with open(path_to_pdb, 'r') as f:
                pdb_file = f.read()
            
            # Get critic feedback if PDB and stability score are available
            if pdb_file and stability_score is not None:
                critic_feedback = get_critic_feedback(model, tokenizer, pdb_file, stability_score, current_response)
                print(critic_feedback)
                
                # Simple scoring based on stability
                current_score = stability_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_response = current_response
                    
                print(f"Iteration {iteration + 1}: Current score = {current_score}, Best score = {best_score}")
                
            # Move to the next iteration only if everything succeeded
            iteration += 1
                
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            # If we have a previous successful response, we'll try again with that
            if last_successful_response:
                print("Continuing with the previous successful response")
            else:
                print("No previous successful response, restarting from the beginning")
                # We don't increment iteration here, so we'll retry the same iteration
    
    # Return the best response if we found one, otherwise the last successful response,
    # or if all else fails, the current response (which might be None)
    return best_response or last_successful_response or current_response

if __name__ == "__main__":
    # Set up constants
    checkpoint_path = "all-lm-grpo-mega-run/checkpoints/checkpoint-20250225-025056-step40"
    
    # Initialize model using unsloth's FastLanguageModel
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/meta-Llama-3.1-8B-Instruct",
        max_seq_length=MAX_LENGTH,
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
    
    # Initialize stability calculator
    stability_calculator = StabilityRewardCalculator()

    # Example sequence and active site residues
    sequence = "MGYARRVMDGIGEVAVTGAGGSVTGARLRHQVRLLAHALTEAGIPPGRGVACLHANTWRAIALRLAVQAIGCHYVGLRPTAAVTEQARAIAAADSAALVFEPSVEARAADLLERVSVPVVLSLGPTSRGRDILAASVPEGTPLRYREHPEGIAVVAFTSGTTGTPKGVAHSSTAMSACVDAAVSMYGRGPWRFLIPIPLSDLGGELAQCTLATGGTVVLLEEFQPDAVLEAIERERATHVFLAPNWLYQLAEHPALPRSDLSSLRRVVYGGAPAVPSRVAAARERMGAVLMQNYGTQEAAFIAALTPDDHARRELLTAVGRPLPHVEVEIRDDSGGTLPRGAVGEVWVRSPMTMSGYWRDPERTAQVLSGGWLRTGDVGTFDEDGHLHLTDRLQDIIIVEAYNVYSRRVEHVLTEHPDVRAAAVVGVPDPDSGEAVCAAVVVADGADPDPEHLRALVRDHLGDLHVPRRVEFVRSIPVTPAGKPDKVKVRTWFTD"
    active_site_residues = []

    if sequence == "":
        print('ERROR: No sequence provided')
        exit()

    # Format enzyme data dictionary
    enzyme_data = {
        "name": "amide bond synthetase",
        "ec_number": "6.3.1",
        "general_information": "Amide bond synthetase is an enzyme that catalyzes the formation of amide bonds between a carbonyl group and an amine group. It is found in the liver of mammals and plays a crucial role in the synthesis of proteins.",
        "reaction": [{
            "substrates": ["4-chlorobenzoic acid", "morpholinoethylamine", 'ATP'],
            "products": ["N-(2-morpholinoethyl)-4-chlorobenzamide", "phosphate", 'ADP']
        }],
        "metal_ions": [],
        "active_site_residues": active_site_residues
    }

    # Run inference
    result = run_inference(
        sequence=sequence,
        enzyme_data=enzyme_data,
        model=model,
        tokenizer=tokenizer,
        stability_calculator=stability_calculator
    )
    print("\nFinal result:", result)
