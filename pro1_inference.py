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
from google import genai
import os
from dotenv import load_dotenv
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import openai
from openai import OpenAI

load_dotenv()

MAX_LENGTH=32768

def extract_sequence_from_response(response):
    """Extract sequence from model response using regex pattern matching"""
    try:
        # Look for sequence in \boxed{...}
        sequence_match = re.search(r'\\boxed{([A-Z]+)}', response)
        if sequence_match:
            return sequence_match.group(1).strip()
        
        # Try alternative pattern without escaping the backslash
        sequence_match = re.search(r'boxed{([A-Z]+)}', response)
        if sequence_match:
            return sequence_match.group(1).strip()
            
        # Try to find any sequence-like content (continuous uppercase letters)
        sequence_match = re.search(r'<answer>.*?([A-Z]{10,})', response, re.DOTALL)
        if sequence_match:
            return sequence_match.group(1).strip()
        
        # Print a portion of the response to debug
        print("Response excerpt for debugging:")
        print(response[-500:])  # Print the last 500 characters
        
        # If no sequence found in expected format
        print("Warning: No sequence found in expected format, retry applier model or sequence gen")
        return None
        
    except Exception as e:
        print(f"Error extracting sequence: {e}")
        print("Response excerpt:")
        print(response[-200:])  # Print the last 200 characters
        return None

def lm_sequence_applier(original_sequence, reasoning, max_attempts=3):
    """Use OpenAI to extract the modified sequence based on reasoning"""
    try:
        client = OpenAI()
        
        prompt = f"""
You are a helpful assistant that applies the mutations and modifications described in the reasoning to the original sequence.

Original sequence:
{original_sequence}

Proposed modifications:
{reasoning}

Given the natural language reasoning above, infer the mutations and modifications that the user wants to apply to the original sequence, and apply them. Return ONLY the modified sequence with all changes applied correctly in the \\boxed{{}} tag. ex. \\boxed{{MGYARRVMDGIGEVAV...}}. IT IS CRUCIAL YOU APPLY THE MUTATIONS CORRECTLY AND RETURN THE MODIFIED SEQUENCE.
"""
        
        attempt = 0
        
        while attempt < max_attempts:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can analyze natural language reasoning and apply the proposed mutations to the original sequence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            print(f"OpenAI response (attempt {attempt + 1}): {response.choices[0].message.content.strip()}")
            
            modified_sequence = extract_sequence_from_response(response.choices[0].message.content.strip())
            
            if modified_sequence is None:
                print(f"Failed to extract sequence on attempt {attempt + 1}")
                attempt += 1
                continue
            
            # Validate the sequence contains only valid amino acids
            valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_amino_acids for aa in modified_sequence):
                print(f"Invalid amino acids found on attempt {attempt + 1}, trying again...")
                attempt += 1
                continue
                
            return modified_sequence
            
        print("Max attempts reached without getting a valid sequence")
        return None
        
    except Exception as e:
        print(f"Error getting modified sequence from OpenAI: {e}")
        return None

def run_inference(sequence, enzyme_data, model, tokenizer, stability_calculator=None, device="cuda", max_iterations=3, use_pdb=False, use_applier=True, original_stability_score=None):
    """Run inference with the trained model and critic feedback loop"""

    if stability_calculator is None:
        stability_calculator = StabilityRewardCalculator()
    
    # Store the original sequence for reference throughout all iterations
    original_sequence = sequence
    
    # Store the base prompt for reuse
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

Provide detailed reasoning for each mutation, including the position number and the amino acid change (e.g., A23L means changing Alanine at position 23 to Leucine)."""

    # Initialize conversation history
    conversation_history = []
    
    best_response = None
    best_score = float('inf')
    
    # Keep track of sequences and responses across iterations
    previous_sequence = original_sequence
    last_successful_response = None
    last_successful_sequence = None
    
    # Store all iterations for reference
    iteration_history = []
    
    iteration = 0
    while iteration < max_iterations:
        try:
            print(f"\n=== ITERATION {iteration + 1} ===")
            
            prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Think deeply and logically. |eot_id|><|start_header_id|>user<|end_header_id|>
{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            # Check token count and truncate if necessary
            token_count = len(tokenizer.encode(prompt))
            if token_count > MAX_LENGTH - 4096:
                print(f"Warning: Prompt too long ({token_count} tokens)...")

            # Generate response with the model using streaming
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Initialize the streamer
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
            
            # Create generation kwargs
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            # Create a thread to run the generation
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Initialize response collection
            current_response = ""
            print("\nModel response (streaming):")
            
            # Collect the response as it streams
            for new_text in streamer:
                current_response += new_text
                print(new_text, end="", flush=True)
            print("\n")  # Add newline after streaming completes
            
            thread.join()  # Ensure generation is complete
            
            # Count tokens in the response
            response_token_count = len(tokenizer.encode(current_response))
            print(f"Response token count: {response_token_count}")
            print(f"Total tokens (prompt + response): {token_count + response_token_count}")

            # Try to extract sequence from response first (in case it's still in the old format)
            modified_sequence = extract_sequence_from_response(current_response)
            
            # If no sequence found, use OpenAI to generate the modified sequence
            if use_applier:
                print("Using LM sequence applier to generate the modified sequence...")
                # Use the sequence from the previous iteration or the original sequence
                modified_sequence = lm_sequence_applier(original_sequence, current_response)
                
            # Check if sequence only contains valid amino acids
            if modified_sequence is None:
                print("No sequence could be generated, trying again...")
                continue
            
                
            # Update sequence tracking variables
            previous_sequence = last_successful_sequence if last_successful_sequence else original_sequence
            last_successful_response = current_response
            last_successful_sequence = modified_sequence

            print(f"Last successful sequence: {last_successful_sequence[:20]}... (length: {len(last_successful_sequence)})")
            
            # Add validation and debugging for the sequence
            if not isinstance(modified_sequence, str):
                print(f"Error: modified_sequence is not a string, it's {type(modified_sequence)}")
                continue
                
            if len(modified_sequence) == 0:
                print("Error: modified_sequence is empty")
                continue
                
            print(f"Sequence length: {len(modified_sequence)}")
            print(f"First 20 characters: {modified_sequence[:20]}")
                
            # Generate structure and calculate stability
            try:
                print("Predicting structure and calculating stability...")
                path_to_pdb = stability_calculator.predict_structure(modified_sequence)
                stability_score = stability_calculator.calculate_stability(modified_sequence)
                ddg = stability_score - original_stability_score
            except Exception as e:
                print(f"Error in stability calculation: {e}")
                print(f"Error occurred on line {e.__traceback__.tb_lineno}")
                continue
                
            # Simple scoring based on stability
            current_score = stability_score
            
            if current_score < best_score:
                best_score = current_score
                best_response = current_response
                
            print(f"Iteration {iteration + 1}: Current score = {current_score}, Best score = {best_score}")
            
            # Store this iteration in history
            iteration_history.append({
                "iteration": iteration + 1,
                "sequence": modified_sequence,
                "stability_score": stability_score,
                "response": current_response,
            })

            iteration += 1
                
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            print(f"Error occurred on line {e.__traceback__.tb_lineno}")
            # If we have a previous successful response, we'll try again with that
            iteration += 1
            if last_successful_response:
                print("Continuing with the previous successful response")
            else:
                print("No previous successful response, restarting from the beginning")
    
    # Print summary of all iterations
    print("\n=== SUMMARY OF ALL ITERATIONS ===")
    for item in iteration_history:
        print(f"Iteration {item['iteration']}: Score = {item['stability_score']}")
        print(f"Sequence: {item['sequence'][:20]}... (length: {len(item['sequence'])})")
        print("-" * 50)
    
    # Return the best sequence and response based on stability score
    if iteration_history:
        best_iteration = min(iteration_history, key=lambda x: x['stability_score'])
        return best_iteration['stability_score'], best_iteration['sequence'], best_iteration['response']
    return None, None, None 

if __name__ == "__main__":
    # Make sure to set your OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        # You can set it here if needed
        # os.environ["OPENAI_API_KEY"] = "your-api-key"

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

    original_stability_score = stability_calculator.calculate_stability(sequence)

    # Run inference
    result = run_inference(
        sequence=sequence,
        enzyme_data=enzyme_data,
        model=model,
        tokenizer=tokenizer,
        stability_calculator=stability_calculator, 
        max_iterations=10, 
        original_stability_score=original_stability_score
    )

    # Print final results
    print("\n=== FINAL RESULTS ===")
    if result:
        print("\nBest response:")
        print(result[2])
        print("\nBest sequence:")
        print(result[1])
        print("\nBest stability score:")
        print(result[0])
        print("\nImprovement (ddg):")
        print( original_stability_score - result[0])
    else:
        print("No results returned from inference")