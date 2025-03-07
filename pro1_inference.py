import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from stability_reward import StabilityRewardCalculator
import re
import threading
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def extract_sequence_from_response(response):
    """Extract sequence from model response using regex pattern matching"""
    try:
        sequence_match = None

        # Look for all occurrences of \boxed{...} and take the last one
        boxed_matches = re.findall(r'\\boxed{([A-Z]+)}', response)
        if boxed_matches:
            sequence_match = boxed_matches[-1].strip()
            return sequence_match

        # Try alternative pattern without escaping the backslash and take the last one
        boxed_matches_alt = re.findall(r'boxed{([A-Z]+)}', response)
        if boxed_matches_alt:
            sequence_match = boxed_matches_alt[-1].strip()
            return sequence_match

        # Try to find any sequence-like content (continuous uppercase letters) after <answer> and take the last one
        answer_matches = re.findall(r'<answer>.*?([A-Z]{10,})', response, re.DOTALL)
        if answer_matches:
            sequence_match = answer_matches[-1].strip()
            return sequence_match

        # If no sequence found in expected format
        print("Warning: No sequence found in expected format, retry applier model or sequence gen")
        return None

    except Exception as e:
        print(f"Error extracting sequence: {e}")
        print("Response excerpt:")
        print(response[-200:])  # Print the last 200 characters
        return None
        
    except Exception as e:
        print(f"Error extracting sequence: {e}")
        print("Response excerpt:")
        print(response[-200:])  # Print the last 200 characters
        return None

def lm_sequence_applier(original_sequence, reasoning, max_attempts=3):
    """Use GPT-4o to extract the modified sequence based on reasoning"""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""
You are a helpful assistant that applies the mutations and modifications described in the reasoning to the original sequence.

Original sequence:
{original_sequence}

Proposed modifications:
{reasoning}

Given the natural language reasoning above, infer the mutations and modifications that the user wants to apply to the original sequence, and apply them. Return ONLY the modified sequence with all changes applied correctly in the \\boxed{{}} tag. ex. \\boxed{{MGYARRVMDGIGEVAV...}}. Go step by step listing all of the mutations and then applying them. IT IS CRUCIAL YOU APPLY THE MUTATIONS CORRECTLY AND RETURN THE MODIFIED SEQUENCE.
"""
        
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            print(f"Sending request to GPT-4o (attempt {attempt})...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can analyze natural language reasoning and apply the proposed mutations to the original sequence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=False
            )
            
            # Get the full response content
            full_response = response.choices[0].message.content
            print(f"GPT-4o response received (attempt {attempt})")
            
            modified_sequence = extract_sequence_from_response(full_response)
            
            if modified_sequence is None:
                print(f"Failed to extract sequence on attempt {attempt}")
                continue
            
            # Validate the sequence contains only valid amino acids
            valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_amino_acids for aa in modified_sequence):
                print(f"Invalid amino acids found on attempt {attempt}, trying again...")
                continue
            
            # Count and report differences between original and modified sequences
            if len(original_sequence) == len(modified_sequence):
                differences = sum(1 for a, b in zip(original_sequence, modified_sequence) if a != b)
                print(f"Found {differences} mutations in the sequence")
                
                # List the specific mutations
                if differences > 0:
                    mutations = []
                    for i, (orig, mod) in enumerate(zip(original_sequence, modified_sequence)):
                        if orig != mod:
                            mutations.append(f"{orig}{i+1}{mod}")
                    print(f"Mutations: {', '.join(mutations)}")
            else:
                length_diff = len(modified_sequence) - len(original_sequence)
                print(f"Sequence length changed by {length_diff} ({len(original_sequence)} → {len(modified_sequence)})")
                # This could be an insertion or deletion - more complex to report precisely
            
            # If no differences, warn but still return the sequence
            if modified_sequence == original_sequence:
                print("WARNING: Modified sequence is identical to original sequence!")
                
            return modified_sequence
            
        print("Max attempts reached without getting a valid sequence")
        return None
        
    except Exception as e:
        print(f"Error getting modified sequence from GPT-4o: {e}")
        return None

def run_inference(sequence, protein_data, model, tokenizer, stability_calculator=None, device="cuda", max_iterations=3, use_pdb=False, use_applier=False, original_stability_score=None, max_length=32768):
    """Run inference with the trained model and critic feedback loop"""

    if stability_calculator is None:
        stability_calculator = StabilityRewardCalculator()
    
    # Store the original sequence for reference throughout all iterations
    original_sequence = sequence
    
    # Store the base prompt for reuse
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with a protein sequence given below, as well as other useful information regarding the enzyme/reaction (if applicable): 

ENZYME NAME: {protein_data.get('name', 'Unknown')}
EC NUMBER: {protein_data.get('ec_number', 'Unknown')}
ENZYME SEQUENCE: {sequence}
SUBSTRATES: {', '.join(protein_data['reaction'][0]['substrates'])}
PRODUCTS: {', '.join(protein_data['reaction'][0]['products'])}
GENERAL INFORMATION: {protein_data.get('general_information', 'No additional information available')}
METALS/IONS: {', '.join(protein_data.get('metal_ions', ['None']))}
ACTIVE SITE RESIDUES (DO NOT MODIFY): {', '.join(protein_data.get('active_site_residues', ['None']))}
KNOWN MUTATIONS: {', '.join([f"{mutation['mutation']} ({mutation['effect']})" for mutation in protein_data.get('known_mutations', [{'mutation': 'None', 'effect': ''}])])}

Propose NOVEL mutations to optimize the stability of the protein given the information above. If applicable, be creative with your modifications, including insertions or deletions of sequences that may help improve stability (make sure to have good reasoning for these types of modifications). Ensure that you preserve the activity or function of the enzyme as much as possible.

****all reasoning must be specific to the protein and reaction specified in the prompt. cite scientific literature. consider similar proteins and reactions****

Provide detailed reasoning for each mutation, including the position number and the amino acid change (e.g., A23L means changing Alanine at position 23 to Leucine)."""

    if not use_applier:
        base_prompt += "\nReturn the modified sequence with all changes applied correctly in the \\boxed{{}} tag. ex. \\boxed{{MGYARRVMDGIGEVAV...}}. DO NOT INCLUDE ANY OTHER TEXT OR FORMATTING WITHIN THE BRACKETS. IT IS CRUCIAL YOU APPLY THE MUTATIONS CORRECTLY AND RETURN THE MODIFIED SEQUENCE in the \\boxed{{}} tag."
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
    top_performers = []
    while iteration < max_iterations:
        try:
            iteration += 1
            print(f"\n=== ITERATION {iteration} ===")
            
            prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein engineering tasks. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Think deeply and logically. |eot_id|><|start_header_id|>user<|end_header_id|>
{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            # Check token count and truncate if necessary
            token_count = len(tokenizer.encode(prompt))
            print(f"TOKEN COUNT: {token_count}")
            if token_count > max_length - 8192:
                print(f"Warning: Prompt too long ({token_count} tokens)...")

            # Generate response with the model using streaming
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Initialize the streamer
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
            
            # Create generation kwargs
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=8192,
                temperature=0.95,
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


            
            # If no sequence found, use OpenAI to generate the modified sequence
            if use_applier:
                print("Using LM sequence applier to generate the modified sequence...")
                # Use the sequence from the previous iteration or the original sequence
                modified_sequence = lm_sequence_applier(original_sequence, current_response)
            else: 
                # Try to extract sequence from response with regex
                modified_sequence = extract_sequence_from_response(current_response)
                
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
                
            print(f"Modified sequence length: {len(modified_sequence)}")
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
            current_result = {
                "iteration": iteration + 1,
                "sequence": modified_sequence,
                "stability_score": stability_score,
                "response": current_response,
            }
            
            iteration_history.append(current_result)
            
            # Update top performers list
            top_performers.append(current_result)
            top_performers.sort(key=lambda x: x['stability_score'])
            top_performers = top_performers[:10]  # Keep only top 10
                
        except KeyboardInterrupt:
            print("\n=== INTERRUPTED BY USER ===")
            print(f"\nBest score achieved: {best_score}")
            print(f"\nBest response:\n{best_response}")
            print(f"\nSaving top 10 performers to 'top_performers.json'...")
            
            with open('top_performers.json', 'w') as f:
                json.dump(top_performers, f, indent=2)
                
            return best_score, last_successful_sequence, best_response
            
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

# use this if you are extracting from uniprot mutagenesis data
def parse_mutagenesis_data(json_file_path):
    """Parse CA2 mutagenesis data from JSON file and format it like known mutations in enzyme_data"""
    import json
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract mutagenesis features
    mutations = []
    for feature in data.get('features', []):
        if feature.get('type') == 'Mutagenesis':
            location = feature.get('location', {}).get('start', {}).get('value')
            if location is None:
                continue
                
            original_seq = feature.get('alternativeSequence', {}).get('originalSequence')
            alt_seqs = feature.get('alternativeSequence', {}).get('alternativeSequences', [])
            description = feature.get('description', '')
            
            if not original_seq or not alt_seqs:
                continue
                
            # Format each mutation like in enzyme_data
            for alt_seq in alt_seqs:
                mutation = f"{original_seq}{location}{alt_seq}"
                mutations.append({
                    "mutation": mutation,
                    "effect": description
                })
    
    return mutations

if __name__ == "__main__":
    """
    Example usage of the protein engineering inference system.
    To use this system for your own protein:
    1. Modify the configuration below with your protein details
    2. (optional) add OPENAI_API_KEY to .env file
    3. Run the script
    """

    # === USER CONFIGURATION ===

    use_lm_applier = False # set to True if you want to use the LM sequence applier (highly recommended)
    
    # Your protein sequence
    PROTEIN_SEQUENCE = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK" ## plain sequence here 
    
    # Define your enzyme information
    PROTEIN_DATA = {
        # Basic protein information
        "name": "Human Carbonic Anhydrase II",  # Name of your protein
        "ec_number": "4.2.1.1",  # EC number if available
            
        # Reaction details
        "reaction": [{
            "substrates": ["Carbon dioxide", "Water"],  # List of substrates
            "products": ["Bicarbonate", "H+"]  # List of products
        }],
        
        # Important residues and cofactors
        "metal_ions": [],  # List any metal ions or cofactors (e.g. ['Zn+2', 'Mg+2'])
        "active_site_residues": [],  # example ["H64", "H19", "H198", "H200"]
        
        # Additional information (can be left empty)
        "general_information": """
        """, ## replace this string with your general information about the protein
        
        # Known mutations (optional)
        "known_mutations": [
            # takes list of dictionaries, each with mutation and effect
            # Example mutation, must be in this format
            # {
            #     "mutation": "W19A",
            #     "effect": "Description of the mutation's effect"
            # },
            # Add more mutations as needed
        ]
    }

    # Model configuration
    MODEL_CONFIG = {
        "checkpoint_path": "all-lm-grpo-mega-run/checkpoints/checkpoint-20250225-025056-step40", # change based on the checkpoint you want to use
        "max_iterations": 10,  # Number of optimization iterations
        "max_length": 32768  # Maximum sequence length
    }

    # === SETUP AND EXECUTION ===
    
    if not os.getenv("OPENAI_API_KEY") and use_lm_applier:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/meta-Llama-3.1-8B-Instruct",
        max_seq_length=MODEL_CONFIG["max_length"],
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.6, ## tune this to use more or less VRAM, note that esmfold requires at least 10 GB
    )
    
    # Load adapter weights
    print(f"Loading adapter from {MODEL_CONFIG['checkpoint_path']}...")
    model.load_adapter(MODEL_CONFIG['checkpoint_path'])
    FastLanguageModel.for_inference(model)

    # Initialize stability calculator
    stability_calculator = StabilityRewardCalculator()
    if PROTEIN_SEQUENCE != "":
        original_stability_score = stability_calculator.calculate_stability(PROTEIN_SEQUENCE)
    else:
        raise ValueError("PROTEIN_SEQUENCE is required")

    # Run optimization
    result = run_inference(
        sequence=PROTEIN_SEQUENCE,
        protein_data=PROTEIN_DATA,
        model=model,
        tokenizer=tokenizer,
        stability_calculator=stability_calculator,
        max_iterations=MODEL_CONFIG["max_iterations"],
        max_length=MODEL_CONFIG["max_length"],
        original_stability_score=original_stability_score, 
        use_applier=use_lm_applier
    )

    # Print and save final results
    print("\n=== FINAL RESULTS ===")
    if result:
        print("\nBest response:")
        print(result[2])
        print("\nBest sequence:")
        print(result[1]) 
        print("\nBest stability score:")
        print(result[0])
        print("\nImprovement (ddg):")
        ddg = result[0] - original_stability_score
        print(ddg)

        # Create results dictionary
        results_data = {
            "best_result": {
                "stability_score": result[0],
                "sequence": result[1],
                "response": result[2],
                "ddg": ddg
            }
        }

        # Save to JSON file
        output_file = "my_results.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"\nResults saved to {output_file}")

    else:
        print("No results returned from inference")
        with open("my_results.json", "w") as f:
            json.dump({"error": "No results returned from inference"}, f, indent=4)