import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead
from stability_reward import StabilityRewardCalculator
import re
import threading
import json
from pathlib import Path
from google import genai
import os
from dotenv import load_dotenv
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import openai
from openai import OpenAI
from together import Together

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
    """Use DeepSeek-V3 to extract the modified sequence based on reasoning"""
    try:
        # Initialize Together client
        client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
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
            attempt += 1
            
            print(f"Sending request to DeepSeek-V3 (attempt {attempt})...")
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can analyze natural language reasoning and apply the proposed mutations to the original sequence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=False
            )
            
            # Get the full response content
            full_response = response.choices[0].message.content
            print(f"DeepSeek response received (attempt {attempt})")
            
            modified_sequence = extract_sequence_from_response(full_response)
            
            if modified_sequence is None:
                print(f"Failed to extract sequence on attempt {attempt}")
                continue
            
            # Validate the sequence contains only valid amino acids
            valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_amino_acids for aa in modified_sequence):
                print(f"Invalid amino acids found on attempt {attempt}, trying again...")
                continue
                
            return modified_sequence
            
        print("Max attempts reached without getting a valid sequence")
        return None
        
    except Exception as e:
        print(f"Error getting modified sequence from DeepSeek: {e}")
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
SUBSTRATES: {', '.join(enzyme_data['reaction'][0]['substrates'])}
PRODUCTS: {', '.join(enzyme_data['reaction'][0]['products'])}
GENERAL INFORMATION: {enzyme_data.get('general_information', 'No additional information available')}
METALS/IONS: {', '.join(enzyme_data.get('metal_ions', ['None']))}
ACTIVE SITE RESIDUES (DO NOT MODIFY): {', '.join(enzyme_data.get('active_site_residues', ['None']))}
KNOWN MUTATIONS: {', '.join([f"{mutation['mutation']} ({mutation['effect']})" for mutation in enzyme_data.get('known_mutations', [{'mutation': 'None', 'effect': ''}])])}

First summarize the GENERAL INFORMATION and KNOWN MUTATIONS sections provided above (be specific in your summary). Then, propose mutations to optimize the stability of the enzyme given the information above. If applicable, be creative with your modifications, including insertions or deletions of sequences that may help improve stability (make sure to have good reasoning for these types of modifications). Ensure that you preserve the activity or function of the enzyme as much as possible.

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
            if token_count > MAX_LENGTH - 8192:
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
    # Make sure to set your OpenAI API key
    lm_key = os.getenv("TOGETHER_API_KEY")
    if not lm_key:
        print("Warning: TOGETHER_API_KEY not found in environment variables")
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
    sequence = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    active_site_residues = [sequence[63] + "64"]

    if sequence == "":
        print('ERROR: No sequence provided')
        exit()

    # Format enzyme data dictionary
    enzyme_data = {
        "name": "Human Carbonic Anhydrase II",
        "ec_number": "4.2.1.1",
        "general_information": "Carbonic anhydrase II is an enzyme that catalyzes the hydration of carbon dioxide to form " +
            "bicarbonate and water. The side chain of Leu-198 is located at the mouth of " +
            "the active site hydrophobic pocket, and this pocket is required for substrate association. " +
            "Hydrophobic-->hydrophilic amino acid substitutions at the mouth of the pocket decrease kcat/KM for CO2 " +
            "hydration: the CO2 hydrase activities of Leu-198-->Glu, Leu-198-->His, and Leu-198-->Arg CAIIs are " +
            "diminished 19-fold, 10-fold, and 17-fold, respectively, relative to the wild-type enzyme; however, the " +
            "substitution of a compact aliphatic side chain for Leu-198 has a smaller effect on catalysis, in that " +
            "Leu-198-->Ala CAII exhibits only a 3-fold decrease in CO2 hydrase activity [Krebs, J. F., Rana, F., " +
            "Dluhy, R. A., & Fierke, C. A.(1993) Biochemistry (preceding paper in this issue)]. It is intriguing " +
            "that CO2 hydrase activity is not severely diminished in Leu-198-->Arg CAII, even though the side chain " +
            "of Arg-198 blocks the hydrophobic pocket. Therefore, the bulky side chain of Arg-198 must be reasonably " +
            "mobile in order to accommodate substrate association. Significantly, a residue larger than the wild-type " +
            "Leu-198 side chain does not necessarily block the substrate association pocket; e.g., the side chain of " +
            "Glu-198 packs against a hydrophobic patch, the net result of which is a wider mouth for the pocket." + 
            "\n\nPositions 1 to 10 are disordered in the wild-type enzyme."+ 
            "\nPositions 16 to 39 are disordered in the wild-type enzyme."+ 
            "\nPositions 250 to 260 are disordered in the wild-type enzyme."+ 
            f"\n{sequence[93]}94, {sequence[95]}96, {sequence[118]}119 are involved with coordiunation of the zinc ion"+ 
            """\n\na recent study from McKenna's group reported hCA II variants addressing both enzyme thermal stability and catalytic efficiency [21]. In particular, starting from a previous study which identified 10 possible mutations to produce thermostable hCA II variants [99], the substitution of three surface leucine residues (Leu100His, Leu224Ser and Leu240Pro) was used to obtain a triple hCA II mutant (named TS1) with enhanced thermal stability with respect to the wild-type enzyme of about 7 °C. This mutant was then used as starting point to obtain new derivatives with improved catalytic activity. Among these, the variant obtained introducing the contemporary substitutions Tyr7Phe and Asn67Gln retained the same thermal stability as TS1 but showed an improvement in rate constants of proton transfer by about six-fold with respect to the native enzyme [21]. These substitutions were chosen, considering that the double mutant Tyr7Phe/Asn67Gln was previously shown to have a nine-fold increase in the rate of proton transfer compared to the wild-type enzyme [100,101].

The substitution in hCA II of residues 23 and 203 with two cysteines (dsHCA II) [98] to reproduce a disulfide bridge conserved in many members of α-CA class [67,69,70,102,103,104,105,106] was also used to improve hCA II biotechnological properties. Indeed, thermal stability investigations of this variant showed that the melting temperature was enhanced of 14 °C compared to the wild-type enzyme, while the catalytic efficiency was similar to that of native enzyme (kcat/KM of 1.31 × 108 M−1·s−1) [98].

Finally, considering that protein stability can be improved by increasing the rigidity of surface loops with the inclusion of proline residues [107] and enhancing surface compactness via loop deletion [108,109], novel hCA II variants have been designed [110]. In particular, adopting the first approach, residue Glu234, which is positioned in a surface loop, has been substituted with a proline residue. Thermal stability analysis of this variant indicated an enhanced melting temperature of about 3 °C compared to the wild-type enzyme [110]. On the other side, adopting the second approach, the region encompassing residues 230–240 has been deleted, since it forms an extended surface loop with peculiar destabilizing features [110]. In particular, it is characterized by high thermal fluctuations and the presence of two solvent exposed hydrophobic residues previously shown to affect the thermal stability of the enzyme [21]. The choice was further supported by the observation that in SspCA, one of the most thermostable α-CAs (see Paragraph 3.1), this loop is absent [67]. The thermal characterization of this variant revealed, as expected, an improvement of the melting temperature with respect to the wild-type enzyme.

Altogether these data indicate that the punctual substitutions of critical residues can be effective to obtain active and stable hCA II variants that could be exploited in the harsh conditions required by current industrial processes for atmospheric CO2 sequestration.""", 
        "reaction": [{
            "substrates": ["Carbon dioxide", "Water"],
            "products": ["Bicarbonate", "H+"]
        }],
        "metal_ions": ['Zn+2'],
        "active_site_residues": active_site_residues,
    }

    # Add known mutations from P40881 data
    enzyme_data["known_mutations"] = parse_mutagenesis_data("data/ca2_mutagenesis.json")
    
    enzyme_data["known_mutations"].extend([
        {
            "mutation": "W19A",
            "effect": "kcat/km at pH 7.5 is 2.5fold lower than wild-type value, kcat/km at pH 8.8 is 2.2fold lower than wild-type value"
        },
        {
            "mutation": "Y200A", 
            "effect": "kcat/km at pH 7.5 is 3.5fold higher than wild-type value, kcat/km at pH 8.8 is 3.3fold higher than wild-type value"
        },
        {
            "mutation": "W19N",
            "effect": "kcat/km at pH 7.5 is 3.2fold lower than wild-type value, kcat/km at pH 8.8 is 2.4fold lower than wild-type value"
        },
        {
            "mutation": "W19F",
            "effect": "kcat/km at pH 7.5 is 5fold lower than wild-type value, kcat/km at pH 8.8 is 5.2fold lower than wild-type value"
        },
        {
            "mutation": "Y200S",
            "effect": "kcat/km at pH 7.5 is 3fold higher than wild-type value, kcat/km at pH 8.8 is 3.3fold higher than wild-type value"
        },
        {
            "mutation": "Y200F",
            "effect": "kcat/km at pH 7.5 is 3.6fold higher than wild-type value, kcat/km at pH 8.8 is 2.5fold lower than wild-type value"
        }
    ])

    original_stability_score = stability_calculator.calculate_stability(sequence)

    # Run inference
    result = run_inference(
        sequence=sequence,
        enzyme_data=enzyme_data,
        model=model,
        tokenizer=tokenizer,
        stability_calculator=stability_calculator, 
        max_iterations=200, 
        original_stability_score=original_stability_score
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
        output_file = "inference_results.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"\nResults saved to {output_file}")

    else:
        print("No results returned from inference")
        with open("inference_results.json", "w") as f:
            json.dump({"error": "No results returned from inference"}, f, indent=4)