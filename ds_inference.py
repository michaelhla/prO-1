import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead
from stability_reward import StabilityRewardCalculator
import re

def get_critic_feedback(critic_model, critic_tokenizer, pdb_file, stability_score, base_output):
    """Get feedback from the critic model based on structural and stability analysis"""
    critic_prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> REASONING PROCESS HERE </think>
<answer> ANSWER HERE </answer>. Follow the formatting instructions exactly as specified. [USER]: As a protein structure expert, critique the following:

Scientist A's Suggestions: {base_output}
PDB Structure: {pdb_file}
Stability Score: {stability_score}

Provide hyperspecific feedback using the pdb structure generated and the stability score. If confident, suggest mutations that may remedy your critiques as well. [ASSISTANT]:"""
    inputs = critic_tokenizer(critic_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = critic_model.generate(
            **inputs,
            max_new_tokens=5000,
            do_sample=True,
            temperature=0.7
        )
    return critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

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

    # Use the construct_prompt function format
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data['name']}
EC NUMBER: {enzyme_data['ec_number']}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data['general_information']}
SUBSTRATES: {', '.join(enzyme_data['reaction'][0]['substrates'])}
PRODUCTS: {', '.join(enzyme_data['reaction'][0]['products'])}
METALS/IONS: None
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in enzyme_data['active_site_residues']])}

Propose mutations to optimize the stability of the enzyme given the information above. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects (or does not affect) protein structure
2. How the mutation affects (or does not affect) protein function
3. The chemical properties of the amino acids and substrates/products

****all reasoning must be specific to the enzyme and reaction specified in the prompt. cite scientific literature. consider similar enzymes and reactions****

COPY THE FINAL SEQUENCE IN THE BRACKETS OF \\boxed{{}} TO ENCLOSE THE SEQUENCE:"""

    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> REASONING PROCESS HERE </think>
<answer> ANSWER HERE </answer>. Follow the formatting instructions exactly as specified. [USER]: {base_prompt}. [ASSISTANT]:"""

    best_response = None
    best_score = float('inf')
    critic_feedback = None
    
    for iteration in range(max_iterations):
        # Add critic's previous feedback to the prompt if available
        if iteration > 0:
            prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> REASONING PROCESS HERE </think>
<answer> ANSWER HERE </answer>. Follow the formatting instructions exactly as specified. [USER]: 
CRITIC'S FEEDBACK ON YOUR DESIGNS: {critic_feedback}
RECONSIDER YOUR DESIGNS BASED ON THE FEEDBACK. [ASSISTANT]:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        current_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(current_response)
        
        # Extract sequence from response
        modified_sequence = extract_sequence_from_response(current_response)
        if modified_sequence is None:
            continue
            
        try:
            # Generate structure and calculate stability
            path_to_pdb = stability_calculator.predict_structure(modified_sequence)
            stability_score = stability_calculator.calculate_stability(modified_sequence)
            # Convert PDB file path to string content
            with open(path_to_pdb, 'r') as f:
                pdb_file = f.read()
        except Exception as e:
            print(f"Error calculating stability score: {e}")
            stability_score = None
        
        # Get critic feedback if PDB and stability score are available
        if pdb_file and stability_score is not None:
            critic_feedback = get_critic_feedback(model, tokenizer, pdb_file, stability_score, current_response)
            print(critic_feedback)
            
            # Simple scoring based on stability (you might want to implement a more sophisticated scoring)
            current_score = stability_score
            
            if current_score < best_score:
                best_score = current_score
                best_response = current_response
                
            print(f"Iteration {iteration + 1}: Current score = {current_score}, Best score = {best_score}")
            

    return best_response or current_response

if __name__ == "__main__":
    device = "cuda"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    peft_lora_path = "./sft_lora_output/checkpoint-27"
    
    # Load base model with same quantization as in GRPO
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=True  # Can keep cache for inference
    )
    
    # Load LoRA adapter directly
    # model = PeftModel.from_pretrained(base_model, peft_lora_path)
    model = base_model
    model.eval()  # Set to evaluation mode
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if device == "cuda":
    #     model = model.cuda()
    

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
        device=device, 
        stability_calculator=stability_calculator
    )
    print("\nFinal result:", result)
