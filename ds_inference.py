import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

def get_critic_feedback(pdb_file, stability_score, proposed_mutations):
    """Get feedback from the critic model based on structural and stability analysis"""
    critic_prompt = f"""As a protein structure expert, analyze the following:
PDB Structure: {pdb_file}
Stability Score: {stability_score}
Proposed Mutations: {proposed_mutations}

Provide specific feedback on:
1. Structural implications of the proposed mutations
2. Impact on protein stability
3. Suggested refinements to improve stability while maintaining function
Keep feedback concise and actionable."""

    # Initialize critic model (using same base model for now)
    critic_model = AutoModelForCausalLM.from_pretrained("./sft_final")
    critic_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    
    inputs = critic_tokenizer(critic_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = critic_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )
    return critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_inference(sequence, active_site_residues, pdb_file=None, stability_score=None, device="cuda", max_iterations=3):
    """Run inference with the trained model and critic feedback loop"""
    # Initialize model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    model = AutoModelForCausalLM.from_pretrained("./sft_final")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device == "cuda":
        model = model.cuda()

    # Format enzyme data dictionary to match grpo structure
    enzyme_data = {
        "name": "amide bond synthetase",
        "ec_number": "6.3.1",
        "general_information": "Amide bond synthetase is an enzyme that catalyzes the formation of amide bonds between a carbonyl group and an amine group. It is found in the liver of mammals and plays a crucial role in the synthesis of proteins.",
        "reaction": [{
            "substrates": ["C1=CC(=CC=C1C(=O)O)Cl", "C1COCCN1CCN"],
            "products": ["C1COCCN1CCNC(=O)C2=CC=C(C=C2)Cl", "[O-]P(=O)([O-])[O-]"]
        }],
        "metal_ions": [],
        "active_site_residues": active_site_residues
    }

    # Use the construct_prompt function format
    base_prompt = f"""You are an expert protein engineer in rational protein design. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {enzyme_data['name']}
EC NUMBER: {enzyme_data['ec_number']}
ENZYME SEQUENCE: {sequence}
GENERAL INFORMATION: {enzyme_data['general_information']}
SUBSTRATES: {', '.join(enzyme_data['reaction'][0]['substrates'])}
PRODUCTS: {', '.join(enzyme_data['reaction'][0]['products'])}
METALS/IONS: None
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in active_site_residues])}

Propose mutations to optimize the stability of the enzyme given the information above. Ensure that you preserve the activity or function of the enzyme as much as possible. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects (or does not affect) protein structure
2. How the mutation affects (or does not affect) protein function
3. The chemical properties of the amino acids and substrates/products

****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTIFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** 

Copy the final sequence in the brackets of \\boxed{{}} to enclose the sequence:"""

    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. If there is a single final answer, wrap it in \\boxed{{}}. User: {base_prompt}. Assistant:"""

    best_response = None
    best_score = float('inf')
    
    for iteration in range(max_iterations):
        # Add critic's previous feedback to the prompt if available
        if iteration > 0:
            prompt += f"\n\nCRITIC'S FEEDBACK: {critic_feedback}\nPlease refine your mutations based on this feedback."

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        current_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get critic feedback if PDB and stability score are available
        if pdb_file and stability_score is not None:
            critic_feedback = get_critic_feedback(pdb_file, stability_score, current_response)
            
            # Simple scoring based on stability (you might want to implement a more sophisticated scoring)
            current_score = stability_score
            
            if current_score > best_score:
                best_score = current_score
                best_response = current_response

    return best_response or current_response
