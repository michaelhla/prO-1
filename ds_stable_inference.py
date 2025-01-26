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

    # Format prompt similar to MCTS
    prompt = f"""You are an expert protein engineer. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: amide bond synthetase
EC NUMBER: 6.3.1
GENERAL INFORMATION: Amide bond synthetase is an enzyme that catalyzes the formation of amide bonds between a carbonyl group and an amine group. It is found in the liver of mammals and plays a crucial role in the synthesis of proteins.
SUBSTRATES: C1=CC(=CC=C1C(=O)O)Cl, C1COCCN1CCN
PRODUCTS: C1COCCN1CCNC(=O)C2=CC=C(C=C2)Cl, [O-]P(=O)([O-])[O-]
METALS/IONS: 
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in active_site_residues])}

Propose a few mutations that will optimize enzymatic activity given the substrates and products above. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects protein structure and function
2. The chemical properties of the amino acids and substrates/products
3. The position's importance in the protein sequence

For each mutation you propose, provide clear, scientific reasoning for why the mutation would be beneficial, ****USE YOUR KNOWLEDGE OF THIS SPECIFIC ENZYME AND REACTION****. Keep your response to under 100 words."""

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
