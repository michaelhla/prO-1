# Pro-1

[![GitHub](https://img.shields.io/badge/GitHub-michaelhla/pro--1-181717?logo=github)](https://github.com/michaelhla/pro-1)
[![Twitter](https://img.shields.io/badge/Twitter-@hla__michael-1DA1F2?logo=twitter&style=social)](https://twitter.com/hla_michael)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-mhla/pro--1-yellow)](https://huggingface.co/mhla/pro-1)
[![Blog Post](https://img.shields.io/badge/Blog-pro--1-red)](https://michaelhla.com/blog/pro1.html)

Pro-1 is a reasoning model trained using GRPO towards a physics based reward function for protein stability.

It takes in a protein sequence + text description of the protein + effects of previous engineering attempts, reasons over the information given, and proposes modifications to improve the stability of the given sequence. 

![Pro-1 hCA II](pro1-grpo.gif)

# Running Pro-1:

requirements: 
- NVIDIA GPU instance with drivers installed, works best on A100 80GB but can run on smaller GPUs
- at least 60 GB of storage for 8b, 200 gb for 70b

1. after cloning the repo and ssh'ing into the GPU instance, run:

```
bash setup.sh
source venv/bin/activate 
```

2. to download the adapter weights run 

```
bash hf_download.sh
```

If you want to use a different checkpoint, modify the hf_download script and replace all_lm_grpo_mega_run with your checkpoint name. Default is the 8b creativity tuned model. 

(optional) create a .env file in the main pro-1 directory and paste OPENAI_API_KEY=your_api_key_here. This is only necessary if you want to use the LM sequence applier. You will need to set use_lm_applier=True in pro1_inference.py

3. set your fields for your protein data in pro1_inference.py

```
    # Your protein sequence
    PROTEIN_SEQUENCE = "" ## plain sequence here 
    
    # Define your enzyme information
    ENZYME_DATA = {
        # Basic enzyme information
        "name": "Human Carbonic Anhydrase II",  # Name of your enzyme
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

        Brief description of your enzyme and any relevant literature.
        Include key findings from previous studies or important characteristics.
        Most flexible field in prompt, replace this string with whatever you want here

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
```

4. Modify the model configs

```
    # Model configuration
    MODEL_CONFIG = {
        "checkpoint_path": "all-lm-grpo-mega-run/checkpoints/checkpoint-20250225-025056-step40", # change based on the checkpoint you want to use
        "max_iterations": 10,  # Number of optimization iterations
        "max_length": 32768  # Maximum sequence length
    }
```

5. run the script with 

```
python pro1_inference.py
```
Note: While the model was specifically trained on enzymes, it should work for any protein sequence. Curious to hear if this is true!

Disclaimer: This is a preview version and as a result the model can be very dumb. Always double check sure your modified sequences have the correct mutations applied. Assume all references from the model are hallucinated. 



