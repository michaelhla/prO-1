import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from typing import List, Tuple, Dict, Optional
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from stability_reward import StabilityRewardCalculator
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Results summary:
# Number of mutations evaluated: 259
# Number of successful improvements: 120
# Success rate: 46.3%
# Max stability improvement: -1548.566


# Results summaryn (run 2, top esm scores):
# Number of enzymes evaluated: 37
# Number of successful improvements: 20
# Success rate: 54.1%

class ESMAttentionMutator:
    """
    A class that uses ESM2 to analyze protein sequences, extract attention scores,
    and generate mutations based on those scores.
    """
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        """
        Initialize the ESM model.
        
        Args:
            model_name: The name of the ESM model to use
        """
        # Load the ESM model and tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"ESM2 model {model_name} loaded successfully")
    
    def get_attention_scores(self, sequence: str) -> torch.Tensor:
        """
        Get attention scores for a protein sequence.
        
        Args:
            sequence: The protein sequence to analyze
            
        Returns:
            Attention scores tensor
        """
        # Tokenize the sequence
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Extract attention weights from the last layer
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attentions = outputs.attentions[-1]
        
        # Average across heads to get a single attention score per position
        # Shape: [seq_len, seq_len]
        avg_attention = attentions.mean(dim=1).squeeze(0)
        
        return avg_attention
    
    def identify_mutation_sites(self, attention_scores: torch.Tensor, k: int) -> List[int]:
        """
        Identify the bottom k positions based on attention scores.
        
        Args:
            attention_scores: The attention scores tensor
            k: Number of positions to select
            
        Returns:
            List of indices to mutate
        """
        # Sum attention across the sequence dimension to get importance per position
        # Skip the first token (CLS) and last token (EOS)
        position_importance = attention_scores[1:-1, 1:-1].sum(dim=1)
        
        # Get the indices of the top k positions
        top_k_indices = torch.argsort(position_importance, descending=True)[:k].cpu().numpy()
        
        # Add 1 to account for the CLS token offset and convert to 0-indexed positions in the sequence
        return [int(idx) + 1 for idx in bottom_k_indices]
    
    def generate_mutations(self, sequence: str, mutation_sites: List[int]) -> str:
        """
        Generate a single mutated sequence by applying mutations at all specified sites.
        
        Args:
            sequence: The original protein sequence
            mutation_sites: List of indices to mutate
            
        Returns:
            A single mutated sequence with all mutations applied
        """
        # Convert sequence to list for easier manipulation
        seq_list = list(sequence)
        
        # For each mutation site, predict likely substitutions
        for site in mutation_sites:
            # Prepare masked sequence
            masked_seq = seq_list.copy()
            original_aa = masked_seq[site-1]  # Convert to 0-indexed
            masked_seq[site-1] = self.tokenizer.mask_token
            masked_sequence = "".join(masked_seq)
            
            # Tokenize the masked sequence
            inputs = self.tokenizer(masked_sequence, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Find the position of the mask token in the tokenized input
            mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()
            
            # Get predictions for the masked position
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits[0, mask_token_index]
            
            # Get the top amino acid predictions (excluding the original)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            aa_indices = [self.tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]
            aa_scores = predictions[aa_indices]
            
            # Sort by score and filter out the original amino acid
            aa_scores_dict = {aa: score.item() for aa, score in zip(amino_acids, aa_scores)}
            sorted_aa = sorted(aa_scores_dict.items(), key=lambda x: x[1], reverse=True)
            filtered_aa = [aa for aa, _ in sorted_aa if aa != original_aa]
            
            # Take the top prediction and apply it to the sequence
            if filtered_aa:
                top_aa = filtered_aa[0]
                seq_list[site-1] = top_aa
        
        # Return the final mutated sequence with all mutations applied
        return "".join(seq_list)
    
    def run_evaluation(self, sequence: str, k: int = 5) -> Dict:
        """
        Run the full evaluation pipeline.
        
        Args:
            sequence: The protein sequence to analyze
            k: Number of positions to mutate
            
        Returns:
            Dictionary with evaluation results
        """
        # Get attention scores
        attention_scores = self.get_attention_scores(sequence)
        
        # Identify mutation sites
        mutation_sites = self.identify_mutation_sites(attention_scores, k)
        
        # Generate a single mutated sequence with all mutations
        mutated_sequence = self.generate_mutations(sequence, mutation_sites)
        
        # Prepare results
        results = {
            "original_sequence": sequence,
            "mutation_sites": mutation_sites,
            "mutated_sequence": mutated_sequence,
            "attention_scores": attention_scores.cpu().numpy().tolist()  # Convert to list for JSON serialization
        }
        
        return results


def main():
    # Load previous results
    with open('results/selected_enzymes.json', 'r') as f:
        selected_enzymes = json.load(f)
    
    # Initialize the evaluator and stability calculator
    evaluator = ESMAttentionMutator()
    stability_calculator = StabilityRewardCalculator()
    
    results = []
    
    for enzyme_id, data in tqdm(selected_enzymes.items(), desc="Processing enzymes"):
        sequence = data['sequence']
        
        try: 
            # Calculate original stability
            original_stability = stability_calculator.calculate_stability(sequence)
        except Exception as e:
            print(f"Error calculating original stability for {enzyme_id}: {str(e)}")
            continue

        try:    
            # Run ESM attention-based mutation
            eval_results = evaluator.run_evaluation(sequence, k=7)  # Generate mutations at 7 sites
            
            # Calculate stability for the mutated sequence
            mutated_sequence = eval_results["mutated_sequence"]
            new_stability = stability_calculator.calculate_stability(mutated_sequence)
            
            results.append({
                'enzyme_id': enzyme_id,
                'mutation_sites': eval_results["mutation_sites"],
                'original_sequence': sequence,
                'original_stability': original_stability,
                'mutated_sequence': mutated_sequence,
                'new_stability': new_stability,
                'stability_change': new_stability - original_stability,
                'is_improvement': new_stability < original_stability
            })
            
        except Exception as e:
            print(f"Error processing enzyme {enzyme_id}: {str(e)}")
            results.append({
                'enzyme_id': enzyme_id,
                'mutation_sites': None,
                'original_sequence': sequence,
                'original_stability': original_stability,
                'mutated_sequence': None,
                'new_stability': None,
                'stability_change': None,
                'is_improvement': False
            })
            continue
        
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'esm_attention_stability_mutations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    total_attempts = len(results)
    successful_attempts = sum(1 for r in results if r['is_improvement'])
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\nResults summary:")
    print(f"Number of enzymes evaluated: {total_attempts}")
    print(f"Number of successful improvements: {successful_attempts}")
    print(f"Success rate: {success_rate:.1f}%")
    
    stability_changes = [r['stability_change'] for r in results if r['stability_change'] is not None]
    if stability_changes:
        print(f"Max stability improvement: {min(stability_changes):.3f}")
    else:
        print("No valid stability improvements found")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error encountered: {str(e)}")
        # Save current results before exit
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'esm_attention_stability_mutations_error_state.json', 'w') as f:
            if 'results' in locals():
                json.dump(results, f, indent=2)
            else:
                json.dump({"error": "Failed before results initialization"}, f, indent=2)
        raise e
