import os
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from openfold.utils.protein import OFProtein
from openfold.utils.rigid_utils import atom14_to_atom37
from openfold.utils.pdb_utils import to_pdb
import pyrosetta

class StabilityRewardCalculator:
    def __init__(self, protein_model_path="facebook/esmfold_v1", device="cuda"):
        self.protein_model = self._load_protein_model(protein_model_path, device)
        self.device = device
        self.cached_structures = {}
        
        # Initialize PyRosetta
        pyrosetta.init()
        
    def _load_protein_model(self, model_path, device):
        """Load ESMFold model for structure prediction"""
        local_path = '/root/prO-1/model_cache/'
        if os.path.exists(local_path):
            model = EsmForProteinFolding.from_pretrained(local_path)
        else:
            model = EsmForProteinFolding.from_pretrained(model_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            model.save_pretrained(local_path)
            
        if device == "cuda":
            model = model.cuda()
        return model

    def predict_structure(self, sequence):
        """Predict protein structure using ESMFold"""
        if sequence in self.cached_structures:
            return self.cached_structures[sequence]
            
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        tokenized_input = tokenizer(
            [sequence], 
            return_tensors="pt", 
            add_special_tokens=False
        )['input_ids']
        
        if self.device == "cuda":
            tokenized_input = tokenized_input.cuda()

        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        pdb_file = self.convert_outputs_to_pdb(output)[0]
        
        # Cache the result
        self.cached_structures[sequence] = pdb_file
        return pdb_file
        
    def convert_outputs_to_pdb(self, outputs):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))
            
        output_dir = "predicted_structures"
        os.makedirs(output_dir, exist_ok=True)
        
        pdb_files = []
        for i, pdb in enumerate(pdbs):
            pdb_path = os.path.join(output_dir, f"structure_{i}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)
            pdb_files.append(pdb_path)
        return pdb_files

    def calculate_stability(self, sequence):
        """Calculate stability score using Rosetta"""
        # Get structure prediction
        pdb_file = self.predict_structure(sequence)
        
        # Load structure into PyRosetta
        pose = pyrosetta.pose_from_pdb(pdb_file)
        
        # Create score function
        scorefxn = pyrosetta.get_fa_scorefxn()
        
        # Calculate stability score
        stability_score = scorefxn(pose)
        
        # Normalize score (lower scores are better in Rosetta)
        # Convert to 0-1 range where 1 is most stable
        normalized_score = 1.0 / (1.0 + abs(stability_score))
        
        return normalized_score

def get_stability_reward(sequence):
    calculator = StabilityRewardCalculator()
    return calculator.calculate_stability(sequence)
