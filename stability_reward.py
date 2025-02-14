import os
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import pyrosetta_installer
pyrosetta_installer.install_pyrosetta()
import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.core.pose import make_pose_from_sequence
from pyrosetta.rosetta.core.chemical import ResidueTypeSet, ChemicalManager
from pyrosetta.rosetta.core.scoring import ScoreType
import time

class StabilityRewardCalculator:
    def __init__(self, protein_model_path="facebook/esmfold_v1", device="cuda"):
        self.device = torch.device(device)
        self.protein_model = self._load_protein_model(protein_model_path)
        self.cached_structures = {}

        # Initialize PyRosetta
        pyrosetta.init()

    def _load_protein_model(self, model_path):
        """Load ESMFold model for structure prediction"""
        start_time = time.time()
        local_path = '/root/prO-1/model_cache/'
        if os.path.exists(local_path):
            model = EsmForProteinFolding.from_pretrained(local_path)
        else:
            model = EsmForProteinFolding.from_pretrained(model_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            model.save_pretrained(local_path)

        # Move model to specified device
        model = model.to(self.device)
        print(f"ESM loading took {time.time() - start_time:.2f} seconds")
        return model

    def predict_structure(self, sequence, uniprot_id=None):
        """Predict protein structure using ESMFold"""
        if sequence in self.cached_structures:
            return self.cached_structures[sequence]

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        tokenized_input = tokenizer(
            [sequence], 
            return_tensors="pt", 
            add_special_tokens=False
        )['input_ids'].to(self.device)  # Move directly to specified device

        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        pdb_file = self.convert_outputs_to_pdb(output, uniprot_id)[0]

        # Cache the result
        self.cached_structures[sequence] = pdb_file
        print(f"Structure prediction took {time.time() - start_time:.2f} seconds")
        return pdb_file

    def convert_outputs_to_pdb(self, outputs, uniprot_id=None):
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
            if uniprot_id:
                pdb_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
            else:
                pdb_path = os.path.join(output_dir, f"{i}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)
            pdb_files.append(pdb_path)
        return pdb_files

    def calculate_stability(self, sequence):
        start_time = time.time()
        
        """Calculate stability score using Rosetta with optimization"""
        # Get structure prediction
        pdb_file = self.predict_structure(sequence)

        # Load structure into PyRosetta
        pose = pyrosetta.pose_from_pdb(pdb_file)

        # Create score function
        scorefxn = pyrosetta.get_fa_scorefxn()
        
        # Setup packer task for side chain optimization
        task = pyrosetta.standard_packer_task(pose)
        task.restrict_to_repacking()  # Only repack side chains, don't change sequence

        # Optimize side chain conformations
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
        packer.apply(pose)

        # Perform energy minimization
        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        min_mover.score_function(scorefxn)
        min_mover.apply(pose)

        # Calculate final stability score
        stability_score = scorefxn(pose)

        print(f"Stability score: {stability_score} calculated in {time.time() - start_time:.2f} seconds")
        return stability_score

## for testing, deprecated
def get_stability_reward(sequence):
    calculator = StabilityRewardCalculator()
    return calculator.calculate_stability(sequence)