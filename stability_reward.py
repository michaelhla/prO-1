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
from multiprocessing import Pool

class StabilityRewardCalculator:
    def __init__(self, protein_model_path="facebook/esmfold_v1", device="cuda"):
        self.protein_model = self._load_protein_model(protein_model_path, device)
        self.device = device
        self.cached_structures = {}
        
        # Initialize PyRosetta
        pyrosetta.init()
        
    def _load_protein_model(self, model_path, device):
        """Load ESMFold model for structure prediction"""
        start_time = time.time()
        local_path = '/root/prO-1/model_cache/'
        if os.path.exists(local_path):
            model = EsmForProteinFolding.from_pretrained(local_path)
        else:
            model = EsmForProteinFolding.from_pretrained(model_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            model.save_pretrained(local_path)
            
        if device == "cuda":
            model = model.cuda()
        print(f"ESM loading took {time.time() - start_time:.2f} seconds")
        return model

    def predict_structure(self, sequences):
        """Predict protein structure using ESMFold"""
        if not isinstance(sequences, list):
            sequences = [sequences]
        
        uncached_sequences = []
        uncached_indices = []
        results = [None] * len(sequences)
        
        for i, seq in enumerate(sequences):
            if seq in self.cached_structures:
                results[i] = self.cached_structures[seq]
            else:
                uncached_sequences.append(seq)
                uncached_indices.append(i)
        
        if not uncached_sequences:
            return results if len(results) > 1 else results[0]
            
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        tokenized_input = tokenizer(
            uncached_sequences, 
            return_tensors="pt", 
            add_special_tokens=False
        )['input_ids']
        
        if self.device == "cuda":
            tokenized_input = tokenized_input.cuda()

        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        pdb_files = self.convert_outputs_to_pdb(output)
            
        # Cache the result
        for i, seq in enumerate(uncached_sequences):
            self.cached_structures[seq] = pdb_files[i]
        print(f"Structure prediction took {time.time() - start_time:.2f} seconds")
        return pdb_files
        
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

    def _process_single_pose(self, pdb_file):
        """Helper function for parallel Rosetta processing"""
        # Reinitialize PyRosetta for each worker process
        if not pyrosetta.is_initialized():
            pyrosetta.init()
            
        pose = pyrosetta.pose_from_pdb(pdb_file)
        scorefxn = pyrosetta.get_fa_scorefxn()
        task = pyrosetta.standard_packer_task(pose)
        task.restrict_to_repacking()
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
        packer.apply(pose)
        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        min_mover.score_function(scorefxn)
        min_mover.apply(pose)
        return scorefxn(pose)

    def calculate_stability(self, sequences):
        """Calculate stability scores using Rosetta with parallel processing"""
        start_time = time.time()
        
        # Ensure sequences is a list
        if not isinstance(sequences, list):
            sequences = [sequences]
            
        # Get structure predictions
        pdb_files = self.predict_structure(sequences)
        if not isinstance(pdb_files, list):
            pdb_files = [pdb_files]
            
        # Process structures in parallel using multiprocessing
        with Pool() as pool:
            stability_scores = pool.map(self._process_single_pose, pdb_files)
            
        print(f"Batch stability calculation took {time.time() - start_time:.2f} seconds")
        return stability_scores if len(stability_scores) > 1 else stability_scores[0]

## for testing, deprecated
def get_stability_reward(sequence):
    calculator = StabilityRewardCalculator()
    return calculator.calculate_stability(sequence)
