import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import time
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB import PDBIO
import pyrosetta_installer
pyrosetta_installer.install_pyrosetta()
import pyrosetta

class StructureComparator:
    def __init__(self, protein_model_path="facebook/esmfold_v1", device="cuda"):
        self.device = torch.device(device)
        self.protein_model = self._load_protein_model(protein_model_path)
        self.cached_structures = {}
        self.output_dir = "ca2_creative_structures"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize PyRosetta
        pyrosetta.init(silent=True)
        
        # Initialize PDB parser
        self.parser = PDBParser(QUIET=True)

    def _load_protein_model(self, model_path):
        """Load ESMFold model for structure prediction"""
        start_time = time.time()
        local_path = 'model_cache/'
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

    def predict_structure(self, sequence, identifier):
        """Predict protein structure using ESMFold"""
        if sequence in self.cached_structures:
            return self.cached_structures[sequence]

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        tokenized_input = tokenizer(
            [sequence], 
            return_tensors="pt", 
            add_special_tokens=False
        )['input_ids'].to(self.device)

        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        pdb_file = self.convert_outputs_to_pdb(output, identifier)[0]

        # Cache the result
        self.cached_structures[sequence] = pdb_file
        print(f"Structure prediction for {identifier} took {time.time() - start_time:.2f} seconds")
        return pdb_file

    def convert_outputs_to_pdb(self, outputs, identifier):
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

        pdb_files = []
        for i, pdb in enumerate(pdbs):
            pdb_path = os.path.join(self.output_dir, f"{identifier}_predicted.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)
            pdb_files.append(pdb_path)
        return pdb_files

    def extract_ca_atoms(self, structure):
        """Extract CA atoms from a structure"""
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
        return ca_atoms

    def calculate_rmsd(self, ref_structure, sample_structure):
        """Calculate RMSD between two structures using CA atoms"""
        ref_atoms = self.extract_ca_atoms(ref_structure)
        sample_atoms = self.extract_ca_atoms(sample_structure)
        
        # If structures have different lengths, use the shorter one
        min_length = min(len(ref_atoms), len(sample_atoms))
        ref_atoms = ref_atoms[:min_length]
        sample_atoms = sample_atoms[:min_length]
        
        # Superimpose structures
        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, sample_atoms)
        super_imposer.apply(sample_structure.get_atoms())
        
        return super_imposer.rms

    def compare_structures(self, original_pdb, predicted_pdb, identifier):
        """Compare original and predicted structures"""
        # Load structures
        original_structure = self.parser.get_structure("original", original_pdb)
        predicted_structure = self.parser.get_structure("predicted", predicted_pdb)
        
        # Calculate RMSD
        rmsd = self.calculate_rmsd(original_structure, predicted_structure)
        
        # Save the superimposed structure
        io = PDBIO()
        io.set_structure(predicted_structure)
        aligned_pdb_path = os.path.join(self.output_dir, f"{identifier}_aligned.pdb")
        io.save(aligned_pdb_path)
        
        return rmsd, aligned_pdb_path

def main():
    # Load sequences from JSON
    with open("creative_top_performers_with_rmsd.json", "r") as f:
        data = json.load(f)
    
    comparator = StructureComparator()
    
    # Original sequence to fold
    original_sequence = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    
    # Predict structure for original sequence
    print("Folding original sequence...")
    original_pdb = comparator.predict_structure(original_sequence, "original")
    print(f"Original structure saved to {original_pdb}")
    
    # Track the best RMSD
    best_rmsd = float('inf')
    best_sequence_id = None
    
    # Process each sequence
    for i, item in enumerate(data):
        sequence = item.get("sequence")
        identifier = str(i)
        
        if not sequence:
            print(f"Skipping item {identifier}: No sequence found")
            continue
        
        print(f"Processing sequence {identifier}")
        
        # Predict structure
        predicted_pdb = comparator.predict_structure(sequence, identifier)
        
        # Compare with original structure
        rmsd, aligned_pdb = comparator.compare_structures(original_pdb, predicted_pdb, identifier)
        
        # Update the JSON data
        item["rmsd"] = rmsd
        item["predicted_pdb"] = predicted_pdb
        item["aligned_pdb"] = aligned_pdb
        
        print(f"Sequence {identifier}: RMSD = {rmsd:.4f}")
        
        # Track best RMSD
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_sequence_id = identifier
    
    # Save updated JSON
    with open("creative_top_performers_with_rmsd.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nBest RMSD: {best_rmsd:.4f} for sequence {best_sequence_id}")

if __name__ == "__main__":
    main() 