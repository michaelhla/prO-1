import Bio.PDB
from alphafold.model import model
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openbabel import pybel
import pymol
from scipy.spatial.distance import cdist
import torch
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import prody
from fpocket import fpocket
import os
from vina import Vina
import tempfile
from meeko import MoleculePreparation


class BindingEnergyCalculator:
    def __init__(self, protein_model_path="facebook/esmfold-v1", device="cuda"):
        self.protein_model = self._load_protein_model(protein_model_path)
        self.device = device
        self.cached_structures = {}
        
    def _load_protein_model(self, model_path):
        """Load ESMFold model for structure prediction"""
        model = EsmForProteinFolding.from_pretrained(model_path)
        model = model.to(self.device)
        return model

    def predict_structure(self, sequence, model_path="facebook/esmfold-v1"):
        """Predict protein structure using ESMFold"""
        if sequence in self.cached_structures:
            return self.cached_structures[sequence]
            
        # Run ESMFold prediction
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        predicted_structure = self.convert_outputs_to_pdb(output)

        
        # Cache the result
        self.cached_structures[sequence] = predicted_structure
        return predicted_structure
    
    def convert_outputs_to_pdb(outputs):
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
        return pdbs

    def identify_active_site(self, pdb):
        """Identify active site residues using multiple pocket detection methods
        
        Uses a combination of:
        - fpocket: Fast and accurate pocket detection
        - DSSP: Secondary structure identification
        - ProDy: Protein dynamics and conservation analysis
        """
        # Run fpocket for pocket detection
        pockets = fpocket(pdb)
        
        # Use DSSP to identify secondary structure elements
        dssp_dict = dssp_dict_from_pdb_file(pdb)[0]
        
        # Use ProDy for conservation analysis
        protein = prody.parsePDB(pdb)
        conservation = prody.conservationMatrix(protein)
        
        # Score pockets based on multiple criteria
        scored_pockets = []
        for pocket in pockets:
            score = self._evaluate_pocket(
                pocket,
                conservation,
                dssp_dict,
                protein
            )
            scored_pockets.append((pocket, score))
        
        # Return residues from top-scoring pocket
        best_pocket = max(scored_pockets, key=lambda x: x[1])[0]
        return self._get_pocket_residues(best_pocket)

    def calculate_binding_energy(self, protein_structure, ligand, active_site_residues):
        """Calculate binding energy between protein and ligand using AutoDock Vina
        
        Args:
            protein_structure: PDB structure string
            ligand: RDKit Mol object
            active_site_residues: List of (chain_id, residue_number) tuples
        
        Returns:
            float: Binding energy score from Vina (kcal/mol)
        """
        # Create temporary files for Vina input
        with tempfile.NamedTemporaryFile(suffix='.pdb') as protein_file, \
             tempfile.NamedTemporaryFile(suffix='.pdbqt') as ligand_file:
            
            # Save protein structure
            protein_file.write(protein_structure.encode())
            protein_file.flush()
            
            # Convert ligand to PDBQT format
            self._mol_to_pdbqt(ligand, ligand_file.name)
            
            # Initialize Vina
            v = Vina(sf_name='vina')
            v.set_receptor(protein_file.name)
            v.set_ligand_from_file(ligand_file.name)
            
            # Calculate search box centered on active site
            center, size = self._get_binding_box(active_site_residues)
            v.compute_vina_maps(center=center, box_size=size)
            
            # Generate and score poses
            energy_scores = v.dock(exhaustiveness=8, n_poses=10)
            
            # Return best score (most negative energy)
            return min(energy_scores)

    def _get_binding_box(self, active_site_residues):
        """Calculate binding box dimensions from active site residues
        
        Args:
            active_site_residues: List of (chain_id, residue_number) tuples
            
        Returns:
            tuple: (center_coords, box_size) where each is a list of [x, y, z]
        """
        # Get coordinates of active site residues
        coords = []
        structure = Bio.PDB.PDBParser().get_structure("protein", self.tmp_pdb)
        
        for chain_id, res_num in active_site_residues:
            residue = structure[0][chain_id][res_num]
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())
        
        coords = np.array(coords)
        
        # Calculate box center and size
        center = coords.mean(axis=0)
        size = coords.max(axis=0) - coords.min(axis=0) + 8.0  # Add 8Å padding
        
        return center.tolist(), size.tolist()

    def _mol_to_pdbqt(self, mol, output_file):
        """Convert RDKit molecule to PDBQT format
        
        Args:
            mol: RDKit Mol object
            output_file: Path to save PDBQT file
        """
        # Prepare molecule (add hydrogens, charges, etc)
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        
        # Write PDBQT file
        preparator.write_pdbqt_file(output_file)

    def _evaluate_pocket(self, pocket, conservation, dssp_dict, protein):
        """Score a pocket based on multiple criteria
        
        Criteria:
        1. Pocket volume and depth
        2. Conservation scores of residues
        3. Secondary structure elements
        4. Physicochemical properties
        """
        # Calculate geometric score
        volume_score = pocket.volume / 1000  # Normalize volume
        
        # Conservation score
        pocket_residues = self._get_pocket_residues(pocket)
        conservation_score = np.mean([conservation[res] for res in pocket_residues])
        
        # Secondary structure score (prefer mixed alpha/beta)
        ss_types = [dssp_dict[res]['ss'] for res in pocket_residues]
        ss_diversity = len(set(ss_types)) / len(ss_types)
        
        # Combine scores (weights can be adjusted)
        total_score = (0.4 * volume_score + 
                      0.4 * conservation_score +
                      0.2 * ss_diversity)
        
        return total_score

    def _get_pocket_residues(self, pocket):
        """Extract residue IDs from a pocket object
        
        Args:
            pocket: fpocket Pocket object containing pocket information
            
        Returns:
            list: List of residue IDs in the format (chain_id, residue_number)
        """
        residues = []
        
        # fpocket stores residues in pocket.residues
        # Each residue has attributes: chain, number, name
        for residue in pocket.residues:
            res_id = (residue.chain, residue.number)
            residues.append(res_id)
            
        # Get neighboring residues within 4Å of pocket center
        pocket_center = pocket.center_of_mass
        structure = Bio.PDB.PDBParser().get_structure("protein", self.tmp_pdb)
        
        for chain in structure.get_chains():
            for residue in chain:
                # Calculate distance from residue CA to pocket center
                if "CA" in residue:
                    ca_pos = residue["CA"].get_coord()
                    distance = np.linalg.norm(ca_pos - pocket_center)
                    
                    # Include residues within cutoff
                    if distance < 4.0:
                        res_id = (chain.id, residue.id[1])
                        if res_id not in residues:
                            residues.append(res_id)
        
        return residues

def calculate_reward(sequence, reagent, ts, product, id_active_site=None):
    """
    Calculate overall reward based on binding energies across reaction pathway
    
    Args:
        sequence: Protein sequence string
        reagent: RDKit Mol object of reagent
        ts: RDKit Mol object of transition state
        product: RDKit Mol object of product
    
    Returns:
        float: Reward score based on binding energies and reaction profile
    """
    # Initialize calculator
    calculator = BindingEnergyCalculator(device="cuda")
    
    # Predict protein structure
    protein_structure = calculator.predict_structure(sequence)
    
    # Identify active site
    if id_active_site is None:
        active_site_residues = calculator.identify_active_site(protein_structure)
    else:
        active_site_residues = id_active_site
    
    # Calculate binding energies for each state
    reagent_energy = calculator.calculate_binding_energy(
        protein_structure, 
        reagent,
        active_site_residues
    )
    
    ts_energy = calculator.calculate_binding_energy(
        protein_structure,
        ts,
        active_site_residues
    )
    
    product_energy = calculator.calculate_binding_energy(
        protein_structure,
        product,
        active_site_residues
    )
    
    # Calculate reward based on energy profile
    # We want:
    # 1. Strong TS binding (stabilization of transition state)
    # 2. Moderate reagent binding (not too strong to prevent product release)
    # 3. Weaker product binding (to facilitate product release)
    
    reward = (
        -ts_energy * 2.0 +  # Heavily weight TS stabilization
        -abs(reagent_energy + 5.0) +  # Penalize too strong/weak reagent binding
        -abs(product_energy + 2.0)    # Prefer slightly weaker product binding
    )
    
    return reward