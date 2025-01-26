import Bio.PDB
import numpy as np
import torch
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from diy_fpocket import DIYFpocket
from vina import Vina
import tempfile
from meeko import MoleculePreparation
import requests
import time
import json
import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem



class BindingEnergyCalculator:
    def __init__(self, protein_model_path="facebook/esmfold_v1", device="cuda"):
        self.protein_model = self._load_protein_model(protein_model_path, device)
        self.device = device
        self.cached_structures = {}

        
    def _load_protein_model(self, model_path, device):
        """Load ESMFold model for structure prediction"""
        # Check if model is already cached locally
        local_path = '/root/prO-1/model_cache/'
        if os.path.exists(local_path):
            model = EsmForProteinFolding.from_pretrained(local_path)
        else:
            # Download and cache the model
            model = EsmForProteinFolding.from_pretrained(model_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            model.save_pretrained(local_path)
            
        if device == "cuda":
            model = model.cuda()
        return model

    def predict_structure(self, sequence, model_path="facebook/esmfold_v1"):
        """Predict protein structure using ESMFold"""
        if sequence in self.cached_structures:
            return self.cached_structures[sequence]
            
        # Run ESMFold prediction
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenized_input = tokenizer(
            [sequence], 
            return_tensors="pt", 
            add_special_tokens=False, 
        )['input_ids']
        if self.device == "cuda":
            tokenized_input = tokenized_input.cuda()

        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        predicted_structure = self.convert_outputs_to_pdb(output)

        
        # Cache the result
        self.cached_structures[sequence] = predicted_structure
        return predicted_structure
    
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
        # Create directory for PDB files if it doesn't exist
        output_dir = "predicted_structures"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write each PDB to a separate file
        pdb_files = []
        for i, pdb in enumerate(pdbs):
            pdb_path = os.path.join(output_dir, f"structure_{i}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)
            pdb_files.append(pdb_path)
        return pdb_files
    def identify_active_site(self, pdb_string):
        """Identify active site residues using DIYFpocket
        
        Args:
            pdb_string: PDB file path
            
        Returns:
            list: List of (chain_id, residue_number) tuples representing the active site
        """
        try:
            # Use DIYFpocket to detect pockets
            fpocket = DIYFpocket()
            pockets = fpocket.detect_pockets(pdb_string)
            
            # Process results to get the best pocket
            best_pocket = self._process_pockets(pockets)
            
            return best_pocket
        except Exception as e:
            print(f"Error identifying active site: {e}")
            return None

    def _process_pockets(self, pockets):
        """Process DIYFpocket results to identify the best pocket based on score
        
        Args:
            pockets: List of pocket data from DIYFpocket
            
        Returns:
            list: List of (chain_id, residue_number) tuples for the best pocket
        """
        scored_pockets = []
        
        for pocket in pockets:
            # Get residues for this pocket
            pocket_residues = []
            for chain, _, resnum in pocket['residues']:
                pocket_residues.append((chain, resnum))
            
            # Use fpocket score directly
            total_score = pocket['score']
            
            scored_pockets.append((pocket_residues, total_score))
        
        if not scored_pockets:
            raise ValueError("No pockets found in the structure")
            
        # Return residues from the highest scoring pocket
        return max(scored_pockets, key=lambda x: x[1])[0]

    def calculate_binding_energy(self, pdb_file, ligands, active_site_residues, k=2):
        """Calculate binding energy between protein and ligand using AutoDock Vina
        
        Args:
            pdb_file: PDB file path
            ligand: SMILES string
            active_site_residues: List of (chain_id, residue_number) tuples
        
        Returns:
            float: Binding energy score from Vina (kcal/mol)
        """
        # Create temporary files for Vina input
        ligand_files = []
        try:
            # Create receptor file
            receptor_file = tempfile.NamedTemporaryFile(suffix='.pdbqt')
            
            # Convert PDB to PDBQT for the receptor
            convert_pdb_to_pdbqt(pdb_file, receptor_file.name)
            
            # Create temporary files for each ligand
            for ligand in ligands:
                ligand_file = tempfile.NamedTemporaryFile(suffix='.pdbqt')
                convert_smiles_to_pdbqt(ligand, ligand_file.name)
                ligand_files.append(ligand_file)
            
            # Get active site coordinates
            structure = Bio.PDB.PDBParser().get_structure("protein", pdb_file)
            coords = []
            for chain_id, res_num in active_site_residues:
                residue = structure[0][chain_id][res_num]
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
            coords = np.array(coords)
            
            # Calculate box center and size
            center = coords.mean(axis=0)
            size = coords.max(axis=0) - coords.min(axis=0) + 1.0  # Add 8Å padding
            
            # Initialize Vina
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_file.name)
            v.compute_vina_maps(center=center.tolist(), box_size=size.tolist())
            
            # Set all ligands at once
            ligand_file_names = [f.name for f in ligand_files]
            v.set_ligand_from_file(ligand_file_names)
            
            # Generate and score poses
            v.dock(exhaustiveness=8, n_poses=10)
            energy_scores = v.energies()
            
            # Calculate average binding energy for each ligand separately
            num_ligands = len(ligand_files)
            poses_per_ligand = len(energy_scores) // num_ligands
            
            total_energy = 0
            for i in range(num_ligands):
                # Get scores for this ligand
                ligand_scores = energy_scores[i * poses_per_ligand : (i + 1) * poses_per_ligand]
                # Average of top k poses for this ligand
                ligand_avg = np.mean([pose[0] for pose in ligand_scores[:k]])
                total_energy += ligand_avg
                
            return total_energy
            
        finally:
            # Clean up temporary files
            receptor_file.close()
            for f in ligand_files:
                f.close()

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

def convert_pdb_to_pdbqt(pdb_file, pdbqt_file):
    # Use Open Babel to convert PDB to PDBQT
    metals = ['ZN', 'MG', 'FE']
    try:
        subprocess.run(['obabel', pdb_file, '-O', pdbqt_file, '-xr'], check=True)
        # Print the PDBQT file contents and check for zinc
        with open(pdbqt_file, 'r') as f:
            contents = f.read()
            # Process contents line by line
            new_lines = []
            for line in contents.split('\n'):
                # Check if line contains any of the metals
                if any(metal in line for metal in metals):
                    # Replace +0.000 charge with +0.950 for metal atoms
                    line = line.replace("+0.000", "+0.950")
                    print(f"Found metal atom, modified charge: {line}")
                new_lines.append(line)
            
            # Write modified contents back to file
            with open(pdbqt_file, 'w') as f:
                f.write('\n'.join(new_lines))
    except subprocess.CalledProcessError as e:
        print(f"Error converting PDB to PDBQT: {e}")
        raise

def convert_smiles_to_pdbqt(smiles, pdbqt_file):
    # Generate 3D structure from SMILES
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    # Write to a temporary PDB file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pdb_file:
        Chem.MolToPDBFile(mol, pdb_file.name)
        pdb_file_path = pdb_file.name

    # Convert PDB to PDBQT using Open Babel
    try:
        subprocess.run(['obabel', '-ipdb', pdb_file_path, '-opdbqt', '-O', pdbqt_file, '-h'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting PDB to PDBQT: {e}")
        raise
        
def calculate_reward(sequence, reagents, products, ts=None, id_active_site=None, alpha=2.0, calculator=None):
    """
    Calculate overall reward based on binding energies across reaction pathway
    
    Args:
        sequence: Protein sequence string
        reagents: List of SMILES strings of reagents
        products: List of SMILES strings of products
        ts: Optional SMILES string of transition state
        id_active_site: Optional list of tuples (position, amino_acid) 
                       e.g. [('45', 'ALA'), ('46', 'GLY')]
        alpha: Weight factor for transition state contribution
        calculator: Optional pre-initialized BindingEnergyCalculator
    """
    # Initialize calculator
    if calculator is None:
        calculator = BindingEnergyCalculator(device="cuda")
    
    # Predict protein structure
    protein_structure = calculator.predict_structure(sequence)[0]
    
    # Identify active site
    if id_active_site is None:
        active_site_residues = calculator.identify_active_site(protein_structure)
    else:
        # Convert position strings to integers and assume chain 'A'
        active_site_residues = [('A', int(pos)) for pos, _ in id_active_site]
    
    # Calculate binding energies for each state
    reagent_energy = calculator.calculate_binding_energy(
        protein_structure, 
        reagents,
        active_site_residues
    )
    
    if products is not None:
        product_energy = calculator.calculate_binding_energy(
            protein_structure,
            products,
            active_site_residues
        )
    else:
        product_energy = 0
    
    # Calculate reward based on energy profile
    if ts is not None:
        # Include transition state in scoring if provided
        ts_energy = calculator.calculate_binding_energy(
            protein_structure,
            ts,
            active_site_residues
        )
        
        reward = (
            -ts_energy * 2.0 +  # Heavily weight TS stabilization
            -abs(reagent_energy + 5.0) +  # Penalize too strong/weak reagent binding
            -abs(product_energy + 2.0)    # Prefer slightly weaker product binding
        )
    else:
        # Score based on just reagent and product binding if no TS provided
        # Calculate reward
        # simply delta between reagent and product binding
        reward = - (reagent_energy - product_energy)
    
    return reward
