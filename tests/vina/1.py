import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tempfile
import numpy as np
from Bio import PDB
from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
import os
import subprocess


def test_vina():
    # Create a simple test PDB string
    # Download 1lyz.pdb from PDB
    
    pdb_id = "1hea"
    pdb_path = f"{pdb_id}.pdb"

    # Only download if file doesn't exist
    if not os.path.exists(pdb_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        with open(pdb_path, "w") as f:
            f.write(response.text)

    # Create a simple ligand from SMILES
    smiles = "O=C=O"  # CO2 molecule

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt') as ligand_file, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt') as receptor_file:
        

        # Convert PDB to PDBQT for the receptor
        convert_pdb_to_pdbqt(pdb_path, receptor_file.name)

        # Get active site coordinates
        structure = PDB.PDBParser().get_structure("protein", pdb_path)
        coords = []
        for residue in structure.get_residues():
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())
        coords = np.array(coords)
        
        # Calculate box center and size
        center = coords.mean(axis=0)
        size = coords.max(axis=0) - coords.min(axis=0) + 8.0  # Add 8Å padding

        # Initialize Vina
        v = Vina(sf_name='vina')
        
        # Set up docking
        v.set_receptor(receptor_file.name)
        v.compute_vina_maps(center=center.tolist(), box_size=size.tolist())
        
        # Convert SMILES to PDBQT
        convert_smiles_to_pdbqt(smiles, ligand_file.name)
        v.set_ligand_from_file(ligand_file.name)
        
        try:
            # Dock the ligand
            v.dock()
            energies = v.energies()
            print(f"Docking energy: {energies[0][0]} kcal/mol")
            
            assert energies[0][0] < 0, "Docking energy should be negative"
            
        except Exception as e:
            print(f"Error during docking: {str(e)}")
            raise

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



if __name__ == "__main__":
    test_vina()
