import unittest
import torch
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from reward import BindingEnergyCalculator

class TestESMFoldModel(unittest.TestCase):
    def setUp(self):
        self.calculator = BindingEnergyCalculator(device="cuda")  # Use CPU for testing
        self.test_sequence = "MLKQVEEALGKKLEEL"  # Short test sequence
        
    def test_load_protein_model(self):
        """Test if protein model loads correctly"""
        model = self.calculator._load_protein_model("facebook/esmfold_v1", "cuda")
        self.assertIsInstance(model, EsmForProteinFolding)
        self.assertEqual(model.device.type, "cuda")

    def test_predict_structure(self):
        """Test structure prediction"""
        # Test prediction
        predicted_structure = self.calculator.predict_structure(self.test_sequence)
        
        # Check if output is a list of PDB strings
        self.assertIsInstance(predicted_structure, list)
        self.assertTrue(len(predicted_structure) > 0)
        self.assertIsInstance(predicted_structure[0], str)
        
        # Test caching
        cached_structure = self.calculator.predict_structure(self.test_sequence)
        self.assertEqual(predicted_structure, cached_structure)
        
    def test_convert_outputs_to_pdb(self):
        """Test conversion of model outputs to PDB format"""
        # Create mock outputs
        batch_size = 1
        seq_length = len(self.test_sequence)
        mock_outputs = {
            "positions": torch.rand(1, batch_size, seq_length, 14, 3),
            "aatype": torch.zeros(batch_size, seq_length, dtype=torch.long),
            "atom37_atom_exists": torch.ones(batch_size, seq_length, 37),
            "residue_index": torch.arange(seq_length).unsqueeze(0),
            "plddt": torch.ones(batch_size, seq_length) * 100
        }
        
        pdbs = self.calculator.convert_outputs_to_pdb(mock_outputs)
        
        # Check output format
        self.assertIsInstance(pdbs, list)
        self.assertEqual(len(pdbs), batch_size)
        self.assertIsInstance(pdbs[0], str)
        
        # Basic PDB format checks
        pdb_lines = pdbs[0].split('\n')
        self.assertTrue(any(line.startswith('ATOM') for line in pdb_lines))

if __name__ == '__main__':
    unittest.main()
