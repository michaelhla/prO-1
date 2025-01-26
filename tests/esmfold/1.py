import unittest
import torch
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from activity_reward import BindingEnergyCalculator

class TestESMFoldModel(unittest.TestCase):
    def setUp(self):
        self.calculator = BindingEnergyCalculator(device="cuda")  # Use CPU for testing
        self.test_sequence = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"  # Short test sequence
        
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
        

if __name__ == '__main__':
    unittest.main()
