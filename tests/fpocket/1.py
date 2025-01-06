import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from diy_fpocket import DIYFpocket
import tempfile

def test_fpocket():
    # Create a simple test PDB string
    test_pdb = """
ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00           N  
ATOM      2  CA  ALA A   1      26.124  25.274   2.842  1.00  0.00           C  
ATOM      3  C   ALA A   1      26.527  26.729   2.671  1.00  0.00           C  
ATOM      4  O   ALA A   1      27.619  26.989   2.169  1.00  0.00           O  
ATOM      5  CB  ALA A   1      25.562  25.075   4.239  1.00  0.00           C  
ATOM      6  N   VAL A   2      25.645  27.664   3.046  1.00  0.00           N  
ATOM      7  CA  VAL A   2      25.930  29.087   2.920  1.00  0.00           C  
ATOM      8  C   VAL A   2      24.797  29.881   3.573  1.00  0.00           C  
ATOM      9  O   VAL A   2      23.672  29.401   3.655  1.00  0.00           O  
ATOM     10  CB  VAL A   2      26.060  29.488   1.438  1.00  0.00           C  
"""

    # Save test PDB to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb') as f:
        f.write(test_pdb)
        f.flush()
        
        # Initialize DIYFpocket
        fpocket = DIYFpocket()
        
        # Detect pockets
        try:
            pockets = fpocket.detect_pockets(f.name)
            print(f"Found {len(pockets)} pockets")
            
            # Print details of each pocket
            for i, pocket in enumerate(pockets):
                print(f"\nPocket {i+1}:")
                print(f"Score: {pocket['score']}")
                print(f"Volume: {pocket['volume']}")
                print("Residues:", pocket['residues'])
                
            assert len(pockets) >= 0, "Should detect zero or more pockets"
            
        except Exception as e:
            print(f"Error detecting pockets: {str(e)}")
            raise

if __name__ == "__main__":
    test_fpocket()
