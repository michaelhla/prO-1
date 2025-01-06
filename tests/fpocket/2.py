import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from diy_fpocket import DIYFpocket
import tempfile
import requests

def test_lysozyme_active_site():
    # Download 1LYZ structure using requests
    pdb_id = '1LYZ'
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    
    if response.status_code == 200:
        pdb_path = f"{pdb_id}.pdb"
        with open(pdb_path, 'w') as file:
            file.write(response.text)
        print(f"PDB file {pdb_id} downloaded successfully.")
    else:
        print(f"Failed to download PDB file {pdb_id}. Status code: {response.status_code}")

    try:
        # Initialize DIYFpocket
        fpocket = DIYFpocket()
        
        # Detect pockets
        pockets = fpocket.detect_pockets(pdb_path)
        
        # Check if pockets were found
        assert len(pockets) > 0, "No pockets detected"
        
        # Check if the known active site residues (Asp52 and Glu35) are identified
        active_site_found = False
        for pocket in pockets:
            residues = set((chain, resname, resnum) for chain, resname, resnum in pocket['residues'])
            # Look specifically for ASP52 and GLU35
            if ('A', 'ASP', 52) in residues and ('A', 'GLU', 35) in residues:
                active_site_found = True
                print(f"Active site found in pocket with score {pocket['score']:.2f}")
                break
        
        assert active_site_found, "Failed to identify known active site residues (Asp52 and Glu35)"
        
    except Exception as e:
        print(f"Error in active site detection: {str(e)}")
        raise
        
    finally:
        # Cleanup downloaded PDB file
        if os.path.exists(pdb_path):
            os.remove(pdb_path)

if __name__ == "__main__":
    test_lysozyme_active_site()

