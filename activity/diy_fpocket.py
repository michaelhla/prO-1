import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from sklearn.cluster import DBSCAN
from Bio import PDB
from collections import defaultdict
import random

class DIYFpocket:
    def __init__(self, min_sphere_radius=3.0, max_sphere_radius=6.0, 
                 clustering_cutoff=4.5, min_pocket_size=30):
        """
        Initialize DIYFpocket with parameters matching original Fpocket.
        """
        self.min_sphere_radius = min_sphere_radius
        self.max_sphere_radius = max_sphere_radius
        self.clustering_cutoff = clustering_cutoff
        self.min_pocket_size = min_pocket_size
        
        # Hydrophobicity scale (Kyte-Doolittle)
        self.hydrophobicity_scale = {
            'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8,
            'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4,
            'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3,
            'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5,
            'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
        }
        
        # Atom types for contact analysis
        self.polar_atoms = {'N', 'O', 'OG', 'OG1', 'OG2', 'OE1', 'OE2', 
                           'OH', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'ND1', 
                           'ND2', 'NZ', 'OD1', 'OD2', 'SG'}
        
    def read_pdb(self, pdb_file):
        """Read and parse PDB file using Biopython."""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        return structure
        
    def get_atom_data(self, structure):
        """Extract atom coordinates and properties from structure."""
        coords = []
        atoms = []
        self.atom_residue_map = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append(atom.get_coord())
                        atoms.append({
                            'coord': atom.get_coord(),
                            'name': atom.get_name(),
                            'element': atom.element,
                            'residue': residue.get_resname(),
                            'chain': chain.id,
                            'resid': residue.get_id()[1],
                            'is_polar': atom.get_name() in self.polar_atoms
                        })
                        self.atom_residue_map[tuple(atom.get_coord())] = atoms[-1]
        
        return np.array(coords), atoms

    def generate_alpha_spheres(self, coords, atoms):
        """Generate alpha spheres using Voronoi tessellation."""
        voronoi = Voronoi(coords)
        alpha_spheres = []
        
        for vertex in voronoi.vertices:
            # Find closest atoms to vertex
            distances = np.linalg.norm(coords - vertex, axis=1)
            closest_indices = np.argsort(distances)[:4]  # Get 4 closest atoms
            
            if self.min_sphere_radius <= distances[closest_indices[0]] <= self.max_sphere_radius:
                # Check if sphere is valid (not buried)
                if self.is_valid_alpha_sphere(vertex, coords, closest_indices):
                    contacts = [atoms[i] for i in closest_indices]
                    alpha_spheres.append({
                        'center': vertex,
                        'radius': distances[closest_indices[0]],
                        'contacts': contacts,
                        'is_apolar': self.is_apolar_sphere(contacts)
                    })
                    
        return alpha_spheres

    def is_valid_alpha_sphere(self, vertex, coords, contact_indices):
        """Check if alpha sphere is valid (not buried)."""
        contact_coords = coords[contact_indices]
        center = np.mean(contact_coords, axis=0)
        
        # Check if sphere is accessible from protein surface
        radius = np.linalg.norm(vertex - center)
        distances = np.linalg.norm(coords - vertex, axis=1)
        
        return np.sum(distances < radius * 1.1) <= 4  # Allow some tolerance

    def is_apolar_sphere(self, contacts):
        """Determine if alpha sphere is primarily apolar based on contacts."""
        polar_contacts = sum(1 for atom in contacts if atom['is_polar'])
        return polar_contacts <= 1  # Sphere is apolar if it has 0 or 1 polar contacts

    def cluster_alpha_spheres(self, alpha_spheres):
        """Cluster alpha spheres into potential pockets using DBSCAN."""
        coords = np.array([sphere['center'] for sphere in alpha_spheres])
        clustering = DBSCAN(eps=self.clustering_cutoff, min_samples=3)
        labels = clustering.fit_predict(coords)
        
        pockets = defaultdict(list)
        for sphere, label in zip(alpha_spheres, labels):
            if label >= 0:  # Ignore noise points labeled as -1
                pockets[label].append(sphere)
                
        return {k: v for k, v in pockets.items() 
                if len(v) >= self.min_pocket_size}

    def calculate_pocket_volume(self, alpha_spheres):
        """Calculate pocket volume using Monte Carlo integration."""
        centers = np.array([sphere['center'] for sphere in alpha_spheres])
        try:
            hull = ConvexHull(centers)
            bbox = np.ptp(centers, axis=0)
            volume = hull.volume
            
            # Monte Carlo refinement
            points = 1000
            bbox_volume = np.prod(bbox)
            inside_points = 0
            
            for _ in range(points):
                point = np.random.uniform(centers.min(axis=0), centers.max(axis=0))
                if self.point_in_pocket(point, alpha_spheres):
                    inside_points += 1
                    
            return volume * (inside_points / points)
        except:
            # Fallback if ConvexHull fails
            return len(alpha_spheres) * (4/3 * np.pi * np.mean([s['radius']**3 for s in alpha_spheres]))

    def point_in_pocket(self, point, alpha_spheres):
        """Check if point is inside any alpha sphere."""
        for sphere in alpha_spheres:
            if np.linalg.norm(point - sphere['center']) <= sphere['radius']:
                return True
        return False

    def calculate_pocket_score(self, alpha_spheres):
        """Calculate comprehensive pocket score following Fpocket methodology."""
        # Get basic measurements
        n_spheres = len(alpha_spheres)
        n_apolar = sum(1 for sphere in alpha_spheres if sphere['is_apolar'])
        volume = self.calculate_pocket_volume(alpha_spheres)
        
        # Calculate density score
        centers = np.array([sphere['center'] for sphere in alpha_spheres])
        density = n_spheres / (np.max(centers, axis=0) - np.min(centers, axis=0)).mean()
        density_score = min(density / 10, 1.0)
        
        # Calculate apolar proportion score
        apolar_score = n_apolar / n_spheres if n_spheres > 0 else 0
        
        # Calculate hydrophobicity score
        hydrophobicity_values = []
        for sphere in alpha_spheres:
            for contact in sphere['contacts']:
                if contact['residue'] in self.hydrophobicity_scale:
                    hydrophobicity_values.append(self.hydrophobicity_scale[contact['residue']])
        mean_hydrophobicity = np.mean(hydrophobicity_values) if hydrophobicity_values else 0
        hydrophobicity_score = (mean_hydrophobicity + 4.5) / 9.0  # Normalize to [0,1]
        
        # Basic volume filter following Fpocket
        volume_score = 1.0 if volume >= 100 else 0.0
        
        # Calculate polarity score
        polar_contacts = sum(contact['is_polar'] 
                           for sphere in alpha_spheres 
                           for contact in sphere['contacts'])
        total_contacts = sum(len(sphere['contacts']) for sphere in alpha_spheres)
        polarity_score = 1 - (polar_contacts / total_contacts if total_contacts > 0 else 0)
        
        # Filter out small pockets
        if volume < 100:
            return {
                'total_score': 0,
                'density_score': density_score,
                'volume': volume,
                'apolar_score': apolar_score,
                'hydrophobicity_score': hydrophobicity_score,
                'polarity_score': polarity_score
            }

        # Actual Fpocket scoring emphasis
        total_score = (
            0.45 * density_score +      # Local hydrophobic density (most important)
            0.25 * apolar_score +       # Apolar alpha sphere proportion
            0.20 * hydrophobicity_score + # Mean local hydrophobic density
            0.10 * polarity_score       # Polarity distribution
        )
        
        return {
            'total_score': total_score,
            'density_score': density_score,
            'volume_score': volume_score,
            'apolar_score': apolar_score,
            'hydrophobicity_score': hydrophobicity_score,
            'polarity_score': polarity_score,
            'volume': volume
        }

    def detect_pockets(self, pdb_file):
        """
        Main method to detect and score pockets in a protein structure.
        """
        # Read structure
        structure = self.read_pdb(pdb_file)
        coords, atoms = self.get_atom_data(structure)
        
        # Generate alpha spheres
        alpha_spheres = self.generate_alpha_spheres(coords, atoms)
        
        # Cluster into pockets
        pocket_clusters = self.cluster_alpha_spheres(alpha_spheres)
        
        # Score and prepare output
        pockets = []
        for pocket_id, pocket_spheres in pocket_clusters.items():
            scores = self.calculate_pocket_score(pocket_spheres)
            
            # Get unique residues lining the pocket
            residues = set()
            for sphere in pocket_spheres:
                for contact in sphere['contacts']:
                    residues.add((contact['chain'], 
                                contact['residue'], 
                                contact['resid']))
            
            pocket_info = {
                'pocket_id': pocket_id,
                'coordinates': np.array([sphere['center'] for sphere in pocket_spheres]),
                'alpha_spheres': pocket_spheres,
                'residues': list(residues),
                'volume': scores['volume'],
                'score': scores['total_score'],
                'subscores': {k: v for k, v in scores.items() if k != 'volume'}
            }
            pockets.append(pocket_info)
            
        # Sort pockets by score
        pockets.sort(key=lambda x: x['score'], reverse=True)
        return pockets

def main():
    """Example usage of DIYFpocket."""
    fpocket = DIYFpocket()
    pdb_file = "protein.pdb"  # Replace with your PDB file
    pockets = fpocket.detect_pockets(pdb_file)
    
    # Print results
    for i, pocket in enumerate(pockets):
        print(f"\nPocket {i+1} (Original ID: {pocket['pocket_id']}):")
        print(f"Score: {pocket['score']:.3f}")
        print(f"Volume: {pocket['volume']:.1f} Å³")
        print("Subscores:")
        for name, score in pocket['subscores'].items():
            if name != 'total_score':
                print(f"  {name}: {score:.3f}")
        print(f"Number of alpha spheres: {len(pocket['alpha_spheres'])}")
        print(f"Number of residues: {len(pocket['residues'])}")
        print("Top residues:")
        for chain, resname, resid in sorted(pocket['residues'])[:5]:
            print(f"  Chain {chain}, {resname} {resid}")

if __name__ == "__main__":
    main()