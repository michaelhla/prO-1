import os
import json
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import mdtraj as md

class StructureFlexibilityAnalyzer:
    def __init__(self, output_dir="base_structure_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different outputs
        self.flexibility_dir = os.path.join(output_dir, "flexibility_plots")
        self.structure_dir = os.path.join(output_dir, "structure_plots")
        
        os.makedirs(self.flexibility_dir, exist_ok=True)
        os.makedirs(self.structure_dir, exist_ok=True)
    
    def analyze_structure_flexibility(self, pdb_file, identifier):
        """Analyze structure flexibility using contact map analysis"""
        print(f"Analyzing flexibility for {identifier}...")
        
        # Load structure
        structure = md.load(pdb_file)
        
        # Extract CA atoms
        ca_indices = structure.topology.select('name CA')
        ca_structure = structure.atom_slice(ca_indices)
        
        # Calculate distance matrix between all CA atoms
        distances = md.compute_distances(ca_structure, 
                                         np.array([[i, j] for i in range(len(ca_indices)) 
                                                  for j in range(i+1, len(ca_indices))]))
        
        # Reshape distances to a matrix
        n_atoms = len(ca_indices)
        dist_matrix = np.zeros((n_atoms, n_atoms))
        
        k = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist_matrix[i, j] = dist_matrix[j, i] = distances[0, k]
                k += 1
        
        # Calculate contact map (residues within 8 Angstroms are considered in contact)
        contact_map = dist_matrix < 0.8  # 0.8 nm = 8 Angstroms
        
        # Calculate flexibility score for each residue based on number of contacts
        # Fewer contacts = more flexible
        flexibility = np.sum(contact_map, axis=0)
        flexibility_score = 1.0 / (1.0 + flexibility)  # Normalize
        
        # Plot flexibility
        plt.figure(figsize=(10, 6))
        plt.plot(flexibility_score)
        plt.xlabel('Residue')
        plt.ylabel('Flexibility Score')
        plt.title(f'Residue Flexibility - {identifier}')
        plt.grid(True, linestyle='--', alpha=0.7)
        flex_plot = os.path.join(self.flexibility_dir, f"{identifier}_flexibility.png")
        plt.savefig(flex_plot)
        plt.close()
        
        # Create structure visualizations
        self.create_structure_plots(ca_structure, flexibility_score, identifier)
        
        return {
            'flexibility': flexibility_score.tolist(),
            'avg_flexibility': float(np.mean(flexibility_score)),
            'max_flexibility': float(np.max(flexibility_score)),
            'flexible_regions': self.identify_flexible_regions(flexibility_score)
        }
    
    def identify_flexible_regions(self, flexibility_score, threshold=0.5):
        """Identify regions with high flexibility"""
        flexible_regions = []
        current_region = []
        
        for i, score in enumerate(flexibility_score):
            if score > threshold:
                current_region.append(i)
            elif current_region:
                flexible_regions.append((current_region[0], current_region[-1]))
                current_region = []
        
        if current_region:
            flexible_regions.append((current_region[0], current_region[-1]))
        
        return flexible_regions
    
    def create_structure_plots(self, structure, flexibility_score, identifier):
        """Create static 3D plots of the structure from different angles"""
        print(f"Creating structure plots for {identifier}...")
        
        # Get coordinates
        xyz = structure.xyz[0]
        
        # Normalize flexibility for coloring
        norm_flex = (flexibility_score - np.min(flexibility_score)) / (np.max(flexibility_score) - np.min(flexibility_score))
        
        # Create images from different angles
        angles = [0, 90, 180, 270]
        
        for angle in angles:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points colored by flexibility
            scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                                c=norm_flex, cmap='coolwarm', s=50)
            
            # Connect CA atoms with lines
            for i in range(len(xyz)-1):
                ax.plot([xyz[i, 0], xyz[i+1, 0]], 
                        [xyz[i, 1], xyz[i+1, 1]], 
                        [xyz[i, 2], xyz[i+1, 2]], 'k-', alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Flexibility Score')
            
            # Set axis limits
            max_range = np.max(xyz.max(axis=0) - xyz.min(axis=0))
            mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) / 2
            mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) / 2
            mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) / 2
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            ax.view_init(30, angle)
            ax.set_title(f'Structure Flexibility - {identifier} (Angle: {angle}°)')
            
            # Save image
            img_file = os.path.join(self.structure_dir, f"{identifier}_structure_angle{angle}.png")
            plt.savefig(img_file, dpi=150)
            plt.close()
            
            print(f"Image saved to {img_file}")

def main():
    # Load data with RMSD values
    with open("creative_top_performers_with_rmsd.json", "r") as f:
        data = json.load(f)
    
    # Sort by RMSD
    sorted_data = sorted(data, key=lambda x: x.get('rmsd', float('inf')))
    
    # Initialize analyzer
    analyzer = StructureFlexibilityAnalyzer()
    
    # Analyze all structures
    results = []
    
    for i, item in enumerate(sorted_data):
        if 'aligned_pdb' in item:
            pdb_file = item['aligned_pdb']
            identifier = item.get('id', str(i))
            rmsd = item.get('rmsd', 0)
            
            print(f"\nAnalyzing structure {identifier} (RMSD: {rmsd:.4f})")
            
            try:
                # Analyze structure flexibility
                analysis = analyzer.analyze_structure_flexibility(pdb_file, identifier)
                
                # Store results
                results.append({
                    'identifier': identifier,
                    'rmsd_to_original': rmsd,
                    'flexibility_analysis': analysis
                })
                
                print(f"Completed analysis for structure {identifier}")
                
            except Exception as e:
                print(f"Error processing structure {identifier}: {str(e)}")
    
    # Save results
    with open(os.path.join(analyzer.output_dir, "flexibility_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nStructure Flexibility Summary:")
    print("ID\tRMSD\tAvg Flexibility")
    print("-" * 40)
    
    # Sort results by flexibility
    sorted_results = sorted(results, key=lambda x: x['flexibility_analysis']['avg_flexibility'])
    
    for result in sorted_results:
        print(f"{result['identifier']}\t{result['rmsd_to_original']:.4f}\t{result['flexibility_analysis']['avg_flexibility']:.4f}")
    
    print("\nMost stable structures (lowest flexibility):")
    for result in sorted_results[:3]:
        print(f"Structure {result['identifier']}:")
        print(f"  RMSD to original: {result['rmsd_to_original']:.4f}")
        print(f"  Average flexibility: {result['flexibility_analysis']['avg_flexibility']:.4f}")
        print(f"  Flexible regions: {result['flexibility_analysis']['flexible_regions']}")
        print()

if __name__ == "__main__":
    main() 