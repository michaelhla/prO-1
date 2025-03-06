import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from Bio.PDB import PDBParser
from mpl_toolkits.mplot3d import Axes3D

def extract_ca_coordinates(pdb_file):
    """Extract CA atom coordinates from a PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    
    return np.array(ca_coords)

def plot_structure_comparison(original_pdb, predicted_pdb, rmsd, identifier, output_dir="base_structure_plots"):
    """Create a 3D plot comparing original and predicted structures"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract coordinates
    original_coords = extract_ca_coordinates(original_pdb)
    predicted_coords = extract_ca_coordinates(predicted_pdb)
    
    # Use the shorter length
    min_length = min(len(original_coords), len(predicted_coords))
    original_coords = original_coords[:min_length]
    predicted_coords = predicted_coords[:min_length]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original structure
    ax.plot(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], 
            'b-', label='Original', linewidth=2)
    
    # Plot predicted structure
    ax.plot(predicted_coords[:, 0], predicted_coords[:, 1], predicted_coords[:, 2], 
            'r-', label=f'Predicted (RMSD: {rmsd:.2f}Å)', linewidth=2)
    
    ax.set_title(f'Structure Comparison - Sequence {identifier}')
    ax.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{identifier}_structure_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def create_summary_pdf(data, original_pdb, output_file="base_structure_comparison_summary.pdf"):
    """Create a PDF with all structure comparisons"""
    with PdfPages(output_file) as pdf:
        # Sort data by RMSD
        sorted_data = sorted(data, key=lambda x: x.get('rmsd', float('inf')))
        
        # Create a summary page
        plt.figure(figsize=(10, 8))
        plt.title("RMSD Comparison of Predicted Structures")
        
        identifiers = [item.get('id', str(i)) for i, item in enumerate(sorted_data)]
        rmsds = [item.get('rmsd', 0) for item in sorted_data]
        
        plt.bar(range(len(identifiers)), rmsds)
        plt.xticks(range(len(identifiers)), identifiers, rotation=90)
        plt.ylabel('RMSD (Å)')
        plt.xlabel('Sequence ID')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Add individual structure comparisons
        for i, item in enumerate(sorted_data):
            if 'predicted_pdb' in item and 'rmsd' in item:
                predicted_pdb = item['predicted_pdb']
                rmsd = item['rmsd']
                identifier = item.get('id', str(i))
                
                # Create and add plot to PDF
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Extract coordinates
                original_coords = extract_ca_coordinates(original_pdb)
                predicted_coords = extract_ca_coordinates(predicted_pdb)
                
                # Use the shorter length
                min_length = min(len(original_coords), len(predicted_coords))
                original_coords = original_coords[:min_length]
                predicted_coords = predicted_coords[:min_length]
                
                # Plot structures
                ax.plot(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], 
                        'b-', label='Original', linewidth=2)
                ax.plot(predicted_coords[:, 0], predicted_coords[:, 1], predicted_coords[:, 2], 
                        'r-', label=f'Predicted (RMSD: {rmsd:.2f}Å)', linewidth=2)
                
                ax.set_title(f'Structure Comparison - Sequence {identifier}')
                ax.legend()
                
                pdf.savefig(fig)
                plt.close()

def generate_pymol_script(data, original_pdb, output_file="base_visualize_structures.pml"):
    """Generate a PyMOL script for visualization"""
    with open(output_file, "w") as f:
        f.write("# PyMOL script for structure visualization\n")
        f.write(f"load {original_pdb}, original\n")
        f.write("color blue, original\n\n")
        
        # Sort data by RMSD
        sorted_data = sorted(data, key=lambda x: x.get('rmsd', float('inf')))
        
        for i, item in enumerate(sorted_data):
            if 'aligned_pdb' in item and 'rmsd' in item:
                aligned_pdb = item['aligned_pdb']
                rmsd = item['rmsd']
                identifier = item.get('id', str(i))
                
                f.write(f"# Sequence {identifier} (RMSD: {rmsd:.2f}Å)\n")
                f.write(f"load {aligned_pdb}, seq_{identifier}\n")
                f.write(f"color red, seq_{identifier}\n")
                f.write(f"align seq_{identifier}, original\n")
                f.write(f"group seq_{identifier}, seq_{identifier}\n\n")
        
        f.write("# Show all structures as cartoon\n")
        f.write("show cartoon\n")
        f.write("hide lines\n")
        f.write("set cartoon_transparency, 0.5, original\n")
        f.write("zoom\n")
        
    print(f"PyMOL script generated: {output_file}")
    print("You can run this script in PyMOL when you have access to a GUI.")

def main():
    # Load data with RMSD values
    with open("top_performers_with_rmsd.json", "r") as f:
        data = json.load(f)
    
    # Get original PDB file
    original_pdb = None
    for filename in os.listdir("ca2_structures"):
        if filename.startswith("original_predicted"):
            original_pdb = os.path.join("ca2_structures", filename)
            break
    
    if not original_pdb:
        print("Original PDB file not found!")
        return
    
    print(f"Using original structure: {original_pdb}")
    
    # Create individual plots
    for i, item in enumerate(data):
        if 'predicted_pdb' in item and 'rmsd' in item:
            predicted_pdb = item['predicted_pdb']
            rmsd = item['rmsd']
            identifier = item.get('id', str(i))
            
            plot_path = plot_structure_comparison(original_pdb, predicted_pdb, rmsd, identifier)
            print(f"Created plot for sequence {identifier}: {plot_path}")
    
    # Create summary PDF
    create_summary_pdf(data, original_pdb)
    print("Created summary PDF: base_structure_comparison_summary.pdf")
    
    # Generate PyMOL script for future visualization
    generate_pymol_script(data, original_pdb)

if __name__ == "__main__":
    main() 