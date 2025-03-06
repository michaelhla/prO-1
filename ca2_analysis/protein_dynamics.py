import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mdtraj as md
from scipy.linalg import eigh
from Bio.PDB import PDBParser

class ProteinDynamicsSimulator:
    def __init__(self, output_dir="protein_dynamics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.modes_dir = os.path.join(output_dir, "normal_modes")
        self.trajectory_dir = os.path.join(output_dir, "trajectories")
        self.animation_dir = os.path.join(output_dir, "animations")
        
        os.makedirs(self.modes_dir, exist_ok=True)
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.animation_dir, exist_ok=True)
    
    def calculate_hessian(self, coords, cutoff=0.8):
        """Calculate Hessian matrix using elastic network model"""
        n_atoms = coords.shape[0]
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        
        # Calculate distance matrix
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                d = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d
        
        # Build Hessian using elastic network model
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                d = dist_matrix[i, j]
                if d > cutoff:
                    continue
                
                # Unit vector from i to j
                dr = (coords[j] - coords[i]) / d
                
                # Force constant (simplified)
                k = 1.0
                
                # Update Hessian blocks
                for a in range(3):
                    for b in range(3):
                        h_ij = -k * dr[a] * dr[b]
                        hessian[3*i+a, 3*j+b] = h_ij
                        hessian[3*j+b, 3*i+a] = h_ij
                        
                        # Diagonal blocks
                        hessian[3*i+a, 3*i+b] -= h_ij
                        hessian[3*j+a, 3*j+b] -= h_ij
        
        return hessian
    
    def calculate_normal_modes(self, pdb_file, identifier, n_modes=10):
        """Calculate normal modes for a protein structure"""
        print(f"Calculating normal modes for {identifier}...")
        
        # Load structure
        structure = md.load(pdb_file)
        
        # Extract CA atoms
        ca_indices = structure.topology.select('name CA')
        ca_structure = structure.atom_slice(ca_indices)
        
        # Get coordinates
        coords = ca_structure.xyz[0]
        
        # Calculate Hessian matrix
        hessian = self.calculate_hessian(coords)
        
        # Solve eigenvalue problem (skip first 6 modes - rigid body motions)
        eigenvalues, eigenvectors = eigh(hessian)
        
        # Skip first 6 modes (rigid body motions)
        eigenvalues = eigenvalues[6:]
        eigenvectors = eigenvectors[:, 6:]
        
        # Reshape eigenvectors to (n_modes, n_atoms, 3)
        n_atoms = coords.shape[0]
        modes = []
        for i in range(min(n_modes, len(eigenvalues))):
            mode = eigenvectors[:, i].reshape(n_atoms, 3)
            modes.append(mode)
        
        # Plot mode frequencies
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues[:20], 'o-')
        plt.xlabel('Mode Index')
        plt.ylabel('Frequency')
        plt.title(f'Normal Mode Frequencies - {identifier}')
        plt.grid(True, linestyle='--', alpha=0.7)
        freq_plot = os.path.join(self.modes_dir, f"{identifier}_frequencies.png")
        plt.savefig(freq_plot)
        plt.close()
        
        return {
            'coords': coords,
            'eigenvalues': eigenvalues.tolist(),
            'modes': modes,
            'ca_structure': ca_structure
        }
    
    def generate_trajectory(self, coords, modes, identifier, amplitude=1.0, n_frames=50):
        """Generate trajectory along normal modes"""
        print(f"Generating trajectory for {identifier}...")
        
        n_atoms = coords.shape[0]
        trajectories = []
        
        # Generate trajectory for each mode
        for mode_idx, mode in enumerate(modes[:3]):  # Use first 3 modes
            trajectory = []
            
            # Generate oscillatory motion
            for t in range(n_frames):
                # Oscillation factor
                factor = amplitude * np.sin(2 * np.pi * t / n_frames)
                
                # Displace coordinates along mode
                displaced_coords = coords + factor * mode
                trajectory.append(displaced_coords)
            
            trajectories.append(np.array(trajectory))
            
            # Create animation for this mode
            self.create_mode_animation(coords, mode, identifier, mode_idx)
        
        return trajectories
    
    def create_mode_animation(self, coords, mode, identifier, mode_idx, amplitude=1.0, n_frames=50):
        """Create animation of motion along a normal mode"""
        print(f"Creating animation for mode {mode_idx} of {identifier}...")
        
        # Generate frames
        frames = []
        for t in range(n_frames):
            # Oscillation factor
            factor = amplitude * np.sin(2 * np.pi * t / n_frames)
            
            # Displace coordinates along mode
            displaced_coords = coords + factor * mode
            frames.append(displaced_coords)
        
        # Create animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initial plot
        line, = ax.plot(frames[0][:, 0], frames[0][:, 1], frames[0][:, 2], 'b-', linewidth=2)
        
        # Set axis limits with some padding
        all_coords = np.vstack(frames)
        max_range = np.max(all_coords.max(axis=0) - all_coords.min(axis=0)) * 1.2
        mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) / 2
        mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) / 2
        mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        ax.set_title(f'Normal Mode {mode_idx} - {identifier}')
        
        # Animation function
        def update(frame_idx):
            frame = frames[frame_idx]
            line.set_data(frame[:, 0], frame[:, 1])
            line.set_3d_properties(frame[:, 2])
            return line,
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=range(n_frames), interval=50, blit=True)
        
        # Save as GIF
        try:
            gif_file = os.path.join(self.animation_dir, f"{identifier}_mode{mode_idx}.gif")
            ani.save(gif_file, writer='pillow', fps=10, dpi=100)
            print(f"Animation saved to {gif_file}")
        except Exception as e:
            print(f"Could not save animation: {str(e)}")
            # Save key frames as images instead
            self.save_key_frames(frames, identifier, mode_idx)
        
        plt.close()
    
    def save_key_frames(self, frames, identifier, mode_idx, n_frames=8):
        """Save key frames from a trajectory as images"""
        step = len(frames) // n_frames
        
        for i, frame_idx in enumerate(range(0, len(frames), step)):
            if i >= n_frames:
                break
                
            frame = frames[frame_idx]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(frame[:, 0], frame[:, 1], frame[:, 2], 'b-', linewidth=2)
            
            # Set axis limits
            max_range = np.max(frame.max(axis=0) - frame.min(axis=0)) * 1.2
            mid_x = (frame[:, 0].max() + frame[:, 0].min()) / 2
            mid_y = (frame[:, 1].max() + frame[:, 1].min()) / 2
            mid_z = (frame[:, 2].max() + frame[:, 2].min()) / 2
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            ax.set_title(f'Mode {mode_idx} - Frame {i} - {identifier}')
            
            img_file = os.path.join(self.animation_dir, f"{identifier}_mode{mode_idx}_frame{i}.png")
            plt.savefig(img_file, dpi=150)
            plt.close()
            
            print(f"Frame saved to {img_file}")

def main():
    # Load data with RMSD values
    with open("creative_top_performers_with_rmsd.json", "r") as f:
        data = json.load(f)
    
    # Sort by RMSD and get top 5
    sorted_data = sorted(data, key=lambda x: x.get('rmsd', float('inf')))
    top_structures = sorted_data[:5]  # Analyze top 5 structures
    
    # Initialize simulator
    simulator = ProteinDynamicsSimulator()
    
    # Analyze structures
    results = []
    
    for i, item in enumerate(top_structures):
        if 'aligned_pdb' in item:
            pdb_file = item['aligned_pdb']
            identifier = item.get('id', str(i))
            rmsd = item.get('rmsd', 0)
            
            print(f"\nAnalyzing dynamics for structure {identifier} (RMSD: {rmsd:.4f})")
            
            try:
                # Calculate normal modes
                modes_data = simulator.calculate_normal_modes(pdb_file, identifier)
                
                # Generate trajectory
                trajectories = simulator.generate_trajectory(
                    modes_data['coords'], 
                    modes_data['modes'], 
                    identifier
                )
                
                # Store results
                results.append({
                    'identifier': identifier,
                    'rmsd_to_original': rmsd,
                    'eigenvalues': modes_data['eigenvalues'][:10]  # Store first 10 eigenvalues
                })
                
                print(f"Completed dynamics analysis for structure {identifier}")
                
            except Exception as e:
                print(f"Error processing structure {identifier}: {str(e)}")
    
    # Save results
    with open(os.path.join(simulator.output_dir, "dynamics_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nProtein Dynamics Summary:")
    print("ID\tRMSD\tLowest Frequency")
    print("-" * 40)
    
    for result in results:
        lowest_freq = result['eigenvalues'][0] if result['eigenvalues'] else float('inf')
        print(f"{result['identifier']}\t{result['rmsd_to_original']:.4f}\t{lowest_freq:.6f}")
    
    print("\nStructures with lowest frequencies (most flexible):")
    sorted_by_freq = sorted(results, key=lambda x: x['eigenvalues'][0] if x['eigenvalues'] else float('inf'))
    for result in sorted_by_freq[:3]:
        print(f"Structure {result['identifier']}:")
        print(f"  RMSD to original: {result['rmsd_to_original']:.4f}")
        print(f"  Lowest frequency: {result['eigenvalues'][0]:.6f}")
        print()

if __name__ == "__main__":
    main() 