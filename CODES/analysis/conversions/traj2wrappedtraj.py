from ase.io.trajectory import Trajectory
from ase.io import write

def wrap_trajectory(input_path, output_path):
    # Load the existing binary .traj file
    orig_traj = Trajectory(input_path)
    
    # Create a new trajectory file to save the wrapped frames
    new_traj = Trajectory(output_path, 'w')
    
    print(f"Wrapping {len(orig_traj)} frames...")
    
    for atoms in orig_traj:
        # This command moves any atom outside the box back to the opposite side
        atoms.wrap()
        
        # Save the modified atoms object
        new_traj.write(atoms)
    
    new_traj.close()
    print(f"Done! Wrapped trajectory saved to: {output_path}")

# Usage
wrap_trajectory("HPC_bin/MD_results/MD_216_PdCuCoNiCr/movies/MD_700C_PdCuCoNiCr (1).traj", "wrapped_MD_700C_PdCuCoNiCr.traj")