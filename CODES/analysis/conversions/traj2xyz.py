from ase.io import read, write

# Load the entire trajectory
trajectory = read('MLIP\MD_analysis\MD_700C_HEO_prod.traj', index=':')

# Write it out as a multi-frame XYZ file
write('wrapped_MD_700C_PdCuCoNiCr.xyz', trajectory, format='extxyz')
print("Conversion complete! Open MD_Animation.xyz in OVITO.")