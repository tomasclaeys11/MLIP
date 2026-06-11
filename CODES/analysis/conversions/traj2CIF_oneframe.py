from ase.io import read, write
traj = read(r'geengithub\HPC_bin\MD_results\MD_500_404_perovskite\MD_500C_HEO_404_perovskite_equil.traj',':')

final_frame = traj[-1]

write('MD_500C_HEO_404_perovskite_finalframe.cif',final_frame)