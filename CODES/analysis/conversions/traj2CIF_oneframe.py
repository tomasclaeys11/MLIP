from ase.io import read, write
traj = read(r'relaxation_V3_results\PdCuCoNiCr\Results_PdCuCoNiCr_20260507_150820\Relaxation_20260507_150820.traj',':')

final_frame = traj[-1]

write('relaxed_PdCuCoNiCr.cif',final_frame)