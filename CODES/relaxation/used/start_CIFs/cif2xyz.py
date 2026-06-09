from ase.io import read, write 

atoms = read(r"CODES\relaxation\used\start_CIFs\PdCuCoNiCrO_unrelaxed_perovskite_3x3x3.cif")
write(r"CODES\relaxation\used\start_CIFs\PdCuCoNiCrO_unrelaxed_perovskite_3x3x3.xyz",atoms)