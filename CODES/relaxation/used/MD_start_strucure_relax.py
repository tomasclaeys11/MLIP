import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter as FrechetCellFilter
from ase.io import write

# 1. SETUP
CATIONS = ["Mg", "Cu", "Co", "Ni", "Zn"]
model = CHGNet.load()
adaptor = AseAtomsAdaptor()

# 2. CREATE 216-ATOM SQS (3x3x3 Supercell)
# Generating the Rocksalt (Fm-3m) lattice
prim = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.25), ["Mg", "O"], [[0,0,0], [0.5,0.5,0.5]])
heo = prim.make_supercell([3, 3, 3]) # This yields exactly 216 atoms (108 Cations, 108 Oxygens)

# Distribute cations randomly
cat_indices = [i for i, s in enumerate(heo) if s.specie.symbol == "Mg"]
atoms_per_metal = len(cat_indices) // len(CATIONS)
metal_list = np.repeat(CATIONS, atoms_per_metal)

# Handle remainder to ensure we hit exactly 108 cations
while len(metal_list) < len(cat_indices): 
    metal_list = np.append(metal_list, CATIONS[-1])

np.random.shuffle(metal_list)
for i, idx in enumerate(cat_indices): 
    heo.replace(idx, metal_list[i])

# Convert to ASE Atoms object
atoms_heo = adaptor.get_atoms(heo)

# --- NEW: SAVE THE UNRELAXED ORIGINAL ---
# This is your "Initial State" before CHGNet moves anything
write("MD_unrelaxedstructure.cif", atoms_heo)
print("Saved unrelaxed starting structure to MD_Start_PdCuCoNiCr_216_UNRELAXED.cif")

# --- 3. RELAXATION ---
print("Relaxing the 216-atom cell for Molecular Dynamics...")
atoms_heo.calc = CHGNetCalculator(model)

# BFGS relaxation with Cell Filter (allows volume/shape to optimize)
optimizer = BFGS(FrechetCellFilter(atoms_heo), logfile='-')
optimizer.run(fmax=0.05)

# --- 4. EXPORT RELAXED STARTING POINT ---
output_file = "MD_Start_PdCuCoNiCr_216_RELAXED.cif"
atoms_heo.write(output_file)

print(f"\nSuccess!")
print(f"1. Unrelaxed structure: MD_Start_PdCuCoNiCr_216_UNRELAXED.cif")
print(f"2. Relaxed structure (Ready for MD): {output_file}")