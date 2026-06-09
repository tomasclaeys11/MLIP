import os
import datetime
import torch
import numpy as np
import pandas as pd
from monty.serialization import dumpfn

# ASE Imports
from ase.io import write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory

# Pymatgen Imports
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN

# CHGNet Imports
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator

# --- 1. SETUP & RUN ID [Upgrade 10] ---
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CATIONS = ["Pd", "Cu", "Co", "Ni", "Cr"]
oxi_states = {"Cr": 3, "Co": 2, "Ni": 2, "Cu": 2, "Pd": 2, "O": -2}
FMAX_TOLERANCE = 0.05
SUPERCELL_DIMENSIONS = (3, 3, 3)

current_dir = os.getcwd()
output_folder = os.path.join(current_dir, f"Results_{''.join(CATIONS)}_{RUN_ID}")
os.makedirs(output_folder, exist_ok=True)

# Path to your 216-atom base structure
CIF_INPUT = "CODES\start_CIFs\PdCuCoNiCr_unrelaxed_rocksalt.cif"

# --- 2. IMPROVED SQS OPTIMIZATION 
def get_sro_penalty(struct, indices):
    """Simple penalty function to minimize cation clustering."""
    penalty = 0
    for idx in indices:
        site = struct[idx]
        neighbors = struct.get_neighbors(site, 3.5)
        for nb in neighbors:
            if nb.specie == site.specie:
                penalty += 1
    return penalty

print(f"--- Optimizing SQS via Simulated Annealing ---")
heo = Structure.from_file(CIF_INPUT)
cat_indices = [i for i, site in enumerate(heo) if site.specie.symbol != "O"]
temperature = 0.05

for step in range(5000): # Increased steps for better disorder
    i1, i2 = np.random.choice(cat_indices, 2, replace=False)
    if heo[i1].specie == heo[i2].specie:
        continue

    curr_sro = get_sro_penalty(heo, [i1, i2])
    sym1, sym2 = heo[i1].specie, heo[i2].specie

    heo.replace(i1, sym2)
    heo.replace(i2, sym1)

    new_sro = get_sro_penalty(heo, [i1, i2])
    delta = new_sro - curr_sro

    if not (delta <= 0 or np.random.rand() < np.exp(-delta / temperature)):
        heo.replace(i1, sym1)
        heo.replace(i2, sym2)

    temperature *= 0.9995 # Cooling schedule

# --- 3. RELAXATION WITH TRAJECTORY [Upgrade 1] ---
print("\n--- Starting CHGNet Relaxation ---")
model = CHGNet.load()
adaptor = AseAtomsAdaptor()
atoms_heo = adaptor.get_atoms(heo)
atoms_heo.calc = CHGNetCalculator(model)

traj_file = os.path.join(output_folder, f"Relaxation_{''.join(CATIONS)}.traj")
opt = BFGS(
    FrechetCellFilter(atoms_heo),
    trajectory=traj_file,
    logfile='-'
)
opt.run(fmax=FMAX_TOLERANCE)

# --- 4. EXTRACT RELAXED DATA [Upgrade 9] ---
relaxed_struct = adaptor.get_structure(atoms_heo)
oxi_states = {"Mn": 5.0, "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0, "Cr": 3.0, "Mo": 0.0, "Ti": 0.0, "Pd": 0.0, "Ce": 0.0, "Zn": 0.0, "Mg": 0.0, "Zr": 0.0, "Al": 0.0, "O": -2}
relaxed_struct.add_oxidation_state_by_element(oxi_states)
moms_signed = atoms_heo.get_magnetic_moments()
moms_abs = np.abs(moms_signed)

# --- 5. NEIGHBOR CACHING & DESCRIPTORS [Upgrade 5 & 12] ---
print("--- Calculating Site Descriptors ---")
cnn = CrystalNN()
neighbor_cache = {}
for i in range(len(relaxed_struct)):
    neighbor_cache[i] = cnn.get_nn_info(relaxed_struct, i)

# Definitions for Spin Fidelity Index (SFI) and Ionicity [Upgrade 11]
# Assume target magnetic moments for these cations in an oxide environment
target_moms = {"Pd": 0.0, "Cu": 0.6, "Co": 2.5, "Ni": 1.7, "Cr": 3.0}

sfi_list = []
ionicity_list = []
distortions = []

for i, site in enumerate(relaxed_struct):
    symbol = site.specie.symbol
    if symbol != "O":
        m_abs = moms_abs[i]
        m_target = target_moms.get(symbol, 1.0)
        sfi = 1 - (abs(m_abs - m_target) / (m_target + 0.01))
        sfi_list.append(max(0, sfi))
        
        # Bond Length Distortion for Cations
        bonds = [site.distance(nb['site']) for nb in neighbor_cache[i]]
        dist_val = np.std(bonds) if bonds else 0
        distortions.append(dist_val)
        
    else:
        sfi_list.append(0)
        distortions.append(0)

# --- 6. STORE PROPERTIES & DIGITAL TWIN [Upgrade 3 & 4] ---
relaxed_struct.add_site_property("magmom_signed", list(moms_signed))
relaxed_struct.add_site_property("magmom_abs", list(moms_abs))
relaxed_struct.add_site_property("spin_fidelity", sfi_list)
relaxed_struct.add_site_property("distortion_index", distortions)

json_file = os.path.join(output_folder, f"DigitalTwin_{''.join(CATIONS)}.json")
dumpfn(relaxed_struct, json_file)

# --- 7. EXPORT CIF & EXTXYZ [Upgrade 2 & 7] ---
relaxed_struct.to(filename=os.path.join(output_folder, f"Relaxed_{''.join(CATIONS)}.cif"))

atoms_export = adaptor.get_atoms(relaxed_struct)
atoms_export.arrays["magmom"] = moms_signed
atoms_export.arrays["spin_fidelity"] = np.array(sfi_list)
write(os.path.join(output_folder, f"Relaxed_{''.join(CATIONS)}.extxyz"), atoms_export)

# --- 8. ENERGETICS SUMMARY [Upgrade 8] ---
summary = {
    "Run_ID": RUN_ID,
    "Potential_Energy_eV": atoms_heo.get_potential_energy(),
    "Mean_Spin_Fidelity": np.mean([s for s in sfi_list if s > 0]),
    "FMAX": FMAX_TOLERANCE,
    "Num_Atoms": len(relaxed_struct)
}
pd.DataFrame([summary]).to_csv(os.path.join(output_folder, "Summary.csv"), index=False)

print(f"\n--- RELAXATION COMPLETE ---")
print(f"Results stored in: {output_folder}")