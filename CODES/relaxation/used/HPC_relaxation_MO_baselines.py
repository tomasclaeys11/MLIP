import os
import sys

# Force Python to look at the environment's library folder first
env_path = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(env_path):
    os.add_dll_directory(env_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.optimize import BFGS
try:
    from ase.filters import FrechetCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter as FrechetCellFilter

import warnings

# --- 1. SETUP ---
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))

# UPDATE THESE FOR EACH FOLDER
# --- 1. SETUP ---
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))

# NEW: Dynamic Cation Loading from Command Line
if len(sys.argv) > 1:
    # Captures everything after 'python relaxation.py' as a list
    CATIONS = sys.argv[1:] 
else:
    # Fallback default if you run it locally without arguments
    CATIONS = ["Pd", "Cu", "Co", "Ni", "Cr"] 

# This ensures folders are named correctly for each job
output_folder = os.path.join(current_dir, f"Results_{''.join(CATIONS)}")
os.makedirs(output_folder, exist_ok=True)
torch.set_num_threads(8)
model = CHGNet.load() 
adaptor = AseAtomsAdaptor()
FMAX_TOLERANCE = 0.05 

# --- 2. SRO PENALTY ---
def get_sro_penalty(struct, indices):
    penalty = 0
    for idx in indices:
        for nb in struct.get_neighbors(struct[idx], 3.1):
            if nb.specie.symbol == struct[idx].specie.symbol:
                penalty += 1
    return penalty

# --- 3. UNIVERSAL PHASE-ACCURATE BASELINES (Literature Rock-Salt Standard) ---
def get_baselines():
    print("\n--- Calculating Ground-State Phase Baselines (Fm-3m Rock-Salt) ---")
    energies = []
    for metal in CATIONS:
        print(f"  > Relaxing pure {metal} oxide baseline (Rock-Salt)...")
        
        # Force every element into the Fm-3m rock-salt structure
        # Starting with a standard 4.25 A lattice, which CHGNet will relax to equilibrium
        struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.25), [metal, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

        atoms = adaptor.get_atoms(struct)
        atoms.calc = CHGNetCalculator(model)
        
        # We can drop the fmax=1.0 pre-relaxation step since we aren't using complex lattices anymore
        BFGS(FrechetCellFilter(atoms), logfile='-').run(fmax=FMAX_TOLERANCE)
        energies.append(atoms.get_potential_energy() / len(atoms))
        
    return np.mean(energies)

# --- 4. SQS & RELAXATION ---
baseline_e = get_baselines()
prim = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.25), ["Mg", "O"], [[0,0,0], [0.5,0.5,0.5]])
SUPERCELL_DIMENSIONS = [4, 4, 5] 
heo = prim.make_supercell(SUPERCELL_DIMENSIONS)

cat_indices = [i for i, s in enumerate(heo) if s.specie.symbol == "Mg"]
num_cations = len(cat_indices)
atoms_per_metal = int(num_cations / len(CATIONS))

if num_cations % len(CATIONS) != 0:
    metal_list = []
    for m in CATIONS: metal_list.extend([m]*atoms_per_metal)
    while len(metal_list) < num_cations: metal_list.append(CATIONS[-1])
else:
    metal_list = np.repeat(CATIONS, atoms_per_metal)

np.random.shuffle(metal_list)
for i, idx in enumerate(cat_indices): heo.replace(idx, metal_list[i])

print(f"Optimizing SQS...")
for _ in range(2000):
    i1, i2 = np.random.choice(cat_indices, 2, replace=False)
    if heo[i1].specie == heo[i2].specie: continue
    curr_sro = get_sro_penalty(heo, [i1, i2])
    sym1, sym2 = heo[i1].specie, heo[i2].specie
    heo.replace(i1, sym2); heo.replace(i2, sym1)
    if get_sro_penalty(heo, [i1, i2]) > curr_sro: heo.replace(i1, sym1); heo.replace(i2, sym2)

atoms_heo = adaptor.get_atoms(heo)
atoms_heo.calc = CHGNetCalculator(model)
BFGS(FrechetCellFilter(atoms_heo), logfile='-').run(fmax=FMAX_TOLERANCE)

# --- 5. STRUCTURAL ANALYSIS ---
e_heo = atoms_heo.get_potential_energy() / len(atoms_heo)
dH_mix = (e_heo - baseline_e) * 1000
relaxed_struct = adaptor.get_structure(atoms_heo)
global_bonds = [nb.nn_distance for site in relaxed_struct if site.specie.symbol in CATIONS for nb in relaxed_struct.get_neighbors(site, 3.0) if nb.specie.symbol == "O"]
mean_b, sigma_b = np.mean(global_bonds), np.std(global_bonds)

# --- 6. CONTINUOUS ANALYSIS (Spin Fidelity & Ionicity Descriptor) ---
print("\n--- Calculating Spin Fidelity Index & Ionicity Descriptors ---")
prediction = model.predict_structure(relaxed_struct)

# Physical baselines for roster (rock-salt) ions
valence_map = {"Mg": 2, "Ti": 4, "V": 5, "Cr": 3, "Mn": 2, "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2, "Zn": 2, "Mo": 6, "Pd": 2, "Ce": 4, "Zr": 4, "Al": 3}
ideal_m = {"Mn": 5.0, "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0, "Cr": 3.0, "Mo": 0.0, "Ti": 0.0, "Pd": 0.0, "Ce": 0.0, "Zn": 0.0, "Mg": 0.0, "Zr": 0.0, "Al": 0.0}

element_results = {cat: {"moms": [], "ionicity": [], "distortions": [], "sfi": []} for cat in CATIONS}
sfi_individual = [] 
ionicity_individual = []
distortions_global = []
moms_per_atom = [] # SAVE RAW DATA FOR REUSE

for i, site in enumerate(relaxed_struct):
    sym = site.specie.symbol
    
    # Structural Distortion (D_idx)
    local_bonds = [nb.nn_distance for nb in relaxed_struct.get_neighbors(site, 3.0)]
    d_idx = (1.0 / len(local_bonds)) * sum([abs(b - np.mean(local_bonds)) / np.mean(local_bonds) for b in local_bonds]) if local_bonds else 0.0
    distortions_global.append(d_idx)

    if sym in CATIONS:
        m_t = prediction['m'][i]
        m_val = np.abs(float(m_t.detach().cpu().numpy()) if torch.is_tensor(m_t) else float(m_t))
        moms_per_atom.append(m_val)
        
        # Invert spin-only formula
        n_calc = np.sqrt(m_val**2 + 1) - 1
        q_target = valence_map.get(sym, 2.0)
        n_target = ideal_m.get(sym, 0.0)
        
        # 2. Spin Fidelity Index (SFI) & Ionicity Descriptor
        if n_target > 0:
            sfi = n_calc / n_target 
            ionicity = q_target * sfi # Multiplicative scaling
        else:
            sfi = 1.0 
            ionicity = q_target + n_calc # Additive for Pd/Ti
        
        element_results[sym]["moms"].append(m_val)
        element_results[sym]["sfi"].append(sfi)
        element_results[sym]["ionicity"].append(ionicity)
        element_results[sym]["distortions"].append(d_idx)
        
        sfi_individual.append(sfi)
        ionicity_individual.append(ionicity)
    else:
        # Oxygen sites
        sfi_individual.append(0.0)
        ionicity_individual.append(0.0)
        moms_per_atom.append(0.0)

# --- 7. EXPORTS (CIF for Visual Heatmaps) ---
# We keep the function but call it for our new physics descriptors
def write_global_cif(structure, data, filename):
    with open(filename, 'w') as f:
        f.write("data_HEO_Global\n")
        f.write(f"_cell_length_a {structure.lattice.a:.6f}\n_cell_length_b {structure.lattice.b:.6f}\n_cell_length_c {structure.lattice.c:.6f}\n")
        f.write(f"_cell_angle_alpha {structure.lattice.alpha:.6f}\n_cell_angle_beta {structure.lattice.beta:.6f}\n_cell_angle_gamma {structure.lattice.gamma:.6f}\n")
        f.write("_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n")
        for i, site in enumerate(structure):
            f.write(f"{site.specie.symbol}{i+1} {site.specie.symbol} {site.frac_coords[0]:.5f} {site.frac_coords[1]:.5f} {site.frac_coords[2]:.5f} {data[i]:.5f}\n")

# Generating the three main visual proofs for the meeting
write_global_cif(relaxed_struct, sfi_individual, os.path.join(output_folder, f"Global_SFI_{''.join(CATIONS)}.cif"))
write_global_cif(relaxed_struct, ionicity_individual, os.path.join(output_folder, f"Global_Ionicity_{''.join(CATIONS)}.cif"))
write_global_cif(relaxed_struct, distortions_global, os.path.join(output_folder, f"Global_Distortion_{''.join(CATIONS)}.cif"))

# --- 8. MASTER DATA (The "Digital Twin" for re-use) ---
analysis_list = []
for i, site in enumerate(relaxed_struct):
    analysis_list.append({
        "ID": f"{site.specie.symbol}{i+1}",
        "Element": site.specie.symbol,
        "Mag_Mom": moms_per_atom[i], # Raw moments saved here!
        "SFI": sfi_individual[i] if site.specie.symbol in CATIONS else 0,
        "Ionicity_Index": ionicity_individual[i] if site.specie.symbol in CATIONS else 0,
        "Distortion_Index": distortions_global[i]
    })
pd.DataFrame(analysis_list).to_csv(os.path.join(output_folder, f"Bivariate_Master_Data_{''.join(CATIONS)}.csv"), index=False)
final_summary_file = os.path.join(output_folder, f"Final_Results_{''.join(CATIONS)}.csv")

summary_data = {
    "Comp": "".join(CATIONS),
    "dH_mix": dH_mix,                # Already calculated in Section 5
    "bond_avg_global": mean_b,        # Already calculated in Section 5
    "bond_std_global": sigma_b        # Already calculated in Section 5
}

# Save as a single-row CSV
pd.DataFrame([summary_data]).to_csv(final_summary_file, index=False)