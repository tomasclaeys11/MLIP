import os
import sys

# Environmental setup
env_path = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(env_path):
    os.add_dll_directory(env_path)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from chgnet.model.model import CHGNet
import warnings

warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
model = CHGNet.load()

# --- SCIENTIFIC CONSTANTS (Strict Reference States) ---
ideal_n_unpaired = {
    "Mg": 0.0, "Ti": 0.0, "V": 3.0, "Cr": 3.0, "Mn": 5.0, 
    "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0, "Zn": 0.0, 
    "Mo": 0.0, "Pd": 2.0, "Ce": 0.0, "Zr": 0.0, "Al": 0.0
}

formal_charge = {
    "Mg": 2.0, "Ti": 4.0, "V": 3.0, "Cr": 3.0, "Mn": 2.0, 
    "Fe": 3.0, "Co": 2.0, "Ni": 2.0, "Cu": 2.0, "Zn": 2.0, 
    "Mo": 6.0, "Pd": 2.0, "Ce": 4.0, "Zr": 4.0, "Al": 3.0
}

def write_global_cif(structure, data, filename):
    with open(filename, 'w') as f:
        f.write("data_HEO_Bohr_Individual_Sites\n")
        f.write(f"_cell_length_a {structure.lattice.a:.6f}\n")
        f.write(f"_cell_length_b {structure.lattice.b:.6f}\n")
        f.write(f"_cell_length_c {structure.lattice.c:.6f}\n")
        f.write(f"_cell_angle_alpha {structure.lattice.alpha:.6f}\n")
        f.write(f"_cell_angle_beta {structure.lattice.beta:.6f}\n")
        f.write(f"_cell_angle_gamma {structure.lattice.gamma:.6f}\n")
        f.write("_symmetry_space_group_name_H-M 'P 1'\n\n")
        f.write("loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n") 
        for i, site in enumerate(structure):
            sym = site.specie.symbol
            x, y, z = site.frac_coords
            f.write(f"{sym}{i+1} {sym} {x:.5f} {y:.5f} {z:.5f} {data[i]:.5f}\n")

# --- EXECUTION ---
target_folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f)) and f.startswith("Results_")]

for folder in target_folders:
    comp_name = folder.replace("Results_", "")
    folder_path = os.path.join(current_dir, folder)
    relaxed_cif = os.path.join(folder_path, f"Relaxed_{comp_name}.cif")
    
    if not os.path.exists(relaxed_cif): continue
        
    print(f"Bohr Analysis (Individual + Avg): {comp_name}")
    struct = Structure.from_file(relaxed_cif)
    pred = model.predict_structure(struct)
    
    cations = list(set([s.specie.symbol for s in struct if s.specie.symbol != "O"]))
    
    # Track site-specific charges for CIF and element-specific lists for CSV average
    site_charges = []
    element_charge_lists = {cat: [] for cat in cations}
    element_mu_lists = {cat: [] for cat in cations}
    
    for i, site in enumerate(struct):
        sym = site.specie.symbol
        if sym in cations:
            # 1. Get local magmom
            m_raw = pred['m'][i]
            m_local = np.abs(float(m_raw.detach().cpu().numpy())) if torch.is_tensor(m_raw) else np.abs(float(m_raw))
            
            # 2. Convert to local unpaired electrons
            n_local = np.sqrt(m_local**2 + 1) - 1
            
            # 3. Calculate local effective charge
            # eff_q = Formal_Target + (n_local - n_target)
            q_target = formal_charge.get(sym, 2.0)
            n_target = ideal_n_unpaired.get(sym, 0.0)
            eff_q_local = q_target + (n_local - n_target)
            
            # 4. Store data
            site_charges.append(round(eff_q_local, 5))
            element_charge_lists[sym].append(eff_q_local)
            element_mu_lists[sym].append(m_local)
        else:
            site_charges.append(0.0) # Oxygen
            
    # --- EXPORT CIF (Individual site data for scatter plotting) ---
    write_global_cif(struct, site_charges, os.path.join(folder_path, f"Bohr_Individual_{comp_name}.cif"))
    
    # --- EXPORT CSV (Element-averaged data for final tables) ---
    csv_res = {"Comp": comp_name}
    for cat in cations:
        avg_q = np.mean(element_charge_lists[cat])
        avg_mu = np.mean(element_mu_lists[cat])
        csv_res[f"{cat}_eff_q_avg"] = round(avg_q, 4)
        csv_res[f"{cat}_mu_avg"] = round(avg_mu, 4)
        
    pd.DataFrame([csv_res]).to_csv(os.path.join(folder_path, f"Bohr_Averaged_{comp_name}.csv"), index=False)

print("\nDone! Individual CIFs and Averaged CSVs have been saved.")