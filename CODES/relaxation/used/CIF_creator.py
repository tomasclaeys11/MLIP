import os
import numpy as np
from pymatgen.core import Structure, Lattice
from sqsgenerator import optimize, parse_config

# =========================================================
# 1. USER INPUTS
# =========================================================
CATIONS = ["Pd", "Cu", "Co", "Ni", "Mg"]
CATION_OXI = {"Pd": 2, "Cu": 2, "Co": 2, "Ni": 2, "Mg": 2}
ANION = "O"
ANION_OXI = -2

CRYSTAL_STRUCTURE = "rocksalt" 
SUPERCELL_DIM = (3, 3, 3) 

# =========================================================
# 2. DIRECTORY SETUP
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_dir, "start_CIFs")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =========================================================
# 3. GENERATE 8-ATOM CONVENTIONAL TEMPLATE
# =========================================================
def create_template(structure_type):
    if structure_type.lower() == "rocksalt":

        lat = Lattice.cubic(4.21)
        

        species = ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], # Cation sites
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]  # Anion sites
        ]
        return Structure(lat, species, coords)
    else:
        raise ValueError("Structure type not supported.")

print(f"--- Generating {CRYSTAL_STRUCTURE} 8-atom conventional cell ---")
template = create_template(CRYSTAL_STRUCTURE)

# Expand to x atoms 
template.make_supercell(SUPERCELL_DIM) 

# =========================================================
# 4. REPLACE ATOMS & ASSIGN OXIDATION STATES
# =========================================================
print(f"--- Mapping cations into {len(template)} atom supercell ---")

cation_indices = [i for i, site in enumerate(template) if site.specie.symbol != ANION]
num_cation_sites = len(cation_indices)

# Balanced distribution
repeats = int(np.ceil(num_cation_sites / len(CATIONS)))
cation_fill_list = (CATIONS * repeats)[:num_cation_sites]
np.random.shuffle(cation_fill_list) # Initial rough mix

oxi_map = {**CATION_OXI, ANION: ANION_OXI}

for i, cat_idx in enumerate(cation_indices):
    template.replace(cat_idx, cation_fill_list[i])

# Assign oxidation states for CrystalNN robustness
template.add_oxidation_state_by_element(oxi_map)

print(f"--- Starting SQS Optimization for {len(cation_indices)} cation sites ---")

# Define target composition for SQS
# For a 180 cation supercell (3x3x5), this gives 36 of each cation
target_comp = {cat: int(num_cation_sites / len(CATIONS)) for cat in CATIONS}

sqs_config = {
    "structure": template,
    "sublattice": [cation_indices],  # Only swap the cations, leave Oxygen fixed
    "composition": target_comp,
    "iterations": 100000,            # Number of Monte Carlo swaps to try
    "objective": [1.0, 1.0],         # Optimize 1st and 2nd neighbor shells
    "threads": -1                    # Use all available CPU cores
}

# Run the optimization loop
settings = parse_config(sqs_config)
results, _ = optimize(settings)

# Select the best structure (the one with lowest SRO/WC parameters)
best_sqs = results[0]
template = best_sqs.structure()

# Extract and Print Warren-Cowley (SRO) parameters for verification
print("\n--- Final Warren-Cowley Parameters (First Shell) ---")
# WC = 0 means perfectly random; -1 is clustered; +1 is ordered
sro_matrix = best_sqs.sro()[0] 
print(sro_matrix)

# =========================================================
# 5. EXPORT
# =========================================================
comp_str = "".join(CATIONS)
filename = f"{comp_str}_unrelaxed_{CRYSTAL_STRUCTURE}_{SUPERCELL_DIM}.cif"
output_path = os.path.join(output_folder, filename)

template.to(filename=output_path)

print("\n" + "="*40)
print(f"SUCCESS: {len(template)} ATOM CIF GENERATED")
print("="*40)
print(f"File:      {filename}")
print(f"Supercell: {SUPERCELL_DIM}")
print("="*40)