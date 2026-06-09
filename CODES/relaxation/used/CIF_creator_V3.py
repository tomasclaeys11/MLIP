import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter #

# =========================================================
# 1. USER INPUTS
# =========================================================
CATIONS = ["Pd", "Cu", "Co", "Ni", "Cr"]
CATION_OXI = {"Cu": 2, "Ni": 2, "Cr": 3, "Co": 2, "Pd": 2}
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

# Expand to supercell
template.make_supercell(SUPERCELL_DIM) 

# =========================================================
# 4. RANDOM SHUFFLE MAPPING
# =========================================================
print(f"--- Mapping cations into {len(template)} atom supercell ---")

cation_indices = [i for i, site in enumerate(template) if site.specie.symbol != ANION]
num_cation_sites = len(cation_indices)

# Generate balanced distribution
repeats = int(np.ceil(num_cation_sites / len(CATIONS)))
cation_fill_list = (CATIONS * repeats)[:num_cation_sites]
np.random.shuffle(cation_fill_list) # Pure V1 random shuffle mix

# Map the shuffled elements into the structure
for i, cat_idx in enumerate(cation_indices):
    template.replace(cat_idx, cation_fill_list[i])

# =========================================================
# 4.5 NATIVE WARREN-COWLEY (SRO) ANALYSIS
# =========================================================
print("\n--- Warren-Cowley Short-Range Order (SRO) Analysis ---")

target_conc = 1.0 / len(CATIONS)
cutoff_radius = 3.5  # Captures immediate first-shell cation-cation neighbors (~2.98 A)

sro_tracking = {}
for cat_a in CATIONS:
    for cat_b in CATIONS:
        pair_key = tuple(sorted((cat_a, cat_b)))
        if pair_key not in sro_tracking:
            sro_tracking[pair_key] = []

for idx in cation_indices:
    current_site = template[idx]
    element_a = current_site.specie.symbol
    
    all_neighbors = template.get_neighbors(current_site, r=cutoff_radius)
    only_cations = [n for n in all_neighbors if n.specie.symbol != ANION]
    
    if not only_cations:
        continue
        
    for element_b in CATIONS:
        observed_b_count = sum(1 for n in only_cations if n.specie.symbol == element_b)
        p_ij = observed_b_count / len(only_cations)
        alpha_ij = 1.0 - (p_ij / target_conc)
        
        pair_key = tuple(sorted((element_a, element_b)))
        sro_tracking[pair_key].append(alpha_ij)

print(f"{'Pair':<12} | {'WC Parameter':<15}")
print("-" * 32)
for pair_key, values in sorted(sro_tracking.items()):
    print(f"{str(pair_key):<12} | {np.mean(values):>15.4f}")

# =========================================================
# 5. ASSIGN OXIDATION STATES
# =========================================================
print("\nApplying oxidation states for CrystalNN robustness...")
oxi_map = {**CATION_OXI, ANION: ANION_OXI}
template.add_oxidation_state_by_element(oxi_map)

# =========================================================
# 6. EXPORT VIA CIFWRITER
# =========================================================
comp_str = "".join(CATIONS)
filename = f"{comp_str}_unrelaxed_{CRYSTAL_STRUCTURE}_{SUPERCELL_DIM}.cif"
output_path = os.path.join(output_folder, filename)

# Initialize the writer with a symmetry precision of 0.1
writer = CifWriter(template, symprec=0.1)

# Force an explicit Python file handler to bypass OneDrive/Windows sync locks
print(f"Writing CIF file to: {output_path}")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(str(writer))