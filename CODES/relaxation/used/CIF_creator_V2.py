import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter #
from icet import ClusterSpace #
# FIXED: Correct sub-module import for SQS tools
from icet.tools.structure_generation import generate_sqs_from_supercells 
from icet.tools import get_short_range_order_parameters

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
os.makedirs(output_folder, exist_ok=True)

# =========================================================
# 3. GENERATE SKELETON (Pymatgen -> ASE)
# =========================================================
def create_template(structure_type):
    if structure_type.lower() == "rocksalt":
        lat = Lattice.cubic(4.21)
        species = ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
        coords = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5],
                  [0.5,0.5,0.5], [0,0,0.5], [0,0.5,0], [0.5,0,0]]
        return Structure(lat, species, coords)
    raise ValueError("Structure type not supported.")

template_pmg = create_template(CRYSTAL_STRUCTURE)
template_pmg.make_supercell(SUPERCELL_DIM)

# Convert to ASE atoms for icet processing
adaptor = AseAtomsAdaptor()
atoms = adaptor.get_atoms(template_pmg)

# =========================================================
# 4. ICET SQS GENERATION
# =========================================================
print(f"--- Initializing icet ClusterSpace for {len(atoms)} sites ---")

# 4.1 Define allowed symbols for the 8-atom basis (4 cations, 4 anions)
# This MUST match the order in your create_template function: ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
basis_symbols = [CATIONS] * 4 + [[ANION]] * 4

# 4.2 Create ClusterSpace using the 8-atom template
# This is much faster and prevents the length mismatch error
template_small_atoms = adaptor.get_atoms(create_template(CRYSTAL_STRUCTURE))
cs = ClusterSpace(
    structure=template_small_atoms, 
    cutoffs=[6.0], 
    chemical_symbols=basis_symbols
)

# 4.3 Define target concentrations (0.2 for each of the 5 cations)
target_concentrations = {cat: 1/len(CATIONS) for cat in CATIONS}

# 4.4 Run SQS Generation on the 360-atom supercell
print("Optimizing cation distribution via Simulated Annealing...")
sqs_atoms = generate_sqs_from_supercells(
    cluster_space=cs,
    supercells=[atoms], # Your 360-atom atoms object
    target_concentrations=target_concentrations,
    n_steps=5000 
)
# =========================================================
# 4.5 WARREN-COWLEY (SRO) ANALYSIS
# =========================================================
print("\n--- Warren-Cowley Short-Range Order (SRO) Analysis ---")

# Calculate SRO for the first neighbor shell (shell_index=1)
# WC = 0 (Random), WC < 0 (Clustering), WC > 0 (Ordering)
sro_shell_1 = get_short_range_order_parameters(sqs_atoms, cs, shell_index=1)

# Print a nice table of results
print(f"{'Pair':<10} | {'WC Parameter':<15}")
print("-" * 30)
for pair, value in sro_shell_1.items():
    # We only care about cation-cation pairs
    if ANION not in pair:
        print(f"{str(pair):<10} | {value:>15.4f}")

# Convert back to Pymatgen
template_pmg = adaptor.get_structure(sqs_atoms)

# =========================================================
# 5. ASSIGN OXIDATION STATES
# =========================================================
print("Finalizing structure and applying oxidation states...")
oxi_map = {**CATION_OXI, ANION: ANION_OXI}
template_pmg.add_oxidation_state_by_element(oxi_map)

# =========================================================
# 6. EXPORT (Using CifWriter for better symmetry control)
# =========================================================
filename = f"{''.join(CATIONS)}_unrelaxed_icet_SQS_{SUPERCELL_DIM}.cif"
output_path = os.path.join(output_folder, filename)

# Initialize the writer with a symmetry precision of 0.1
# This helps software recognize the rocksalt space group
writer = CifWriter(template_pmg, symprec=0.1) 
writer.write_file(output_path)

print(f"--- SUCCESS: {filename} generated using CifWriter ---")