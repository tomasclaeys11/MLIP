import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# =============================================================================
#  1. USER INPUTS — Edit this block only
# =============================================================================
CRYSTAL_STRUCTURE = "perovskite"  # Options: "rocksalt" | "perovskite" | "spinel"

ANION = "O"
ANION_OXI = -2

ROCKSALT_CONFIG = {
    "cations": ["Pd", "Cu", "Co", "Ni", "Cr"],
    "cat_oxi": {"Pd": 2, "Cu": 2, "Co": 2, "Ni": 2, "Cr": 3},
    "a": 4.21,
    "supercell": (3, 3, 3), #good fro 216 atoms,  unit cell = 8 atoms
}

PEROVSKITE_CONFIG = {
    "A_cations": ["Pd"],
    "B_cations": ["Cu", "Co", "Ni", "Cr"],
    "cat_oxi": {"Pd": 2, "Cu": 2, "Co": 2, "Ni": 2, "Cr": 3},
    "a": 3.90,
    "supercell": (3, 3, 5), #gives 225 atoms, unit cell= 5 atoms
}

SPINEL_CONFIG = {
    "A_cations": ["Fe"],
    "B_cations": ["Cu", "Co", "Ni", "Cr"],
    "cat_oxi": {"Fe": 2, "Cu": 2, "Co": 2, "Ni": 2, "Cr": 3},
    "a": 8.15,
    "supercell": (2, 2, 1), #221, unit cell= 56 atoms
}

# =============================================================================
#  2. DIRECTORY SETUP
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
output_folder = os.path.join(current_dir, "start_CIFs")
os.makedirs(output_folder, exist_ok=True)

# =============================================================================
#  3. SYMMETRY-SAFE TEMPLATE BUILDERS
# =============================================================================

def build_rocksalt(cfg: dict) -> tuple[Structure, dict]:
    """Generates rocksalt template via Spacegroup 225 (Fm-3m)"""
    lat = Lattice.cubic(cfg["a"])
    # Cation on 4a (0,0,0), Anion on 4b (0.5,0.5,0.5)
    struct = Structure.from_spacegroup("Fm-3m", lat, ["X", ANION], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct.make_supercell(cfg["supercell"])
    site_labels = {i: "cation" for i, s in enumerate(struct) if s.specie.symbol == "X"}
    return struct, site_labels

def build_perovskite(cfg: dict) -> tuple[Structure, dict]:
    """Generates perovskite template via Spacegroup 221 (Pm-3m)"""
    lat = Lattice.cubic(cfg["a"])
    # A on 1a (0,0,0) | B on 1b (0.5,0.5,0.5) | O on 3c (0.5,0.5,0)
    struct = Structure.from_spacegroup("Pm-3m", lat, ["Xe", "Kr", ANION], 
                                       [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])
    struct.make_supercell(cfg["supercell"])
    site_labels = {}
    for i, s in enumerate(struct):
        if s.specie.symbol == "Xe": site_labels[i] = "A"
        elif s.specie.symbol == "Kr": site_labels[i] = "B"
    return struct, site_labels

def build_spinel(cfg: dict) -> tuple[Structure, dict]:
    """Generates standard spinel template via Spacegroup 227 (Fd-3m: Origin Choice 2)"""
    lat = Lattice.cubic(cfg["a"])
    # A on 8a (0.125,0.125,0.125) | B on 16d (0.5,0.5,0.5) | O on 32e (0.2548, 0.2548, 0.2548)
    struct = Structure.from_spacegroup("Fd-3m", lat, ["Xe", "Kr", ANION], 
                                       [[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0.2548, 0.2548, 0.2548]])
    struct.make_supercell(cfg["supercell"])
    site_labels = {}
    for i, s in enumerate(struct):
        if s.specie.symbol == "Xe": site_labels[i] = "A"
        elif s.specie.symbol == "Kr": site_labels[i] = "B"
    return struct, site_labels

# =============================================================================
#  4. CATION SHUFFLER
# =============================================================================

def fill_cations(struct: Structure, site_labels: dict, cfg: dict, structure_type: str) -> Structure:
    if structure_type == "rocksalt":
        cations = cfg["cations"]
        indices = list(site_labels.keys())
        repeats = int(np.ceil(len(indices) / len(cations)))
        pool = (cations * repeats)[:len(indices)]
        np.random.shuffle(pool)
        for idx, elem in zip(indices, pool):
            struct.replace(idx, elem)
    else:
        for site_type, list_key in [("A", "A_cations"), ("B", "B_cations")]:
            indices = [i for i, lbl in site_labels.items() if lbl == site_type]
            cations = cfg[list_key]
            repeats = int(np.ceil(len(indices) / len(cations)))
            pool = (cations * repeats)[:len(indices)]
            np.random.shuffle(pool)
            for idx, elem in zip(indices, pool):
                struct.replace(idx, elem)
    return struct

# =============================================================================
#  5. ROBUST WARREN-COWLEY SRO
# =============================================================================

def warren_cowley_sro(struct: Structure, site_labels: dict, cfg: dict, structure_type: str, cutoff: float) -> None:
    def _get_cation_list(site_type: str) -> list:
        return cfg["cations"] if structure_type == "rocksalt" else cfg[f"{site_type}_cations"]

    site_types = sorted(set(site_labels.values()))

    for site_type in site_types:
        indices = [i for i, lbl in site_labels.items() if lbl == site_type]
        cations = _get_cation_list(site_type)
        if len(cations) < 2:
            print(f"\n  [SRO] Site {site_type}: Single cation species present - skipping analysis.")
            continue

        sro_acc = {tuple(sorted((a, b))): [] for a in cations for b in cations}

        for idx in indices:
            site = struct[idx]
            elem_a = site.specie.symbol
            neighbors = struct.get_neighbors(site, r=cutoff)
            
            same_site_neighbors = [
                n for n in neighbors
                if n.specie.symbol in cations          # is a cation of this site
                and site_labels.get(n.index, None) == site_type  # <-- NATIVE INDEX
            ]

            if not same_site_neighbors:
                continue

            for elem_b in cations:
                # Dynamically evaluate real macro-concentration of element B on this sublattice
                count_b_total = sum(1 for i in indices if struct[i].specie.symbol == elem_b)
                true_conc_b = count_b_total / len(indices)
                
                if true_conc_b == 0: continue

                count_b_local = sum(1 for n in same_site_neighbors if n.specie.symbol == elem_b)
                p_ij = count_b_local / len(same_site_neighbors)
                alpha = 1.0 - (p_ij / true_conc_b)
                sro_acc[tuple(sorted((elem_a, elem_b)))].append(alpha)

        header = f"Sublattice {site_type} (Cutoff = {cutoff} A, {len(indices)} sites)"
        print(f"\n--- Warren-Cowley SRO Evaluation: {header} ---")
        print(f"  {'Atom Pair':<14} | {'Mean a_ij':>12} | {'Samples':>10}")
        print("  " + "-" * 44)
        for pair_key in sorted(sro_acc):
            vals = sro_acc[pair_key]
            if vals:
                print(f"  {str(pair_key):<14} | {np.mean(vals):>12.4f} | {len(vals):>10}")
            else:
                print(f"  {str(pair_key):<14} | {'N/A':>12} | {'0':>10}")

# =============================================================================
#  6. MAIN WORKFLOW
# =============================================================================

def run(structure_type: str) -> None:
    structure_type = structure_type.lower()
    cfg_map = {"rocksalt": ROCKSALT_CONFIG, "perovskite": PEROVSKITE_CONFIG, "spinel": SPINEL_CONFIG}
    cfg = cfg_map[structure_type]

    print(f"\n{'='*60}\n  HEO Target: {structure_type.upper()} | Supercell: {cfg['supercell']}\n{'='*60}")

    builder = {"rocksalt": build_rocksalt, "perovskite": build_perovskite, "spinel": build_spinel}[structure_type]
    struct, site_labels = builder(cfg)

    struct = fill_cations(struct, site_labels, cfg, structure_type)

    sro_cutoff = {"rocksalt": 3.5, "perovskite": 4.2, "spinel": 4.0}[structure_type]
    warren_cowley_sro(struct, site_labels, cfg, structure_type, cutoff=sro_cutoff)

    print("\n  Enforcing formal oxidation bounds...")
    oxi_map = {**cfg["cat_oxi"], ANION: ANION_OXI}
    struct.add_oxidation_state_by_element(oxi_map)

    cation_str = "".join(cfg["cations"]) if structure_type == "rocksalt" else "".join(cfg["A_cations"]) + "".join(cfg["B_cations"])
    filename = f"{cation_str}O_unrelaxed_{structure_type}_{'x'.join(str(d) for d in cfg['supercell'])}.cif"    
    output_path = os.path.join(output_folder, filename)

    # CRITICAL FIX: symprec=None protects against accidental symmetric coordinate compression
    writer = CifWriter(struct, symprec=None)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(writer))

    print(f"  Success. Explicit P1 CIF generated at: {output_path}\n")

if __name__ == "__main__":
    run(CRYSTAL_STRUCTURE)