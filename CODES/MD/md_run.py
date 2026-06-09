"""
NPT Molecular Dynamics — High-Entropy Oxide (HEO) at 700 °C for 1 ns
CHGNet + ASE | Nose-Hoover thermostat/barostat

Run parameters
--------------
Temperature  : 973.15 K  (700 °C)
Pressure     : 1 atm  (1.01325e-4 GPa)
Timestep     : 2 fs
Total steps  : 500 000  (= 1 ns)
Ensemble     : NPT, Nose-Hoover  (ttime = 200 fs, pfactor from EOS bulk modulus)

Workflow
--------
1. Relax the input structure to < 0.05 eV/Å  [skipped on resume].
2. Compute bulk modulus via Birch-Murnaghan EOS  [cached to disk].
3. Equilibrate at 973 K for 50 ps — NVT  [skipped on resume].
4. Production NPT run with robust checkpoint / resume logic.
5. Observer logs every LOG_INTERVAL steps:
     - Temperature, volume, Epot, Ekin, Etot
     - Per-element |magnetic moment| mean & std
     - Per-element & total Mean Square Displacement (MSD)
     - MSD is computed from the PRODUCTION reference frame,
       carried across restarts via a saved reference positions file.

Notes on Bader charges
----------------------
Bader charges require a full volumetric electron density (VASP CHGCAR).
CHGNet is an MLP and does not produce one — magnetic moments are its
charge-state proxy.  If you need Bader charges, extract snapshots from
the trajectory and run single-point VASP on them post hoc.

Checkpoint / resume
-------------------
Every segment can be interrupted and restarted:
  - Relaxation  : output CIF detected → skipped.
  - EOS         : output PKL detected → skipped.
  - Equilibration : output TRAJ detected → last frame reloaded.
  - Production  : step count read from CSV; MSD reference positions
                  reloaded from NPY file; trajectory appended.
  - The CSV is written atomically (tmp file + rename) so a crash
    during a flush never corrupts existing data.
"""

from __future__ import annotations

import os
import sys
import warnings
import pickle
import tempfile
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

from ase.io import read
from ase.io.trajectory import Trajectory
from pymatgen.core import Structure

from chgnet.model.model import CHGNet
from chgnet.model.dynamics import (
    CHGNetCalculator,
    EquationOfState,
    MolecularDynamics,
    StructOptimizer,
)
from chgnet.graph import CrystalGraphConverter


# ============================================================
#  1.  PARAMETERS
# ============================================================

INPUT_CIF     = "/scratch/gent/514/vsc51474/tests_hpc/MD_PdCuCoNiCr_start.cif"
OUTPUT_PREFIX = "MD_500C_HEO_404"

TARGET_TEMP_K = 773.15        # 700 °C in Kelvin
PRESSURE_GPa  = 1.01325e-4   # 1 atm in GPa  (CHGNet/ASE convention)
TIMESTEP_FS   = 2.0           # 2 fs — safe for oxide MD with CHGNet

# Nose-Hoover coupling times
# taut  = 100 × timestep = 200 fs   (standard thermostat damping)
# taup  = 1000 × timestep = 2000 fs (sluggish barostat → no cell oscillations)
TAUT_FS       = 100  * TIMESTEP_FS   # 200 fs
TAUP_FS       = 1000 * TIMESTEP_FS   # 2000 fs

TOTAL_STEPS   = 500_000   # 500 000 × 2 fs = 1 ns
EQUIL_STEPS   =  25_000   # 50 ps NVT pre-equilibration
LOG_INTERVAL  =     100   # every 100 steps = every 0.2 ps

# File paths
RELAX_CIF     = f"MD_700C_HEO_relaxed.cif"
EQUIL_TRAJ    = f"{OUTPUT_PREFIX}_equil.traj"
PROD_TRAJ     = f"{OUTPUT_PREFIX}_prod.traj"
PROD_LOG      = f"{OUTPUT_PREFIX}_prod.log"
OBS_CSV       = f"{OUTPUT_PREFIX}_observables.csv"   # magmoms + MSD + thermo
EOS_PKL       = f"{OUTPUT_PREFIX}_EOS.pkl"
MSD_REF_NPY   = f"{OUTPUT_PREFIX}_MSD_ref_positions.npy"  # reference frame for MSD


# ============================================================
#  2.  MODEL — load once, attach fast graph converter
# ============================================================

print("Loading CHGNet model...")
model = CHGNet.load()

# "fast" uses a C++ backend — significant speedup for large supercells.
# Cutoffs are CHGNet defaults.
model.graph_converter = CrystalGraphConverter(
    atom_graph_cutoff=5,
    bond_graph_cutoff=3,
    algorithm="fast",
)
print(f"CHGNet v{model.version} loaded with fast graph converter.")


# ============================================================
#  3.  STRUCTURE RELAXATION
#      Relax once so initial forces are small when the thermostat starts.
# ============================================================

if os.path.exists(RELAX_CIF):
    print(f"Found relaxed structure: {RELAX_CIF} — skipping relaxation.")
    struct_relaxed = Structure.from_file(RELAX_CIF)
else:
    print("Relaxing input structure (fmax < 0.05 eV/Å)...")
    struct_input = Structure.from_file(INPUT_CIF)
    relaxer = StructOptimizer(model=model)
    result = relaxer.relax(struct_input, fmax=0.05, steps=500, relax_cell=True)
    struct_relaxed = result["final_structure"]
    struct_relaxed.to(filename=RELAX_CIF)
    print(f"Relaxation done → {RELAX_CIF}")


# ============================================================
#  4.  BULK MODULUS via Birch-Murnaghan EOS
#      Required for the NPT Nose-Hoover pfactor.
#      Cached to disk so it is never recomputed on resume.
# ============================================================

if os.path.exists(EOS_PKL):
    with open(EOS_PKL, "rb") as fh:
        eos_data = pickle.load(fh)
    bulk_modulus_GPa = eos_data["bulk_modulus_GPa"]
    print(f"Loaded cached bulk modulus: {bulk_modulus_GPa:.3f} GPa")
else:
    print("Computing bulk modulus via Birch-Murnaghan EOS...")
    eos = EquationOfState(model=CHGNetCalculator(model=model))
    eos.fit(atoms=struct_relaxed, steps=500, fmax=0.1, verbose=True)
    bulk_modulus_GPa = eos.get_bulk_modulus(unit="GPa")
    print(f"Bulk modulus = {bulk_modulus_GPa:.3f} GPa")
    with open(EOS_PKL, "wb") as fh:
        pickle.dump({"bulk_modulus_GPa": bulk_modulus_GPa}, fh)


# ============================================================
#  5.  NVT EQUILIBRATION  (50 ps)
#      Thermalises the system before the barostat is switched on.
#      Avoids unphysical cell-volume spikes in the first NPT steps.
# ============================================================

if os.path.exists(EQUIL_TRAJ):
    print(f"Found equilibration trajectory: {EQUIL_TRAJ} — reading last frame.")
    atoms = read(EQUIL_TRAJ, index=-1)
    atoms.calc = CHGNetCalculator(model=model)
else:
    print(f"NVT equilibration: {EQUIL_STEPS} steps at {TARGET_TEMP_K:.2f} K...")
    md_equil = MolecularDynamics(
        atoms=struct_relaxed,
        model=model,
        ensemble="nvt",
        thermostat="Nose-Hoover",
        temperature=int(TARGET_TEMP_K),
        timestep=TIMESTEP_FS,
        taut=TAUT_FS,
        trajectory=EQUIL_TRAJ,
        logfile=f"{OUTPUT_PREFIX}_equil.log",
        loginterval=LOG_INTERVAL,
        starting_temperature=int(TARGET_TEMP_K),  # explicit Maxwell-Boltzmann init
    )
    md_equil.run(EQUIL_STEPS)
    print("Equilibration complete.")
    atoms = md_equil.atoms


# ============================================================
#  6.  CHECKPOINT LOGIC — how many production steps are done?
# ============================================================

if os.path.exists(OBS_CSV):
    df_existing = pd.read_csv(OBS_CSV)
    steps_done = int(df_existing["Step"].max()) if len(df_existing) > 0 else 0
else:
    steps_done = 0

remaining_steps = TOTAL_STEPS - steps_done

if remaining_steps <= 0:
    print("Production run already complete. Exiting.")
    sys.exit(0)

if steps_done > 0 and os.path.exists(PROD_TRAJ):
    print(
        f"Resuming from step {steps_done} "
        f"({steps_done * TIMESTEP_FS / 1e3:.3f} ns / "
        f"{TOTAL_STEPS * TIMESTEP_FS / 1e3:.3f} ns)..."
    )
    atoms = read(PROD_TRAJ, index=-1)
    atoms.calc = CHGNetCalculator(model=model)
    traj_append = True
else:
    print("Starting fresh production NPT run...")
    traj_append = False

print(f"{remaining_steps} steps remaining ({remaining_steps * TIMESTEP_FS / 1e3:.3f} ns).")


# ============================================================
#  7.  MSD REFERENCE POSITIONS
#
#      MSD = < |r(t) - r_ref|² >  averaged over all atoms of a species.
#
#      r_ref is the position vector at the *start of the production run*
#      (step 0 of production, not step 0 of a resumed segment).
#      This is saved to disk so that resumed segments compute MSD relative
#      to the same origin, giving a continuous MSD curve across restarts.
#
#      Positions are stored in fractional (scaled) coordinates so that
#      cell-shape changes under NPT do not contaminate the displacement.
#      Displacement is then converted to Cartesian Å for the MSD value.
#
#      NOTE: MSD computed this way is the "absolute" MSD from t=0.
#      For a diffusion coefficient you would use the Einstein relation:
#        D = MSD / (6 * t)   [3D; use 2 or 4 for 2D/1D]
#      This is valid once the system is in the diffusive regime (linear MSD).
# ============================================================

if os.path.exists(MSD_REF_NPY):
    ref_positions_cart = np.load(MSD_REF_NPY)   # shape (N_atoms, 3)  in Å
    print(f"Loaded MSD reference positions from {MSD_REF_NPY}")
else:
    # First production run — save current positions as reference
    ref_positions_cart = atoms.get_positions(wrap=True)
    np.save(MSD_REF_NPY, ref_positions_cart)
    print(f"Saved MSD reference positions → {MSD_REF_NPY}")


# ============================================================
#  8.  OBSERVER — thermo + magmoms + MSD
# ============================================================

symbols        = np.array(atoms.get_chemical_symbols())
unique_elements = sorted(set(symbols))
element_indices = {el: np.where(symbols == el)[0] for el in unique_elements}

log_buffer: list[dict] = []
csv_needs_header = not os.path.exists(OBS_CSV)


def _compute_msd(current_positions_cart: np.ndarray) -> dict[str, float]:
    """
    Compute per-element and total MSD in Å² from the reference frame.

    Minimum-image convention is applied so that atoms which have crossed
    a periodic boundary are handled correctly.  This requires the cell
    to be close to orthorhombic; for strongly sheared cells a full
    mic displacement would be needed (ase.geometry.find_mic).
    For a dense solid/liquid HEO at 1 atm the cell stays nearly cubic,
    so the wrapped-coordinate approach here is adequate.
    """
    # Displacement vectors in Cartesian Å
    disp = current_positions_cart - ref_positions_cart   # (N, 3)

    # Minimum-image correction using current cell
    cell = atoms.get_cell()
    # For orthorhombic-ish cells: wrap each displacement into [-L/2, L/2]
    cell_lengths = np.linalg.norm(cell, axis=1)
    for i in range(3):
        disp[:, i] -= np.round(disp[:, i] / cell_lengths[i]) * cell_lengths[i]

    sq_disp = np.sum(disp ** 2, axis=1)   # (N,)  in Å²

    msd_dict: dict[str, float] = {}
    for el in unique_elements:
        idx = element_indices[el]
        msd_dict[f"MSD_{el}_A2"] = round(float(np.mean(sq_disp[idx])), 6)
    msd_dict["MSD_total_A2"] = round(float(np.mean(sq_disp)), 6)
    return msd_dict


def _atomic_write_csv(rows: list[dict]) -> None:
    """
    Write rows to the CSV atomically via a temp file + rename.
    A crash during writing never leaves a truncated or corrupt CSV.
    """
    global csv_needs_header
    df = pd.DataFrame(rows)
    dir_ = os.path.dirname(OBS_CSV) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_, suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp_path, mode="w", index=False, header=csv_needs_header)
    # On POSIX (Linux HPC) os.replace is atomic
    os.replace(tmp_path, OBS_CSV if csv_needs_header else OBS_CSV + ".part")

    if csv_needs_header:
        # First write: the file IS the CSV
        csv_needs_header = False
    else:
        # Subsequent writes: append the part file to the main CSV
        with open(OBS_CSV, "a") as main, open(OBS_CSV + ".part") as part:
            main.write(part.read())
        os.remove(OBS_CSV + ".part")


def log_observables():
    """Observer called every LOG_INTERVAL steps."""
    current_step = prod_dyn.get_number_of_steps() + steps_done

    # --- thermodynamics ---
    temp  = atoms.get_temperature()
    vol   = atoms.get_volume()
    epot  = atoms.get_potential_energy()
    ekin  = atoms.get_kinetic_energy()

    row: dict = {
        "Step":      current_step,
        "Time_ps":   round(current_step * TIMESTEP_FS / 1000.0, 4),
        "Temp_K":    round(temp,  3),
        "Volume_A3": round(vol,   4),
        "Epot_eV":   round(epot,  6),
        "Ekin_eV":   round(ekin,  6),
        "Etot_eV":   round(epot + ekin, 6),
    }

    # --- magnetic moments ---
    moms = np.abs(atoms.get_magnetic_moments())
    for el in unique_elements:
        idx = element_indices[el]
        row[f"MagMom_{el}_avg"] = round(float(np.mean(moms[idx])), 6)
        row[f"MagMom_{el}_std"] = round(float(np.std(moms[idx])),  6)

    # --- MSD ---
    # get_positions(wrap=True) folds atoms back into the primary cell so
    # the minimum-image displacement in _compute_msd stays well-defined.
    row.update(_compute_msd(atoms.get_positions(wrap=True)))

    log_buffer.append(row)

    # Flush every 50 entries (every 5000 steps / 10 ps)
    if len(log_buffer) >= 50:
        _atomic_write_csv(log_buffer)
        log_buffer.clear()

    # Progress print every 10 000 steps (20 ps)
    if current_step % 10_000 == 0:
        msd_tot = row["MSD_total_A2"]
        print(
            f"  Step {current_step:>7d}/{TOTAL_STEPS} | "
            f"T = {temp:7.1f} K | "
            f"V = {vol:9.3f} Å³ | "
            f"Epot = {epot:.4f} eV | "
            f"MSD = {msd_tot:.4f} Å²"
        )


# ============================================================
#  9.  PRODUCTION DYNAMICS
# ============================================================
prod_md = MolecularDynamics(
    atoms=atoms,
    model=model,
    ensemble="npt",
    thermostat="Nose-Hoover",
    temperature=int(TARGET_TEMP_K),
    timestep=TIMESTEP_FS,
    pressure=PRESSURE_GPa,
    taut=TAUT_FS,
    taup=TAUP_FS,
    bulk_modulus=bulk_modulus_GPa,
    trajectory=PROD_TRAJ,          # <-- Keep this here so the wrapper handles restart state!
    logfile=PROD_LOG,
    loginterval=LOG_INTERVAL,
)

prod_dyn = prod_md.dyn

# 2. FIX THE TUPLE CRASH NATIVELY
# If it's an append/restart run, we fix the tuple structure without breaking the observer list
if traj_append:
    from ase.io.trajectory import Trajectory
    # Reconstruct the observer entry matching your ASE version's expected 4-element structure:
    # (function/instance, interval, args, kwargs)
    append_traj = Trajectory(PROD_TRAJ, mode='a', atoms=atoms)
    prod_dyn.observers[0] = (append_traj, LOG_INTERVAL, [], {})

# Attach your secondary text logger
prod_dyn.attach(log_observables, interval=LOG_INTERVAL)


# ============================================================
# 10.  RUN
# ============================================================

try:
    prod_md.run(remaining_steps)
finally:
    # Flush any remaining buffered rows — guaranteed even on SIGTERM/timeout
    if log_buffer:
        _atomic_write_csv(log_buffer)
        log_buffer.clear()

    final_step = steps_done + prod_dyn.get_number_of_steps()
    print(f"\nRun ended at step {final_step} / {TOTAL_STEPS} "
          f"({final_step * TIMESTEP_FS / 1e3:.4f} ns).")
    print(f"Observables (thermo + magmoms + MSD) : {OBS_CSV}")
    print(f"Trajectory                           : {PROD_TRAJ}")
    print(f"ASE thermo log                       : {PROD_LOG}")
    print(f"MSD reference positions              : {MSD_REF_NPY}")