"""
NPT Molecular Dynamics — High-Entropy Oxide (HEO) at 500 °C for 1 ns
CHGNet + ASE | Nose-Hoover thermostat/barostat
 
Run parameters
--------------
Temperature  : 773.15 K  (500 °C)
Pressure     : 1 atm  (1.01325e-4 GPa)
Timestep     : 2 fs
Total steps  : 500 000  (= 1 ns)
Ensemble     : NPT, Nose-Hoover  (ttime = 200 fs, pfactor from EOS bulk modulus)
 
Workflow
--------
1. Relax the input structure to < 0.05 eV/Å  [skipped on resume].
2. Compute bulk modulus via Birch-Murnaghan EOS  [cached to disk].
3. Temperature ramp: NVT, 300 K → TARGET_TEMP_K in steps  [skipped on resume].
4. Production NPT run with robust checkpoint / resume logic.
5. Observer logs every LOG_INTERVAL steps:
     - Temperature, volume, Epot, Ekin, Etot
     - Per-element |magnetic moment| mean & std
     - Per-element & total Mean Square Displacement (MSD)
     - MSD is computed from the PRODUCTION reference frame,
       carried across restarts via a saved reference positions file.
 
Temperature ramp design
-----------------------
The NVT equilibration block is replaced by a staged linear ramp:
 
  RAMP_START_K  →  TARGET_TEMP_K   in  RAMP_N_STEPS  increments
  each increment runs for  RAMP_SEGMENT_STEPS  steps
 
At each stage the Nose-Hoover thermostat temperature is updated by
calling  dyn.set_temperature(Temperature=T_K * units.kB)  on the
underlying ASE Langevin/NHC dynamics object before the next segment
runs.  This is the canonical ASE approach — it modifies the target
temperature *in place* without rebuilding the integrator, so momenta
accumulated from the previous stage carry over naturally.
 
Default ramp: 300 K → 773 K in 10 steps of 0.5 ps each = 5 ps total.
Paper recommendation: "at least 1 ps".  5 ps is conservative and safe.
 
Checkpoint / resume
-------------------
Every segment can be interrupted and restarted:
  - Relaxation    : output CIF detected → skipped.
  - EOS           : output PKL detected → skipped.
  - Ramp          : output TRAJ detected → last frame + last T reloaded.
  - Production    : step count read from CSV; MSD reference positions
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
 
from ase import units
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
 
INPUT_CIF     = "MD_PdCuCoNiCr_perovskite.cif"
OUTPUT_PREFIX = "MD_Tramp_HEO_404_perovskite"
 
TARGET_TEMP_K = 973.15        # 700 °C in Kelvin
PRESSURE_GPa  = 1.01325e-4   # 1 atm in GPa  (CHGNet/ASE convention)
TIMESTEP_FS   = 2.0           # 2 fs — safe for oxide MD with CHGNet
 
# Nose-Hoover coupling times
TAUT_FS       = 100  * TIMESTEP_FS   # 200 fs  thermostat damping
TAUP_FS       = 1000 * TIMESTEP_FS   # 2000 fs barostat damping
 
TOTAL_STEPS   = 500_000   # 500 000 × 2 fs = 1 ns
LOG_INTERVAL  =     100   # every 100 steps = every 0.2 ps
 
# ------------------------------------------------------------------
#  Temperature ramp settings
#  ------------------------------------------------------------------
#  The ramp replaces the old single-shot NVT equilibration block.
#
#  RAMP_START_K       : initial temperature for the ramp (K).
#                       300 K is a safe room-temperature starting point.
#                       Reduce to 100 K if you want to minimise early
#                       kinetic energy spikes in stiff perovskite cells.
#
#  RAMP_N_STEPS       : number of discrete temperature steps.
#                       e.g. 10 steps from 300 → 773 K = ~47 K per step.
#
#  RAMP_SEGMENT_STEPS : MD steps run at each temperature plateau.
#                       250 steps × 2 fs = 0.5 ps per plateau.
#                       Total ramp time = RAMP_N_STEPS × RAMP_SEGMENT_STEPS
#                       × TIMESTEP_FS  =  10 × 250 × 2 fs = 5 ps.
#                       The paper asks for "at least 1 ps"; 5 ps is safe.
#
#  If you want a finer ramp (e.g. 1 K/step), just increase RAMP_N_STEPS
#  and reduce RAMP_SEGMENT_STEPS proportionally to keep the total time
#  the same.
RAMP_START_K       = 300
RAMP_N_STEPS       = 10     # number of temperature plateaux
RAMP_SEGMENT_STEPS = 250    # steps per plateau  (0.5 ps each at 2 fs)
 
# File paths
RELAX_CIF    = f"MD_PdCuCoNiCr_perovskite.cif"
RAMP_TRAJ    = f"{OUTPUT_PREFIX}_ramp.traj"    # replaces _equil.traj
RAMP_LOG     = f"{OUTPUT_PREFIX}_ramp.log"
RAMP_META    = f"{OUTPUT_PREFIX}_ramp_meta.pkl"  # stores last completed T
PROD_TRAJ    = f"{OUTPUT_PREFIX}_prod.traj"
PROD_LOG     = f"{OUTPUT_PREFIX}_prod.log"
OBS_CSV      = f"{OUTPUT_PREFIX}_observables.csv"
EOS_PKL      = f"{OUTPUT_PREFIX}_EOS.pkl"
MSD_REF_NPY  = f"{OUTPUT_PREFIX}_MSD_ref_positions.npy"
 
 
# ============================================================
#  2.  MODEL
# ============================================================
 
print("Loading CHGNet model...")
model = CHGNet.load()
model.graph_converter = CrystalGraphConverter(
    atom_graph_cutoff=5,
    bond_graph_cutoff=3,
    algorithm="fast",
)
print(f"CHGNet v{model.version} loaded with fast graph converter.")
 
 
# ============================================================
#  3.  STRUCTURE RELAXATION
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
#  5.  NVT TEMPERATURE RAMP  (replaces single-shot equilibration)
#
#  Strategy
#  --------
#  Build one MolecularDynamics object at RAMP_START_K.
#  Then loop over the temperature schedule, calling
#    dyn.set_temperature(Temperature=T * units.kB)
#  between each RAMP_SEGMENT_STEPS-step segment.
#
#  set_temperature() is an ASE method on the underlying integrator
#  (NVTBerendsen or Langevin — CHGNet wraps ASE's NVT_NoseHoover).
#  It changes the *target* temperature in place without reinitialising
#  velocities, so the kinetic energy accumulated in the previous
#  plateau carries over — this is the correct physical behaviour.
#
#  Checkpoint logic
#  ----------------
#  RAMP_META stores the index of the last *completed* ramp step.
#  On resume we reload the last frame from RAMP_TRAJ and skip all
#  completed steps, continuing from where we left off.
#  If the ramp is fully complete (meta says so), we skip entirely.
# ============================================================
 
# Build the temperature schedule: N evenly spaced values from
# RAMP_START_K up to and including TARGET_TEMP_K.
RAMP_N_STEPS = 2000                  # Number of small temperature increments
STEPS_PER_INCREMENT = TOTAL_STEPS // RAMP_N_STEPS  # 500,000 / 2000 = 250 steps per increment

# Build the complete continuous temperature array across the entire run
ramp_temperatures = np.linspace(RAMP_START_K, TARGET_TEMP_K, RAMP_N_STEPS)

# Setup Checkpoint/Resume Tracker for the continuous run
if os.path.exists(OBS_CSV):
    df_existing = pd.read_csv(OBS_CSV)
    steps_done = int(df_existing["Step"].max()) if len(df_existing) > 0 else 0
else:
    steps_done = 0

current_increment_idx = steps_done // STEPS_PER_INCREMENT
remaining_steps = TOTAL_STEPS - steps_done

if remaining_steps <= 0:
    print("Continuous ramping run already complete. Exiting.")
    sys.exit(0)

# Load the correct starting structure frame based on progress
if steps_done > 0 and os.path.exists(PROD_TRAJ):
    print(f"Resuming continuous ramp from step {steps_done} (T = {ramp_temperatures[current_increment_idx]:.2f} K)...")
    atoms = read(PROD_TRAJ, index=-1)
    atoms.calc = CHGNetCalculator(model=model)
    traj_mode = "a"
else:
    print(f"Starting fresh continuous NPT ramp from {RAMP_START_K} K to {TARGET_TEMP_K} K...")
    atoms = struct_relaxed
    atoms.calc = CHGNetCalculator(model=model)
    traj_mode = "w"

# ============================================================
#  6.  INITIALIZE MOLECULAR DYNAMICS OBJECT
# ============================================================

# Initialize at the current temperature milestone
T_start = ramp_temperatures[current_increment_idx]

prod_md = MolecularDynamics(
    atoms=atoms,
    model=model,
    ensemble="npt",             # Using NPT so the box expands as it heats up!
    thermostat="Nose-Hoover",
    temperature=int(T_start),
    timestep=TIMESTEP_FS,
    pressure=PRESSURE_GPa,
    taut=TAUT_FS,
    taup=TAUP_FS,
    bulk_modulus=bulk_modulus_GPa,
    trajectory=PROD_TRAJ,
    logfile=PROD_LOG,
    loginterval=LOG_INTERVAL,
)

prod_dyn = prod_md.dyn

# Fix trajectory handler for resume appending
if traj_mode == "a":
    for i, obs in enumerate(prod_dyn.observers):
        if hasattr(obs[0], "write"):
            append_traj = Trajectory(PROD_TRAJ, mode="a", atoms=atoms)
            prod_dyn.observers[i] = (append_traj, obs[1], obs[2], obs[3])
            break

# ============================================================
#  7.  MSD REFERENCE POSITIONS & OBSERVERS
# ============================================================

if hasattr(atoms, "to_ase_atoms"):
    atoms = atoms.to_ase_atoms()
elif not hasattr(atoms, "get_positions"):
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(atoms)

if os.path.exists(MSD_REF_NPY):
    ref_positions_cart = np.load(MSD_REF_NPY)
    print(f"Loaded MSD reference positions from {MSD_REF_NPY}")
else:
    ref_positions_cart = atoms.get_positions(wrap=True)
    np.save(MSD_REF_NPY, ref_positions_cart)
    print(f"Saved MSD reference positions → {MSD_REF_NPY}")

symbols = np.array(atoms.get_chemical_symbols())
unique_elements = sorted(set(symbols))
element_indices = {el: np.where(symbols == el)[0] for el in unique_elements}
log_buffer: list[dict] = []
csv_needs_header = not os.path.exists(OBS_CSV)

# (Keep your _compute_msd, _atomic_write_csv, and log_observables functions exactly as they are)
def _compute_msd(current_positions_cart: np.ndarray) -> dict[str, float]:
    disp = current_positions_cart - ref_positions_cart
    cell_lengths = np.linalg.norm(atoms.get_cell(), axis=1)
    for i in range(3):
        disp[:, i] -= np.round(disp[:, i] / cell_lengths[i]) * cell_lengths[i]
    sq_disp = np.sum(disp ** 2, axis=1)
    msd_dict: dict[str, float] = {}
    for el in unique_elements:
        idx = element_indices[el]
        msd_dict[f"MSD_{el}_A2"] = round(float(np.mean(sq_disp[idx])), 6)
    msd_dict["MSD_total_A2"] = round(float(np.mean(sq_disp)), 6)
    return msd_dict

def _atomic_write_csv(rows: list[dict]) -> None:
    global csv_needs_header
    df = pd.DataFrame(rows)
    dir_ = os.path.dirname(OBS_CSV) or "."
    with tempfile.NamedTemporaryFile(mode="w", dir=dir_, suffix=".tmp", delete=False) as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp_path, mode="w", index=False, header=csv_needs_header)
    os.replace(tmp_path, OBS_CSV if csv_needs_header else OBS_CSV + ".part")
    if csv_needs_header: csv_needs_header = False
    else:
        with open(OBS_CSV, "a") as main, open(OBS_CSV + ".part") as part: main.write(part.read())
        os.remove(OBS_CSV + ".part")

def log_observables():
    current_step = prod_dyn.get_number_of_steps() + steps_done
    temp  = atoms.get_temperature()
    vol   = atoms.get_volume()
    epot  = atoms.get_potential_energy()
    ekin  = atoms.get_kinetic_energy()
    row: dict = {
        "Step":      current_step,
        "Time_ps":   round(current_step * TIMESTEP_FS / 1000.0, 4),
        "Temp_K":    round(temp,  3),
        "Volume_A3": round(vol,  4),
        "Epot_eV":   round(epot,  6),
        "Ekin_eV":   round(ekin,  6),
        "Etot_eV":   round(epot + ekin, 6),
    }
    moms = np.abs(atoms.get_magnetic_moments())
    for el in unique_elements:
        idx = element_indices[el]
        row[f"MagMom_{el}_avg"] = round(float(np.mean(moms[idx])), 6)
        row[f"MagMom_{el}_std"] = round(float(np.std(moms[idx])),  6)
    row.update(_compute_msd(atoms.get_positions(wrap=True)))
    log_buffer.append(row)
    if len(log_buffer) >= 50:
        _atomic_write_csv(log_buffer)
        log_buffer.clear()
    if current_step % 10_000 == 0:
        msd_tot = row["MSD_total_A2"]
        print(f"  Step {current_step:>7d}/{TOTAL_STEPS} | T_real = {temp:7.1f} K | V = {vol:9.3f} Å³ | MSD = {msd_tot:.4f} Å²")

prod_dyn.attach(log_observables, interval=LOG_INTERVAL)

# ============================================================
# 10.  RUN THE DYNAMIC RAMP LOOP
# ============================================================
print(f"Executing continuous dynamic NPT ramp across remaining chunks...")

try:
    for idx in range(current_increment_idx, RAMP_N_STEPS):
        T_next = ramp_temperatures[idx]
        
        # Dynamically push the target temperature of the thermostat up in eV
        prod_dyn.set_temperature(temperature=T_next * units.kB)
        
        # Run this slice segment
        prod_md.run(STEPS_PER_INCREMENT)
finally:
    if log_buffer:
        _atomic_write_csv(log_buffer)
        log_buffer.clear()
    print("\nSimulation slice loop ended cleanly.")