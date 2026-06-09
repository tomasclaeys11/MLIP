from ase.io import read
import pandas as pd
import numpy as np

# 1. Load the binary trajectory (The true backup)
traj_file = 'MD_700C_PdCuCoNiCr (1).traj'
print(f"Reading {traj_file}...")
frames = read(traj_file, index=':')
print(f"Success! Found {len(frames)} frames of data.")

recovered_data = []

# 2. Extract the data from every frame
for i, atoms in enumerate(frames):
    # Calculate time/step (based on your LOG_INTERVAL of 100 and TIMESTEP 2.0)
    step = i * 100 
    
    # Extract MagMoms (CHGNet saves these in the traj results)
    moms = np.abs(atoms.get_magnetic_moments())
    symbols = np.array(atoms.get_chemical_symbols())
    
    step_data = {
        "Step": step,
        "Time_ps": (step * 2.0) / 1000.0,
        "Temp_K": atoms.get_temperature(),
        "Volume_A3": atoms.get_volume()
    }
    
    # Average per cation
    for elem in ["Pd", "Cu", "Co", "Ni", "Cr"]:
        idx = np.where(symbols == elem)[0]
        if len(idx) > 0:
            step_data[f"{elem}_MagMom_Avg"] = np.mean(moms[idx])
            step_data[f"{elem}_MagMom_Std"] = np.std(moms[idx])
            
    recovered_data.append(step_data)

# 3. Save the RECOVERED file
df_recovered = pd.DataFrame(recovered_data)
df_recovered.to_csv("RECOVERED_MD_700C_MagMoms.csv", index=False)
print("--- RECOVERY COMPLETE: Check RECOVERED_MD_700C_MagMoms.csv ---")