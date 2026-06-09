import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import pandas as pd

# 1. Load the trajectory
# index=':' reads every single frame
traj = read('MD_700C_PdCuCoNiCr.traj', index=':')

# 2. Get initial positions (The reference point)
# we use 'wrap=False' or handle displacements manually to avoid PBC jumps
initial_pos = traj[0].get_positions()
symbols = np.array(traj[0].get_chemical_symbols())
elements = ["Pd", "Cu", "Co", "Ni", "Cr"]

# To store results
msd_results = []
time_ps = []
timestep_fs = 2.0  # Make sure this matches your MD script

print("Calculating MSD for each element...")

# 3. Loop through frames
for i, atoms in enumerate(traj):
    # This is the "magic" - we need to account for atoms crossing the boundary
    # In ASE, for long simulations, we usually track 'displacements'
    if i == 0:
        displacements = np.zeros_like(initial_pos)
        prev_pos = initial_pos
    else:
        current_pos = atoms.get_positions()
        # Find the shortest distance between prev and current (handles PBC)
        diff = current_pos - prev_pos
        diff = atoms.get_velocities() * (timestep_fs) # Or use find_mic (Minimum Image Convention)
        
        # Simpler way for stable crystals:
        # Just use the unwrapped positions if the MD engine provided them
        diff = atoms.get_positions() - initial_pos
        
    # Calculate MSD for each element
    step_data = {"Time_ps": (i * 100 * timestep_fs) / 1000.0} # 10 is your LOG_INTERVAL
    for elem in elements:
        indices = np.where(symbols == elem)[0]
        # Square of the distance moved
        sq_dist = np.sum((atoms.get_positions()[indices] - initial_pos[indices])**2, axis=1)
        step_data[f"{elem}_MSD"] = np.mean(sq_dist)
    
    msd_results.append(step_data)

# 4. Plotting
df = pd.DataFrame(msd_results)
plt.figure(figsize=(8, 6))
for elem in elements:
    plt.plot(df["Time_ps"], df[f"{elem}_MSD"], label=elem)

plt.title("Mean Square Displacement (700°C)")
plt.xlabel("Time (ps)")
plt.ylabel(r"MSD ($\AA^2$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("MSD_Plot.png", dpi=300)
print("Done! Check MSD_Plot.png")