import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SMART SETUP ---
path = os.path.dirname(os.path.abspath(__file__))

# Find files automatically
charge_files = [f for f in os.listdir(path) if f.startswith("Global_Charge") and f.endswith(".cif")]
dist_files = [f for f in os.listdir(path) if f.startswith("Global_Distortion") and f.endswith(".cif")]

if not charge_files or not dist_files:
    print(f"Can't find CIF files in: {path}")
    exit()

charge_cif = os.path.join(path, charge_files[0])
dist_cif = os.path.join(path, dist_files[0])

# --- 2. HARVEST DATA ---
def parse_custom_cif(filename):
    elements, values = [], []
    start_reading = False
    with open(filename, 'r') as f:
        for line in f:
            if "_atom_site_occupancy" in line:
                start_reading = True
                continue
            if start_reading and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    elements.append(parts[1]) 
                    values.append(float(parts[5])) 
    return elements, values

print(f"Reading: {charge_files[0]}")
elements, charges = parse_custom_cif(charge_cif)
_, distortions = parse_custom_cif(dist_cif)

df = pd.DataFrame({"Element": elements, "Charge": charges, "Distortion": distortions})
df = df[df["Element"] != "O"]

# --- 3. PLOT ---
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="Charge", y="Distortion", hue="Element", s=100, alpha=0.7)
plt.title(f"Bivariate Analysis: {os.path.basename(path)}")
plt.savefig(os.path.join(path, ".png"), dpi=300)
plt.show()