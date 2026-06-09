import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 1. SETUP ---
warnings.filterwarnings("ignore")
# The script stays at the same level as the 'Results_...' folders
root_path = os.path.dirname(os.path.abspath(__file__))

# Find all result folders
result_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and f.startswith("Results_")]

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

print(f"Found {len(result_folders)} folders to analyze.\n")

# --- 2. THE LOOP ---
for folder in result_folders:
    folder_path = os.path.join(root_path, folder)
    print(f"Processing: {folder}...")

    # Look for the Charge and Distortion CIFs inside this specific folder
    charge_files = [f for f in os.listdir(folder_path) if f.startswith("Global_Charge") and f.endswith(".cif")]
    dist_files = [f for f in os.listdir(folder_path) if f.startswith("Global_Distortion") and f.endswith(".cif")]

    if not charge_files or not dist_files:
        print(f"  [SKIP] Missing CIF files in {folder}")
        continue

    # Path to the files
    charge_cif = os.path.join(folder_path, charge_files[0])
    dist_cif = os.path.join(folder_path, dist_files[0])

    # Harvest Data
    try:
        elements, charges = parse_custom_cif(charge_cif)
        _, distortions = parse_custom_cif(dist_cif)

        df = pd.DataFrame({"Element": elements, "Charge": charges, "Distortion": distortions})
        # Filter out Oxygen to see the Cation behavior clearly
        df = df[df["Element"] != "O"]

        # --- 3. PLOT ---
        plt.figure(figsize=(10, 7))
        # Use a consistent palette so the same element is always the same color across folders
        sns.scatterplot(data=df, x="Charges", y="Distortion", hue="Element", s=80, alpha=0.6, edgecolor='w')
        
        plt.title(f"Bivariate Analysis: {folder}")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save plot inside the specific folder
        plot_name = f"Bivariate_Analysis_{folder}.png"
        plt.savefig(os.path.join(folder_path, plot_name), dpi=300, bbox_inches='tight')
        plt.close() # Close plot to save memory during the loop
        
        print(f"  [DONE] Plot saved: {plot_name}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {folder}: {e}")

print("\nAll folders processed successfully.")