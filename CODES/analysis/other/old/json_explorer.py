import os
import json
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

print(f"Found {len(result_folders)} folders to analyze.\n")

# --- 2. THE LOOP ---
for folder in result_folders:
    folder_path = os.path.join(root_path, folder)
    print(f"Processing: {folder}...")

    # Look for the Digital Twin JSON file
    json_files = [f for f in os.listdir(folder_path) if f.startswith("DigitalTwin") and f.endswith(".json")]

    if not json_files:
        print(f"  [SKIP] No DigitalTwin JSON found in {folder}")
        continue

    json_path = os.path.join(folder_path, json_files[0])

    # Harvest Data from JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        site_list = []
        for site in data.get("sites", []):
            # Extract element from the species list
            element = site["species"][0]["element"]
            
            # Skip Oxygen to see cation behavior clearly
            if element == "O":
                continue
                
            props = site.get("properties", {})
            site_list.append({
                "Element": element,
                "MagMom": props.get("magmom_abs", 0),
                "Distortion": props.get("distortion_index", 0)
            })

        df = pd.DataFrame(site_list)

        # --- 3. PLOT ---
        plt.style.use('seaborn-v0_8-paper')
        plt.figure(figsize=(10, 7))
        
        # Use a consistent palette
        sns.scatterplot(
            data=df, 
            x="MagMom", 
            y="Distortion", 
            hue="Element", 
            style="Element",
            s=100, 
            alpha=0.7, 
            edgecolor='w',
            palette="deep"
        )
        
        plt.title(f"Bivariate Analysis: {folder}\n(Magnetostructural Coupling)", fontweight='bold')
        plt.xlabel("Absolute Magnetic Moment ($\mu_B$)")
        plt.ylabel("Local Distortion Index ($\sigma_{bonds}$)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Cation", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot inside the specific folder
        plot_name = f"Bivariate_Analysis_{folder}.png"
        plt.savefig(os.path.join(folder_path, plot_name), dpi=300, bbox_inches='tight')
        plt.close() 
        
        print(f"  [DONE] Plot saved: {plot_name}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {folder}: {e}")

print("\nAll folders processed successfully.")