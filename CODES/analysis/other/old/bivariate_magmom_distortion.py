import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 1. SETUP ---
warnings.filterwarnings("ignore")
root_path = os.path.dirname(os.path.abspath(__file__))

# Consistent color palette for your HEO elements
HEO_PALETTE = {
    "Ni": "#56DA63",    # Mid-Green
    "Co": "#B6707E",    # Dusty Rose
    "Cu": "#EE8E2D",    # Bronze/Brown
    "Cr": "#66799F",    # Muted Blue-Grey
    "Pd": "#004E63",    # Dark Teal/Navy
    "O": "#FFFFFF"      # Keep Oxygen white/hidden
}

result_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and f.startswith("Results_")]
print(f"Found {len(result_folders)} folders to analyze.\n")

# --- 2. THE LOOP ---
for folder in result_folders:
    folder_path = os.path.join(root_path, folder)
    print(f"Processing: {folder}...")

    # Identify the Master Data file (One file, all columns)
    master_files = [f for f in os.listdir(folder_path) if f.startswith("Bivariate_Master_Data") and f.endswith(".csv")]

    if not master_files:
        print(f"   [WARN] Missing Master CSV file in {folder}")
        continue

    try:
        # 3. Harvest Data (Reading from the SAME file, different columns)
        csv_path = os.path.join(folder_path, master_files[0])
        df = pd.read_csv(csv_path)

        # Filter for cations only
        df = df[df["Element"] != "O"]

        # 4. SCIENTIFIC PLOT: Mag_Mom vs Distortion_Index
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df, 
            x="Mag_Mom",           # Switched from SFI/Ionicity to column 2
            y="Distortion_Index",  # Keeping column 5
            hue="Element", 
            palette=HEO_PALETTE,
            s=100, 
            alpha=0.6, 
            edgecolor='none'
        )
        
        # Aesthetics for the progress meeting
        plt.title(f"Magnetic-Structural Correlation: {folder}", fontsize=14)
        plt.xlabel(r"Magnetic Moment ($\mu_B$)", fontsize=12)
        plt.ylabel("Lattice Distortion Index (Bond Variance)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(title="Cations", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        plot_name = f"Bivariate_MagMom_vs_Distortion_{folder}.png"
        plt.savefig(os.path.join(folder_path, plot_name), dpi=300, bbox_inches='tight')
        plt.close() 
        
        print(f"   [DONE] Created: {plot_name}")

    except Exception as e:
        print(f"   [ERROR] Failed to process {folder}: {e}")

print("\nAll bivariate plots updated using Master Data columns.")