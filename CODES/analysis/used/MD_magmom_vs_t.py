import pandas as pd
import matplotlib.pyplot as plt

# --- 1. SETUP & PARAMETERS ---
csv_file = "MD_700C_PdCuCoNiCr_MagMoms_500ps.csv"

# Make sure this matches the TIMESTEP_FS used in your NPT MD script!
TIMESTEP_FS = 2.0  

# Consistent color palette from your previous scripts
HEO_PALETTE = {
    "Ni": "#56DA63",    # Mid-Green
    "Co": "#B6707E",    # Dusty Rose
    "Cu": "#EE8E2D",    # Bronze/Brown
    "Cr": "#66799F",    # Muted Blue-Grey
    "Pd": "#004E63"     # Dark Teal/Navy
}

# --- 2. LOAD & PROCESS DATA ---
df = pd.read_csv(csv_file)

# Convert MD 'Step' to Time in picoseconds (ps)
# Time (ps) = (Step * Timestep in fs) / 1000
df['Time_ps'] = df['Step'] * TIMESTEP_FS / 1000.0

# --- 3. PLOTTING ---
plt.figure(figsize=(10, 7))

# The metals we want to plot
elements = ["Pd", "Cu", "Co", "Ni", "Cr"]

for elem in elements:
    col_avg = f"{elem}_MagMom_Avg"
    col_std = f"{elem}_MagMom_Std"
    
    # Check if the element exists in the CSV
    if col_avg in df.columns and col_std in df.columns:
        
        # Plot the average line
        plt.plot(
            df['Time_ps'], 
            df[col_avg], 
            label=elem, 
            color=HEO_PALETTE[elem], 
            linewidth=2
        )
        
        # Plot the error bars as a shaded region (Standard Deviation)
        plt.fill_between(
            df['Time_ps'], 
            df[col_avg] - df[col_std], 
            df[col_avg] + df[col_std], 
            color=HEO_PALETTE[elem], 
            alpha=0.2  # 20% opacity for the shaded error band
        )

# --- 4. AESTHETICS & EXPORT ---
plt.title("Evolution of Average Magnetic Moments (700°C MD)", fontsize=14)
plt.xlabel("Time (ps)", fontsize=12)
plt.ylabel(r"Average Magnetic Moment ($\mu_B$)", fontsize=12)

# Keep the same grid and legend layout
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title="Cations", bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout so the legend isn't cut off
plt.tight_layout()

# Save the plot
output_filename = "MD_PdCuCoNiCr_MagMom_vs_Time_700C_500ps.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved as: {output_filename}")