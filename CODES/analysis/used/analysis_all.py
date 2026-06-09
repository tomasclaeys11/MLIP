import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Structure
from sklearn.metrics import mean_squared_error
from scipy import stats

# --- 1. CONFIGURATION ---
JSON_FILE = "Results_MgCuCoNiZn_20260507_221038\DigitalTwin_MgCuCoNiZn.json" 
CATIONS = ["Mg", "Cu", "Co", "Ni", "Zn"]
SEARCH_CUTOFF_CAT = 3.5  # For Shell 2 (Metal-Metal)
SEARCH_CUTOFF_O = 2.6    # For Shell 1 (Metal-Oxygen)

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)

# --- 2. DATA EXTRACTION ---
print(f"Loading {JSON_FILE}...")
with open(JSON_FILE, "r") as f:
    data = json.load(f)

struct = Structure.from_dict(data)
master_data = []
raw_bond_lengths = {cat: [] for cat in CATIONS} # Store raw distances for distributions

print("Performing unified structural analysis...")
for i, site in enumerate(struct):
    element = site.specie.symbol
    if element not in CATIONS:
        continue
    
    props = site.properties
    
    # A. Get Cation Neighbors (Shell 2) for Influence Analysis
    cat_neighbors = struct.get_neighbors(site, SEARCH_CUTOFF_CAT)
    cat_nbs = [nb.specie.symbol for nb in cat_neighbors if nb.specie.symbol != "O"]
    
    # B. Get Oxygen Neighbors (Shell 1) for Distributions
    o_neighbors = struct.get_neighbors(site, SEARCH_CUTOFF_O)
    for nb in o_neighbors:
        if nb.specie.symbol == "O":
            raw_bond_lengths[element].append(site.distance(nb))

    # C. Compile Site Metrics
    entry = {
        "Element": element,
        "Strain": props.get("distortion_index", 0),
        "MagMom": props.get("magmom_abs", 0),
        "SFI": props.get("spin_fidelity", 0),
        "Mean_Bond": props.get("bond_mean", 0)
    }
    entry.update({f"{nb}_nb": cat_nbs.count(nb) for nb in CATIONS})
    master_data.append(entry)

df = pd.DataFrame(master_data)

# --- 3. HEATMAPS & SENSITIVITY ---
def generate_heatmap(df, target_col, title, filename, cmap):
    matrix = []
    for center in CATIONS:
        sub = df[df["Element"] == center]
        if sub[target_col].std() > 0:
            corrs = sub[[f"{c}_nb" for c in CATIONS] + [target_col]].corr()[target_col].drop(target_col)
            corrs.index = [idx.replace("_nb", "") for idx in corrs.index]
            matrix.append(corrs)
        else:
            matrix.append(pd.Series(0.0, index=CATIONS))
    
    impact_df = pd.DataFrame(matrix, index=CATIONS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(impact_df, annot=True, cmap=cmap, center=0, fmt=".2f")
    plt.title(title, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

generate_heatmap(df, "Strain", "Neighbor Influence on Strain", "Heatmap_Strain.png", "RdBu_r")
generate_heatmap(df, "MagMom", "Neighbor Influence on MagMom", "Heatmap_MagMom.png", "PRGn")
generate_heatmap(df, "SFI", "Neighbor Influence on SFI", "Heatmap_SFI.png", "PiYG")

# --- 4. NEW: BOND LENGTH DISTRIBUTIONS (5-PANEL) ---
print("Generating bond length distributions...")
fig, axes = plt.subplots(5, 1, figsize=(8, 18), sharex=True)
colors = sns.color_palette("husl", len(CATIONS))

for i, cat in enumerate(CATIONS):
    sns.histplot(raw_bond_lengths[cat], bins=15, kde=True, ax=axes[i], 
                 color=colors[i], edgecolor='white', alpha=0.7)
    mean_val = np.mean(raw_bond_lengths[cat])
    std_val = np.std(raw_bond_lengths[cat])
    axes[i].axvline(mean_val, color='black', linestyle='--', label=f'Mean: {mean_val:.3f}Å')
    axes[i].set_title(f"Bond Length Distribution: {cat}-O ($\sigma$ = {std_val:.3f}Å)", fontweight='bold')
    axes[i].set_ylabel("Frequency")
    axes[i].legend(loc='upper right')

axes[-1].set_xlabel("Bond Distance ($\AA$)")
plt.tight_layout()
plt.savefig("Cation_Bond_Distributions.png", dpi=300)
plt.close()

#--- boxplot_SFI-----
plt.figure(figsize=(10, 6))

# Main boxplot showing the quartiles and median
sns.boxplot(data=df, x="Element", y="SFI", palette="Set2", order=CATIONS)

# Overlaying individual points (stripplot) to see the distribution of all 216 sites
sns.stripplot(data=df, x="Element", y="SFI", color=".3", size=4, alpha=0.4, order=CATIONS)

plt.title("Actual Spin Fidelity (SFI) Values per Element", fontweight='bold', pad=15)
plt.ylabel("Absolute SFI (1.0 = Perfect Ground State)")
plt.xlabel("Cation Element")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, 1.05) # SFI is a normalized index

plt.tight_layout()
plt.savefig("Absolute_SFI_boxplot.png", dpi=300)
plt.close()

# --- 5. LANDSCAPE & CORRELATIONS ---
# Magnetostructural Landscape
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="MagMom", y="Strain", hue="Element", style="Element", s=120, alpha=0.8, palette="deep")
plt.title("Magnetostructural Landscape: Distortion vs. Magnetic Moment", fontweight='bold')
plt.tight_layout()
plt.savefig("HEO_Magnetostructural_Landscape.png", dpi=300)
plt.close()

# Detailed Correlation (R2/RMSE)
def get_metrics(subset):
    x, y = subset["Mean_Bond"], subset["Strain"]
    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    return r_val**2, np.sqrt(mean_squared_error(y, y_pred))

stats_summary = {cat: get_metrics(df[df["Element"] == cat]) for cat in CATIONS}
global_r2, global_rmse = get_metrics(df)

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df, x="Mean_Bond", y="Strain", hue="Element", s=100, alpha=0.7, palette="viridis")
slope, intercept, _, _, _ = stats.linregress(df["Mean_Bond"], df["Strain"])
x_range = np.array([df["Mean_Bond"].min(), df["Mean_Bond"].max()])
ax.plot(x_range, slope * x_range + intercept, color='red', linestyle='--')

info_text = f"Global: $R^2={global_r2:.3f}$, RMSE={global_rmse:.4f}\n" + "-"*30 + "\n"
for cat, (r2, rmse) in stats_summary.items():
    info_text += f"{cat}: $R^2={r2:.3f}$, RMSE={rmse:.4f}\n"

ax.text(1.02, 0.5, info_text, transform=ax.transAxes, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.title("Per-Cation Correlation: Mean Bond Length vs. Distortion Index", fontweight='bold')
plt.savefig("Detailed_HEO_Correlation.png", dpi=300, bbox_inches='tight')

print("\n--- MASTER ANALYSIS COMPLETE ---")