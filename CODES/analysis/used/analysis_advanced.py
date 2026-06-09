import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open("Results_MgCuCoNiZn_20260507_221038\DigitalTwin_MgCuCoNiZn.json", "r") as f:
    d = json.load(f)

site_data = []
for site in d.get("sites", []):
    element = site["species"][0]["element"]
    if element == "O": continue
    
    props = site.get("properties", {})
    # Calculate Force Norm
    f_vec = props.get("force_vec", [0, 0, 0])
    f_norm = np.linalg.norm(f_vec)
    
    # Calculate Bond Anisotropy (Max - Min)
    b_max = props.get("bond_max", 0)
    b_min = props.get("bond_min", 0)
    anisotropy = b_max - b_min
    
    site_data.append({
        "Element": element,
        "MagMom": props.get("magmom_abs", 0),
        "Distortion": props.get("distortion_index", 0),
        "SFI": props.get("spin_fidelity", 0),
        "Ionicity": props.get("ionicity_index", 0),
        "Force_Norm": f_norm,
        "Anisotropy": anisotropy,
        "Mean_Bond": props.get("bond_mean", 0)
    })

df = pd.DataFrame(site_data)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

# Plot 1: Force Norm by Element (Mechanical Restlessness)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Element", y="Force_Norm", palette="Set2")
plt.title("Mechanical Stability: Residual Forces per Element", fontweight='bold')
plt.ylabel("Force Norm (eV/$\AA$)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Force_Stability.png", dpi=300)
plt.close()

# Plot 2: Bond Anisotropy vs MagMom (Does asymmetry kill spin?)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Anisotropy", y="MagMom", hue="Element", s=100, alpha=0.7)
plt.title("Asymmetry Impact: Bond Anisotropy vs. Magnetic Moment", fontweight='bold')
plt.xlabel("Bond Anisotropy ($Max - Min$) [$\AA$]")
plt.ylabel("Absolute MagMom ($\mu_B$)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Anisotropy_vs_MagMom.png", dpi=300)
plt.close()

# Plot 3: Force Norm vs Distortion ( Frustration check)
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="Distortion", y="Force_Norm", scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Structural Frustration: Distortion vs. Residual Force", fontweight='bold')
plt.xlabel("Distortion Index ($\sigma_{bonds}$)")
plt.ylabel("Force Norm (eV/$\AA$)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Frustration_Analysis.png", dpi=300)
plt.close()

# Plot 4: Ionicity vs SFI (Electronic Consistency)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="SFI", y="Ionicity", hue="Element", style="Element", s=100)
plt.title("Electronic Profile: Spin Fidelity vs. Ionicity Index", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Electronic_Profile.png", dpi=300)
plt.close()