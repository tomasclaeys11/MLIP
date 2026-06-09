import os
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# =========================
# USER INPUTS
# =========================

cif_file = r"relaxation_V3_results\PdCuCoNiCr\Results_PdCuCoNiCr_20260507_150820\relaxed_PdCuCoNiCr.cif"
COMP="PdCuCoNiCr_relaxed"
# symmetry refinement tolerance
symprec = 0.05

# XRD settings
two_theta_range = (10, 90)
wavelength = "CuKa"

# peak broadening (Gaussian sigma in degrees)
sigma = 0.1

# plotting resolution
num_points = 4000

# =========================
# SETUP
# =========================

output_dir = os.path.join(os.getcwd(), "XRD_results")
os.makedirs(output_dir, exist_ok=True)

xrd_calc = XRDCalculator(wavelength=wavelength)

# =========================
# FUNCTIONS
# =========================

def refine_structure(structure):
    """
    Mild symmetry refinement.
    Keeps physical structure while removing tiny numerical distortions.
    """
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        refined = sga.get_refined_structure()
        return refined
    except Exception as e:
        print(f"Refinement failed: {e}")
        return structure


def compute_pattern(structure):
    """
    Compute raw pymatgen XRD pattern.
    """
    return xrd_calc.get_pattern(
        structure,
        two_theta_range=two_theta_range
    )


def gaussian_broadening(x_peaks, y_peaks, sigma, x_grid):
    """
    Apply physically reasonable Gaussian peak broadening.
    """
    y_broadened = np.zeros_like(x_grid)

    for xp, yp in zip(x_peaks, y_peaks):
        y_broadened += yp * np.exp(
            -(x_grid - xp)**2 / (2 * sigma**2)
        )

    return y_broadened


# =========================
# LOAD STRUCTURE
# =========================

print("Loading CIF structure...")

structure = Structure.from_file(cif_file)

# symmetry refinement
structure = refine_structure(structure)

# =========================
# COMPUTE XRD
# =========================

print("Computing XRD pattern...")

pattern = compute_pattern(structure)

x_peaks = np.array(pattern.x)
y_peaks = np.array(pattern.y)

# =========================
# APPLY PEAK BROADENING
# =========================

x_grid = np.linspace(
    two_theta_range[0],
    two_theta_range[1],
    num_points
)

y_broadened = gaussian_broadening(
    x_peaks,
    y_peaks,
    sigma,
    x_grid
)

# normalize
y_broadened /= np.max(y_broadened)

# =========================
# PLOT XRD
# =========================

plt.figure(figsize=(8, 5), dpi=300)

plt.plot(x_grid, y_broadened, linewidth=1.5)

plt.xlabel(r"2$\theta$ (degree)", fontsize=12)
plt.ylabel("Normalized Intensity", fontsize=12)

plt.title("Simulated XRD Pattern (Cu Kα)", fontsize=13)

plt.xlim(two_theta_range)

plt.tight_layout()

output_file = os.path.join(
    output_dir,
    f"Simulated_XRD_{COMP}.pdf"
)

plt.savefig(output_file)
plt.close()
'''
# =========================
# OPTIONAL: STICK PATTERN
# =========================

plt.figure(figsize=(8, 5), dpi=300)

for xp, yp in zip(x_peaks, y_peaks):
    plt.vlines(xp, 0, yp)

plt.xlabel(r"2$\theta$ (degree)", fontsize=12)
plt.ylabel("Intensity (a.u.)", fontsize=12)

plt.title("Raw Stick XRD Pattern", fontsize=13)

plt.xlim(two_theta_range)

plt.tight_layout()

stick_output = os.path.join(
    output_dir,
    "Simulated_XRD_stick.pdf"
)

plt.savefig(stick_output)
plt.close()
'''

plt.figure(figsize=(10, 6), dpi=300)

# broadened XRD curve
plt.plot(x_grid, y_broadened, linewidth=1.5)

# annotate peaks with hkl indices
for peak_x, peak_y, hkls in zip(pattern.x, pattern.y, pattern.hkls):

    # normalize peak height to broadened pattern scale
    peak_y_norm = peak_y / np.max(pattern.y)

    # get first hkl family
    hkl = hkls[0]["hkl"]

    # format nicely
    hkl_label = f"({hkl[0]}{hkl[1]}{hkl[2]})"

    # place annotation slightly above peak
    plt.text(
        peak_x,
        peak_y_norm + 0.03,
        hkl_label,
        rotation=90,
        fontsize=8,
        ha='center',
        va='bottom'
    )

# labels and formatting
plt.xlabel(r"2$\theta$ (degree)", fontsize=12)
plt.ylabel("Normalized Intensity", fontsize=12)

plt.title("Simulated XRD Pattern (Cu Kα)", fontsize=13)

plt.xlim(two_theta_range)

plt.tight_layout()

output_file = os.path.join(
    output_dir,
    f"Simulated_XRD_{COMP}_with_hkl.pdf"
)

plt.savefig(output_file)
plt.close()
# =========================
# DONE
# =========================

print(f"\nDone.")
print(f"Results saved in: {output_dir}")