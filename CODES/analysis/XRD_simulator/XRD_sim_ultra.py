import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Structure
from pymatgen.io.vasp import Xdatcar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# --- 1. CONFIGURATION ---
SYMPREC = 0.1  # Precision for symmetry analysis
FWHM = 0.2     # Gaussian broadening width
THETA_RANGE = np.linspace(10, 90, 500) 

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

class XRDSimulator:
    def __init__(self, wavelength="CuKa", symprec=SYMPREC):
        self.calculator = XRDCalculator(wavelength=wavelength, symprec=symprec)

    def gaussian_broadening(self, x, y, x_grid, fwhm):
        """Converts Dirac spikes into Gaussian curves."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        spectrum = np.zeros_like(x_grid)
        for xi, yi in zip(x, y):
            # Optimized vectorization for Gaussian application
            spectrum += yi * np.exp(-0.5 * ((x_grid - xi) / sigma)**2)
        return spectrum

    def parse_frames(self, user_str, total):
        """Advanced frame selector (strips quotes and handles ranges)."""
        indices = []
        user_str = user_str.replace("'", "").replace('"', "").strip()
        for part in [p.strip() for p in user_str.split(',')]:
            try:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, min(end + 1, total)))
                else:
                    idx = int(part)
                    indices.append(idx if idx >= 0 else total + idx)
            except ValueError: continue
        return sorted(list(set([i for i in indices if 0 <= i < total])))

    def get_structure_at_idx(self, traj_obj, idx, is_ase=False):
        if is_ase:
            return AseAtomsAdaptor.get_structure(traj_obj[idx])
        return traj_obj.structures[idx]

    def run(self):
        target = input("Enter file path (CIF, .traj, or XDATCAR): ").strip()
        if not os.path.exists(target): return print("File not found.")

        # Overlay Logic
        ref_pattern = None
        if input("Overlay perfect CIF? (y/n): ").lower() == 'y':
            ref_path = input("Enter reference CIF: ").strip()
            ref_struct = Structure.from_file(ref_path)
            ref_raw = self.calculator.get_pattern(ref_struct)
            ref_pattern = self.gaussian_broadening(ref_raw.x, ref_raw.y, THETA_RANGE, FWHM)

        # Load Trajectory
        is_ase = target.lower().endswith(".traj")
        try:
            if is_ase:
                from ase.io.trajectory import Trajectory as ASETraj
                traj_obj = ASETraj(target)
                total = len(traj_obj)
            else:
                traj_obj = Xdatcar(target)
                total = len(traj_obj.structures)
        except Exception as e:
            return print(f"Error loading: {e}")

        print(f"Loaded {total} frames.")
        sel_str = input("Enter frame range (e.g. 0-2500): ")
        indices = self.parse_frames(sel_str, total)
        if not indices: return print("No frames selected.")

        window = int(input("Averaging window size (e.g. 25): ") or 1)

        # --- 2. PHASE 1: PRE-CALCULATE & CACHE (FAST MODE) ---
        precalc_spectra = []
        print(f"Phase 1: Pre-calculating {len(indices)} unique patterns...")
        for count, idx in enumerate(indices):
            struct = self.get_structure_at_idx(traj_obj, idx, is_ase)
            raw = self.calculator.get_pattern(struct)
            # Store the 1D spectrum immediately
            spectrum = self.gaussian_broadening(raw.x, raw.y, THETA_RANGE, FWHM)
            precalc_spectra.append(spectrum)
            if (count + 1) % 100 == 0:
                print(f"  Processed {count + 1}/{len(indices)} frames...")

        # --- 3. PHASE 2: MOVING AVERAGE WINDOW ---
        intensity_map = []
        print(f"Phase 2: Applying window (Size: {window}) to smooth thermal noise...")
        for i in range(len(precalc_spectra)):
            win_start = max(0, i - window // 2)
            win_end = min(len(precalc_spectra), i + window // 2 + 1)
            # Average the ALREADY calculated 1D arrays
            avg_spectrum = np.mean(precalc_spectra[win_start:win_end], axis=0)
            intensity_map.append(avg_spectrum)

        # --- 4. 2D PLOTTING ---
        z_data = np.array(intensity_map).T 
        plt.figure(figsize=(12, 8))
        mesh = plt.pcolormesh(indices, THETA_RANGE, z_data, shading='auto', cmap='inferno')
        
        cbar = plt.colorbar(mesh)
        cbar.set_label('XRD Intensity (a.u.)', fontweight='bold')
        
        plt.title(f"XRD Evolution (700°C): {os.path.basename(target)}", fontweight='bold')
        plt.xlabel("Frame Index (Time Evolution)")
        plt.ylabel("$2\\theta$ (degrees)")
        plt.ylim(10, 90)
        
        plt.tight_layout()
        plt.savefig("XRD_2D_Evolution_Fast.png", dpi=300)
        print("Success! Plot saved as XRD_2D_Evolution_Fast.png")

if __name__ == "__main__":
    XRDSimulator().run()