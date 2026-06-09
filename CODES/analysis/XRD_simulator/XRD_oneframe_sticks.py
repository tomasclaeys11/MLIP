import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# 1. Laad de kristalstructuur in vanuit je CIF-bestand
structure = Structure.from_file("extra\MgO_rocksalt\MgO_perfect_UNRELAXED.cif")

# 2. Initialiseer de XRD calculator (standaard met Cu K-alpha straling)
xrd_calc = XRDCalculator(wavelength="CuKa")

# 3. Bereken het XRD-patroon
xrd_pattern = xrd_calc.get_pattern(structure)

# 4. Plot het patroon direct met de ingebouwde pymatgen visualisatie
xrd_calc.get_plot(structure,two_theta_range=(10,90),annotate_peaks=None)

# 5. Toon of sla de grafiek op
plt.savefig("xrd_pattern.png", dpi=300)
plt.show()