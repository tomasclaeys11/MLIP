import pandas as pd

def convert_cif_to_extxyz(cif_filename, output_xyz):
    """
    Converts a custom CIF (with Distortion/Magmom in occupancy) 
    to Extended XYZ for OVITO free version grouping.
    """
    atoms = []
    lattice = []
    
    with open(cif_filename, 'r') as f:
        lines = f.readlines()
        
    # Extract Lattice parameters
    a = b = c = alpha = beta = gamma = 90.0
    for line in lines:
        if "_cell_length_a" in line: a = line.split()[1]
        if "_cell_length_b" in line: b = line.split()[1]
        if "_cell_length_c" in line: c = line.split()[1]
        if "_cell_angle_alpha" in line: alpha = line.split()[1]
        if "_cell_angle_beta" in line: beta = line.split()[1]
        if "_cell_angle_gamma" in line: gamma = line.split()[1]
        
    # Extract Atom data
    start_reading = False
    for line in lines:
        if "_atom_site_occupancy" in line:
            start_reading = True
            continue
        if start_reading and line.strip() and not line.startswith('loop_'):
            parts = line.split()
            if len(parts) >= 6:
                # Element, X, Y, Z, PropertyValue
                atoms.append([parts[1], float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])

    # Write Extended XYZ
    with open(output_xyz, 'w') as f:
        f.write(f"{len(atoms)}\n")
        # Lattice and Properties metadata
        f.write(f'Lattice="{a} 0 0 0 {b} 0 0 0 {c}" Properties=species:S:1:pos:R:3:distortion:R:1\n')
        for atom in atoms:
            f.write(f"{atom[0]} {atom[1]*float(a)} {atom[2]*float(b)} {atom[3]*float(c)} {atom[4]}\n")

# Run the converter
convert_cif_to_extxyz("Paramgoodcatas\Results_PdCuCoNiCr\oxygen vacancy\Relaxed_Surface_With_Vacancy.cif", "Relaxed_Surface_With_Vacancy.xyz")
print("File 'HEO_Grouped.xyz' created. Open this in OVITO.")