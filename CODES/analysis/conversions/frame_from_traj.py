import os
from ase.io import read, write

TRAJ_FILE = "HPC_bin\MD_results\MD_216_PdCuCoNiCr\movies\MD_700C_PdCuCoNiCr (1).traj"

# 1. Lees alle beschikbare frames in (stream-modus om geheugen te sparen)
print(f"Bezig met indexeren van {TRAJ_FILE}...")
frames = read(TRAJ_FILE, index=":")
total_frames = len(frames)
print(f"Succes! Trajectory bevat {total_frames} frames (indices: 0 tot {total_frames - 1}).\n")

# 2. Vraag in de terminal om de gewenste frame-index
try:
    user_input = input(f"Welk frame wil je omzetten naar CIF? (0 t/m {total_frames - 1}, of -1 voor de laatste): ")
    frame_idx = int(user_input)
    
    # 3. Haal het gekozen frame op en schrijf het weg
    selected_atoms = frames[frame_idx]
    
    # Bepaal een logische indexnaam voor de file (vloeit netjes over bij negatieve indices)
    actual_idx = frame_idx if frame_idx >= 0 else total_frames + frame_idx
    output_cif = f"frame_{actual_idx}.cif"
    
    write(output_cif, selected_atoms)
    print(f"[Done] Frame {actual_idx} is succesvol opgeslagen als '{output_cif}'!")

except IndexError:
    print(f"Fout: Index bestaat niet. Kies een getal tussen 0 en {total_frames - 1}.")
except ValueError:
    print("Fout: Voer aanzienlijk een geldig heel getal in.")