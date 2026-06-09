import re

input_file = "Global_Distortion_PdCuCoNiCr.cif"
output_file = "Clean_Distortion_PdCuCoNiCr.cif"

with open(input_file, 'r') as f:
    lines = f.readlines()

with open(output_file, 'w') as f:
    for line in lines:
        # This regex looks for things like "Co123 Co" and turns them into "Co Co"
        cleaned_line = re.sub(r'^([a-zA-Z]{1,2})\d+\s+', r'\1 ', line)
        f.write(cleaned_line)

print(f"Done! Open {output_file} in OVITO.")