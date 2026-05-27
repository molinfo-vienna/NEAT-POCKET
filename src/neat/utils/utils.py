import numpy as np
from Bio.PDB import PDBParser, PDBIO


def center_pdb(input_path, output_path):
    # 1. Initialize the parser and load the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_path)

    # 2. Collect all atom coordinates
    # We use a list comprehension to gather coordinates of all atoms in the structure
    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])

    if len(coords) == 0:
        raise ValueError("No atoms found in the provided PDB file.")

    # 3. Calculate the geometric center (mean of all coordinates)
    # Note: If you want the center of mass instead, you would need to weight
    # these by element masses, but geometric center is the standard for "centering coordinates".
    geometric_center = coords.mean(axis=0)

    # 4. Subtract the center from each atom's coordinates
    # This shifts the structure so that its new geometric center is at (0, 0, 0)
    for atom in structure.get_atoms():
        new_coord = atom.get_coord() - geometric_center
        atom.set_coord(new_coord)

    # 5. Write the modified structure to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"Successfully centered structure and saved to {output_path}")
