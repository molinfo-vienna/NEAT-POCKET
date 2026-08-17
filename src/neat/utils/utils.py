from pathlib import Path

import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
from Bio.PDB import PDBIO, PDBParser
from rdkit.Chem import SDWriter


def center_pdb(input_path, output_path, return_center=False):
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

    if return_center:
        return geometric_center


def cif_2_pdb(input_path, output_path, return_center=False):
    # 1. Initialize the parser and load the structure
    file = pdbx.CIFFile.read(str(input_path))
    cif_model = pdbx.get_structure(file, model=1)
    com = np.mean(cif_model.coord, axis=0)
    cif_model.coord -= com

    import string

    # 3. Remap multi-character chain IDs to 1-character IDs
    unique_chains = list(dict.fromkeys(cif_model.chain_id))
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits

    chain_map = {
        old_id: alphabet[i % len(alphabet)] for i, old_id in enumerate(unique_chains)
    }

    # Update the chain IDs in place
    cif_model.chain_id = [chain_map[c] for c in cif_model.chain_id]

    # 4. Save and print to console
    output_pdb = pdb.PDBFile()
    pdb.set_structure(output_pdb, cif_model)

    output_pdb.write(str(output_path))

    if return_center:
        return com


def save_molecules_to_sdf(mols: list, file_path: Path) -> None:
    """Write non-None RDKit molecules to an SDF file."""
    writer = SDWriter(str(file_path))
    try:
        for mol in mols:
            if mol is None:
                continue
            try:
                writer.write(mol)
            except Exception as e:
                print(f"Error while writing molecules to SDF: {e}")
    finally:
        writer.close()
