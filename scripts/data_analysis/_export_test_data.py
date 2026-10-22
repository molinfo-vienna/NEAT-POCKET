"""
This script is used internally to export the test set pockets and their reference ligands
(CrossDocked or SPINDR) in the same format as the output of generation_conditional.py. This allows
evaluation.py to be run on the reference ligands, so that the metrics of the test set can be
reported alongside those of the generated molecules.
The exported data is made available to the user via the shared data repository found on Figshare (see README).
This script is not intended to be used by the user but is provided for completeness.

The dataset is selected via the DATA_SET global variable:
    - "CROSSDOCKED": pockets are read from PDB files
    - "SPINDR": pockets are read from CIF files and converted to PDB

For each test pocket, the pocket is centered at its geometric center and the reference ligand
(largest fragment, with hydrogens) is translated into the same frame, so that the exported files
are in the same coordinate frame as the molecules generated for that pocket.

It creates one folder per test pocket in the generated_data/<dataset>_test/conditional directory:
    1. pocket_{pdb_code}/pocket.pdb: centered pocket structure
    2. pocket_{pdb_code}/ligand.sdf: reference ligand, translated to match pocket.pdb
    3. pocket_{pdb_code}/generated_mols.pt: reference ligand as a tensor batch (read by evaluation.py)
    4. pocket_{pdb_code}/generated_mols.sdf: reference ligand as an SD file (for visualization)

To compute the metrics, set data_path to generated_data/<dataset>_test and data_set to the
dataset in scripts/config_files/config_evaluation.yaml and run scripts/evaluation.py.

This script must be run from the repository root.
"""

import os
from datetime import datetime

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import SDWriter
from torch_geometric.data import Batch

from neat.dataset import DataModule
from neat.dataset.dataset_utils import _largest_fragment, _ligand_features
from neat.utils import center_pdb, cif_2_pdb

DATA_SET = "CROSSDOCKED"  # "CROSSDOCKED" or "SPINDR"
ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(ROOT, "generated_data", f"{DATA_SET.lower()}_test", "conditional")


def load_ligand_with_hydrogens(ligand_path: str) -> Chem.Mol | None:
    suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=False)
    if suppl is None or len(suppl) == 0 or suppl[0] is None:
        return None
    return _largest_fragment(suppl[0])


def write_centered_pocket(in_pocket_file: str, out_pdb_file: str) -> np.ndarray:
    """Write the pocket centered at its geometric COM as PDB and return the center."""
    if in_pocket_file.endswith(".cif"):
        return cif_2_pdb(in_pocket_file, out_pdb_file, return_center=True)
    if in_pocket_file.endswith(".pdb"):
        return center_pdb(in_pocket_file, out_pdb_file, return_center=True)
    raise ValueError(f"Unsupported pocket file format: {in_pocket_file}")


def center_mol(mol: Chem.Mol, center: np.ndarray) -> None:
    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(
            atom_idx,
            (pos.x - center[0], pos.y - center[1], pos.z - center[2]),
        )


def export() -> None:

    datamodule = DataModule(
        os.path.join(ROOT, "data"),
        DATA_SET,
    )
    datamodule.setup()
    test_data = datamodule.test_data

    exported = 0
    skipped = 0

    for data_idx in range(len(test_data)):
        data_point = test_data[data_idx]
        pdb_code = test_data.get_pdb_code_from_data_point(data_point)
        in_pocket_file = test_data.get_pocket_path_from_data_point(data_point)
        in_ligand_file = test_data.get_ligand_path_from_data_point(data_point)

        mol = load_ligand_with_hydrogens(in_ligand_file)
        if mol is None:
            print(
                f"Skipping pocket_{pdb_code} ({data_point.name}): "
                "ligand not found or failed to sanitize"
            )
            skipped += 1
            continue

        x, pos = _ligand_features(mol, get_charge=False)
        if x is None or pos is None:
            print(
                f"Skipping pocket_{pdb_code} ({data_point.name}): "
                "failed to convert ligand to tensor features"
            )
            skipped += 1
            continue

        out_dir = os.path.join(OUTPUT_DIR, f"pocket_{pdb_code}")
        os.makedirs(out_dir, exist_ok=True)

        center = write_centered_pocket(in_pocket_file, os.path.join(out_dir, "pocket.pdb"))
        center_mol(mol, center)
        pos = pos - torch.as_tensor(center, dtype=pos.dtype)

        writer = SDWriter(os.path.join(out_dir, "generated_mols.sdf"))
        writer.write(mol)
        writer.close()

        writer = SDWriter(os.path.join(out_dir, "ligand.sdf"))
        writer.write(mol)
        writer.close()

        reference_batch = Batch(
            x=x,
            pos=pos,
            batch=torch.zeros(x.size(0), dtype=torch.long),
        )
        torch.save(reference_batch, os.path.join(out_dir, "generated_mols.pt"))

        exported += 1
        print(f"Exported pocket_{pdb_code}: {data_point.name}")

    print(
        f"Exported {exported} {DATA_SET} test pockets to {OUTPUT_DIR}"
        + (f" ({skipped} skipped)" if skipped else "")
    )


if __name__ == "__main__":
    start_time = datetime.now()
    export()
    end_time = datetime.now()
    print(f"Total export time: {end_time - start_time}")
