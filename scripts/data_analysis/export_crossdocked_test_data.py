"""Export CrossDocked test set pockets and reference ligands for evaluation.

Creates one folder per test pocket with the same layout as generation_conditional.py:
    pocket_{pdb_code}/pocket.pdb   - pocket structure, centered at geometric COM
    pocket_{pdb_code}/ligand.sdf   - reference ligand, translated to match pocket.pdb
    pocket_{pdb_code}/generated_mols.pt - reference ligand with hydrogens as a tensor batch (for evaluation.py)
    pocket_{pdb_code}/generated_mols.sdf - reference ligand with hydrogens as an SD file (for visualization)
"""

import os
from datetime import datetime

import numpy as np
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import SDWriter
from torch_geometric.data import Batch

from neat.dataset import DataModule
from neat.dataset.dataset_crossdocked import _ligand_features
from neat.utils import center_pdb

ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(ROOT, "output", "crossdocked_test", "conditional")
DATA_SET = "CROSSDOCKED"


def load_ligand_with_hydrogens(ligand_path: str) -> Chem.Mol | None:
    suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=False)
    if suppl is None or len(suppl) == 0:
        return None
    mol = suppl[0]
    return mol


def pocket_geometric_center(pdb_path: str) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])
    if len(coords) == 0:
        raise ValueError(f"No atoms found in {pdb_path}")
    return coords.mean(axis=0)


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
        in_pdb_file = test_data.get_pocket_path_from_data_point(data_point)
        in_ligand_file = test_data.get_ligand_path_from_data_point(data_point)

        mol = load_ligand_with_hydrogens(in_ligand_file)
        if mol is None:
            print(
                f"Skipping pocket_{pdb_code} ({data_point.name}): "
                "ligand not found or failed to sanitize"
            )
            skipped += 1
            continue

        center = pocket_geometric_center(in_pdb_file)
        center_mol(mol, center)

        x, pos = _ligand_features(mol)
        if x is None or pos is None:
            print(
                f"Skipping pocket_{pdb_code} ({data_point.name}): "
                "failed to convert ligand to tensor features"
            )
            skipped += 1
            continue

        out_dir = os.path.join(ROOT, OUTPUT_DIR, f"pocket_{pdb_code}")
        os.makedirs(out_dir, exist_ok=True)

        center_pdb(in_pdb_file, os.path.join(out_dir, "pocket.pdb"))

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
        f"Exported {exported} test pockets to {os.path.join(ROOT, OUTPUT_DIR)}"
        + (f" ({skipped} skipped)" if skipped else "")
    )


if __name__ == "__main__":
    start_time = datetime.now()
    export()
    end_time = datetime.now()
    print(f"Total export time: {end_time - start_time}")
