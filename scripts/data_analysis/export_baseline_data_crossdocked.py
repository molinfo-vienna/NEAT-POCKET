"""Export benchmark data for evaluation.

Creates one folder per test pocket with the same layout as generation_conditional.py:
    pocket_{idx}/pocket.pdb   - pocket structure, centered at geometric COM
    pocket_{idx}/ligand.sdf   - reference ligand, translated to match pocket.pdb
    pocket_{idx}/generated_mols.pt - reference ligand with hydrogens as a tensor batch (for evaluation.py)
    pocket_{idx}/generated_mols.sdf - reference ligand with hydrogens as an SD file (for visualization)
"""

import logging
import os

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from neat.dataset.dataset_crossdocked import _add_hydrogens

ROOT = os.getcwd()
IN_DIR = os.path.join(ROOT, "data", "BENCHMARK")
OUT_DIR = os.path.join(ROOT, "output")

os.makedirs(OUT_DIR, exist_ok=True)

MODEL1 = "diffsbdd"
MODEL2 = "drugflow"
MODEL3 = "pocket2mol"
MODEL4 = "targetdiff"


def setup_looger(name, filename, level=logging.INFO):
    if os.path.exists(filename):
        os.remove(filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


mapping = {
    1: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    17: 11,
    33: 12,
    35: 13,
    53: 14,
    80: 15,
    83: 16,
}

split = torch.load(
    os.path.join(os.getcwd(), "data", "CROSSDOCKED", "raw", "split_by_name.pt")
)
pocket_name_id_mapping = {}
for i, (pocket_name, ligand_name) in enumerate(split["test"]):
    pocket_name = ligand_name.split("/")[1].split(".")[0]
    pocket_name_id_mapping[pocket_name] = i


for MODEL in [MODEL1, MODEL2, MODEL3, MODEL4]:

    os.makedirs(os.path.join(OUT_DIR, MODEL, "conditional"), exist_ok=True)

    logging_filename = os.path.join(
        OUT_DIR, MODEL, "conditional", "export_benchmark_data.log"
    )
    logger = setup_looger(MODEL, logging_filename)
    logger.info(f"Exporting {MODEL} benchmark data...")

    DIR = MODEL + "_samples"
    num_total_ligands = 0
    num_failed_ligands_reading = 0
    num_failed_ligands_hydrogenating = 0

    reference_ligands_folder = os.path.join(OUT_DIR, "crossdocked_test", "conditional")

    for i, pocket_dir in enumerate(os.listdir(os.path.join(IN_DIR, DIR))):
        # For each pocket directory, check if it exists
        if not os.path.isdir(os.path.join(IN_DIR, DIR, pocket_dir)):
            continue

        # Load the first pocket file (they are all the same)
        pocket_file = os.path.join(IN_DIR, DIR, pocket_dir, "0_pocket.pdb")

        # Load and hydrogenate the generated molecules
        generated_mols = []
        for file in os.listdir(os.path.join(IN_DIR, DIR, pocket_dir)):
            if file.endswith(".sdf"):
                num_total_ligands += 1
                generated_mol_file = os.path.join(IN_DIR, DIR, pocket_dir, file)
                generated_mol = Chem.SDMolSupplier(generated_mol_file)[0]
                generated_mol = _add_hydrogens(generated_mol)
                generated_mols.append(generated_mol)

        # Make the output directory
        pocket_id = pocket_name_id_mapping[pocket_dir]
        output_dir = os.path.join(OUT_DIR, MODEL, "conditional", f"pocket_{pocket_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Copy the pocket file to the output directory and rename it to pocket.pdb
        os.system(f"cp {pocket_file} {output_dir}")
        os.rename(
            os.path.join(output_dir, "0_pocket.pdb"),
            os.path.join(output_dir, "pocket.pdb"),
        )

        # Save the generated molecules as an SDF file
        writer = Chem.SDWriter(os.path.join(output_dir, "generated_mols.sdf"))
        for mol in generated_mols:
            if mol is not None:
                # Set aromaticity flags; careful: SanitizeMol is in place
                Chem.SanitizeMol(
                    mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                )
                if mol is None:
                    num_failed_ligands_hydrogenating += 1
                    continue
                writer.write(mol)
            else:
                num_failed_ligands_reading += 1
                print("Warning: None molecule found in pocket", i)
        writer.close()

        # Copy the correct reference ligand to the output directory
        reference_ligand_file = os.path.join(
            reference_ligands_folder, f"pocket_{pocket_id}", f"ligand.sdf"
        )
        os.system(f"cp {reference_ligand_file} {output_dir}")

        # Convert the generated molecules to a NEAT Batch object and save generated molecules as a pt file
        x = []
        pos = []
        batch = []
        for i, mol in enumerate(generated_mols):
            if mol is None:
                continue
            conformer = mol.GetConformer()
            x.append(
                torch.tensor(
                    [mapping[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
                    dtype=torch.long,
                )
            )
            sub_pos = []
            for atom in mol.GetAtoms():
                coords = conformer.GetAtomPosition(atom.GetIdx())
                coords = torch.tensor([coords.x, coords.y, coords.z], dtype=torch.float)
                sub_pos.append(coords)
            pos.append(torch.stack(sub_pos, dim=0))
            batch.append(torch.full((mol.GetNumAtoms(),), i, dtype=torch.long))
        x = torch.cat(x, dim=0)
        pos = torch.cat(pos, dim=0)
        batch = torch.cat(batch, dim=0)
        generated_mols = Batch(x=x, pos=pos, batch=batch)
        torch.save(generated_mols, os.path.join(output_dir, "generated_mols.pt"))

    logger.info(f"{num_total_ligands} ligands found across all pockets.")
    logger.info(
        f"{num_failed_ligands_reading} ligands failed to load across all pockets."
    )
    logger.info(
        f"{num_failed_ligands_hydrogenating} ligands failed to hydrogenate across all pockets."
    )
    logger.info(
        f"{num_total_ligands - num_failed_ligands_reading - num_failed_ligands_hydrogenating} ligands successfully loaded across all pockets."
    )
