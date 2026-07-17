import copy
import os

import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from tqdm import tqdm

from neat.dataset.dataset_crossdocked import _add_hydrogens_with_openbabel, _add_hydrogens_with_rdkit

ROOT = os.getcwd()
INPUT_PATH = os.path.join(ROOT, "data", "CROSSDOCKED", "raw", "crossdocked_pocket10")
OUTPUT_DIR = os.path.join(ROOT, "output", "crossdocked_all_ligands")
OUTPUT_SUBDIR = "seed_0"
DATA_SET = "CROSSDOCKED"

split_file_path = os.path.join(ROOT, "data", "CROSSDOCKED", "raw", "split_by_name.pt")
data_split = torch.load(split_file_path, map_location="cpu", weights_only=False)

selected_ligand_files_train = [ligand_file for pocket_file, ligand_file in data_split["train"]]
selected_ligand_files_test = [ligand_file for pocket_file, ligand_file in data_split["test"]]
selected_ligand_files = selected_ligand_files_train + selected_ligand_files_test
selected_ligand_files_tupled = [(file.split("/")[0], file.split("/")[1]) for file in selected_ligand_files]


ligands = []

for pocket_dir in tqdm(os.listdir(INPUT_PATH)):
    pocket_path = os.path.join(INPUT_PATH, pocket_dir)
    if os.path.isdir(pocket_path):
        for file in os.listdir(pocket_path):
            if file.endswith(".sdf"):
                if (pocket_dir, file) in selected_ligand_files_tupled:
                    ligand_file_path = os.path.join(pocket_path, file)
                    suppl = Chem.SDMolSupplier(str(ligand_file_path), sanitize=False, removeHs=False)
                    ligand = suppl[0]
                    if ligand is not None:
                        Chem.SanitizeMol(ligand, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                        ligands.append(ligand)

print("Number of ligands:", len(ligands))

canonical_smiles = [Chem.MolToSmiles(ligand, isomericSmiles=True, canonical=True) for ligand in ligands]
ligands_unique = [ligands[i] for i in range(len(ligands)) if canonical_smiles[i] not in canonical_smiles[:i]]
print("Number of unique ligands:", len(set(ligands_unique)))

ligands_unique_hydrogens = []
for ligand in tqdm(ligands_unique):
    te = TautomerEnumerator()
    ligand = te.Canonicalize(ligand)
    ligand_copy = copy.deepcopy(ligand)
    ligand = _add_hydrogens_with_rdkit(ligand)
    if ligand is None:
        ligand = _add_hydrogens_with_openbabel(ligand_copy)
    ligands_unique_hydrogens.append(ligand)

output_path = os.path.join(OUTPUT_DIR, OUTPUT_SUBDIR)
os.makedirs(output_path, exist_ok=True)
writer = Chem.SDWriter(os.path.join(output_path, "ligands.sdf"))
for ligand in ligands_unique_hydrogens:
    if ligand is not None:
        writer.write(ligand)
writer.close()

print(f"Exported {len(ligands_unique_hydrogens)} ligands to {output_path}")
