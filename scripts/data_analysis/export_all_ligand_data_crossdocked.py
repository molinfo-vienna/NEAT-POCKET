import os
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

from neat.dataset.dataset_crossdocked import (_add_hydrogens_with_openbabel,
                                              _add_hydrogens_with_rdkit)

ROOT = Path("/home/rjacob/NEAT-Cond-Dev")

ROOT = os.getcwd()
INPUT_PATH = os.path.join(ROOT, "data", "CROSSDOCKED", "raw", "crossdocked_pocket10")
OUTPUT_DIR = os.path.join(ROOT, "output", "crossdocked_all_ligands")
DATA_SET = "CROSSDOCKED"

split_file_path = os.path.join(ROOT, "data", "CROSSDOCKED", "raw", "split_by_name.pt")
data_split = torch.load(split_file_path, map_location="cpu", weights_only=False)

pairs_train = data_split.get("train")
pairs_test = data_split.get("test")
pairs_all = pairs_train + pairs_test


def load_molecules(pairs, datadir):
    def load_mol(ligand_file_name):
        suppl = Chem.SDMolSupplier(
            str(datadir / Path(ligand_file_name)), sanitize=True, removeHs=False
        )
        return suppl[0] if suppl and len(suppl) > 0 else None

    return list(
        filter(
            None,
            tqdm(
                map(load_mol, (ligand for _, ligand in pairs)),
                desc="Loading molecules",
                total=len(pairs),
            ),
        )
    )


train_mols = load_molecules(pairs_train, INPUT_PATH)
test_mols = load_molecules(pairs_test, INPUT_PATH)
all_mols = train_mols + test_mols

print(f"Number of train ligands: {len(train_mols)}")
print(f"Number of test ligands: {len(test_mols)}")
print(f"Number of all ligands: {len(all_mols)}")

all_mols_canonical_smiles = [
    Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) for mol in all_mols
]
all_mols_unique = [
    all_mols[i]
    for i in range(len(all_mols))
    if all_mols_canonical_smiles[i] not in all_mols_canonical_smiles[:i]
]
print(f"Number of unique ligands: {len(all_mols_unique)}")
with Chem.SDWriter(os.path.join(OUTPUT_DIR, "ligands_raw.sdf")) as w:
    for mol in all_mols_unique:
        w.write(mol)

all_mols_unique_rdkit_default = [
    Chem.AddHs(mol, addCoords=True) for mol in all_mols_unique if mol is not None
]
all_mols_unique_rdkit_default = [
    mol for mol in all_mols_unique_rdkit_default if mol is not None
]
print(
    f"Number of unique ligands with RDKit default hydrogenation: {len(all_mols_unique_rdkit_default)}"
)
with Chem.SDWriter(os.path.join(OUTPUT_DIR, "ligands_rdkit_default.sdf")) as w:
    for mol in all_mols_unique_rdkit_default:
        w.write(mol)

all_mols_unique_rdkit_embed = [
    _add_hydrogens_with_rdkit(mol) for mol in all_mols_unique if mol is not None
]
all_mols_unique_rdkit_embed = [
    mol for mol in all_mols_unique_rdkit_embed if mol is not None
]
print(
    f"Number of unique ligands with RDKit embed hydrogenation: {len(all_mols_unique_rdkit_embed)}"
)
with Chem.SDWriter(os.path.join(OUTPUT_DIR, "ligands_rdkit_embed.sdf")) as w:
    for mol in all_mols_unique_rdkit_embed:
        w.write(mol)

all_mols_unique_openbabel = [
    _add_hydrogens_with_openbabel(mol) for mol in all_mols_unique if mol is not None
]
all_mols_unique_openbabel = [
    mol for mol in all_mols_unique_openbabel if mol is not None
]
print(
    f"Number of unique ligands with OpenBabel hydrogenation: {len(all_mols_unique_openbabel)}"
)
with Chem.SDWriter(os.path.join(OUTPUT_DIR, "ligands_openbabel.sdf")) as w:
    for mol in all_mols_unique_openbabel:
        w.write(mol)
