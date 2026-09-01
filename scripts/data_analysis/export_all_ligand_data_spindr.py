import os
from pathlib import Path

from rdkit.Chem import SDMolSupplier, SDWriter

ROOT = os.getcwd()
INPUT_DIR = os.path.join(ROOT, "data", "SPINDR", "raw")
OUTPUT_DIR = os.path.join(ROOT, "output", "spindr_all_ligands")
DATA_SET = "CROSSDOCKED"

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_mols = []
val_mols = []
test_mols = []

for file in Path(INPUT_DIR, "test").glob("*.sdf"):
    mols = SDMolSupplier(file, removeHs=False, sanitize=False)
    test_mols.extend(mols)

for file in Path(INPUT_DIR, "train").glob("*.sdf"):
    mols = SDMolSupplier(file, removeHs=False, sanitize=False)
    train_mols.extend(mols)

for file in Path(INPUT_DIR, "val").glob("*.sdf"):
    mols = SDMolSupplier(file, removeHs=False, sanitize=False)
    val_mols.extend(mols)

print(f"Number of test ligands: {len(test_mols)}")
print(f"Number of train ligands: {len(train_mols)}")
print(f"Number of val ligands: {len(val_mols)}")

all_mols = test_mols + train_mols + val_mols

print(f"Number of all ligands: {len(all_mols)}")

all_mols = [mol for mol in all_mols if mol is not None]

print(f"Number of all ligands after removal of None: {len(all_mols)}")

writer = SDWriter(Path(OUTPUT_DIR, "ligands.sdf"))
for mol in all_mols:
    writer.write(mol)
writer.close()
