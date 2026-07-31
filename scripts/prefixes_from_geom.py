import json
import os
import pickle
from collections import defaultdict
from datetime import datetime

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, SDWriter, rdDepictor
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ROOT = os.getcwd()
PREFIXES_DIR = os.path.join(ROOT, "prefixes")

def get_ring_system_components(mol: Chem.Mol) -> list[set[int]]:
    """Find sets of atoms constituting ring systems.

    A ring system is defined as the connected components of the subgraph
    induced by atoms that are in any ring, with edges restricted to bonds
    whose both endpoints are ring atoms.

    Args:
        mol (Chem.Mol): RDKit Mol object.

    Returns:
        list[set[int]]: list of sets of atom indices for each ring system.
    """
    ri = mol.GetRingInfo()
    ring_atoms = set(ri.AtomRings()[0]) if ri.AtomRings() else set()
    for r in ri.AtomRings()[1:]:
        ring_atoms.update(r)
    if not ring_atoms:
        return []

    # Build adjacency among ring atoms via ring bonds only
    ring_adj = {a: set() for a in ring_atoms}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in ring_atoms and a2 in ring_atoms:
            ring_adj[a1].add(a2)
            ring_adj[a2].add(a1)

    # Connected components on ring atom subgraph
    visited = set()
    components = []
    for a in ring_atoms:
        if a in visited:
            continue
        stack = [a]
        comp = set()
        visited.add(a)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nb in ring_adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(comp)
    return components


def build_ring_pattern_submol(mol: Chem.Mol, ring_atoms_set: set[int]) -> Chem.Mol:
    """Build a submol for a given ring system

    1. Includes all ring atoms and bonds between them.
    2. Adds explicit H neighbors attached to ring atoms.
    3. Replaces any non-H neighbor outside the ring system with a single 'R' dummy atom.

    Args:
        mol (Chem.Mol): RDKit Mol object.
        ring_atoms_set (set[int]): set of atom indices in the ring system.

    Returns:
        submol (Chem.Mol): RDKit Mol object representing the ring pattern.
    """
    # Work on a version with explicit Hs so we capture H attachments
    molH = Chem.AddHs(mol)

    # Map old ring atom idx -> new idx in submol
    newmol = Chem.RWMol()
    idx_map = {}

    # Add ring atoms with same properties (element, aromaticity)
    for aidx in sorted(ring_atoms_set):
        a = molH.GetAtomWithIdx(aidx)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        idx_map[aidx] = newmol.AddAtom(na)

    # Add bonds among ring atoms
    for b in molH.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in ring_atoms_set and a2 in ring_atoms_set:
            newmol.AddBond(idx_map[a1], idx_map[a2], b.GetBondType())

    # For each ring atom, attach H or R for neighbors outside ring system
    for old_aidx in sorted(ring_atoms_set):
        new_aidx = idx_map[old_aidx]
        a = molH.GetAtomWithIdx(old_aidx)
        for nb in a.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in ring_atoms_set:
                continue
            if nb.GetAtomicNum() == 1:
                # explicit hydrogen: add H atom and bond
                h_atom = Chem.Atom(1)
                h_idx = newmol.AddAtom(h_atom)
                newmol.AddBond(new_aidx, h_idx, Chem.BondType.SINGLE)
            else:
                # non-H substituent: add a single 'R' dummy atom and bond
                r_atom = Chem.Atom(0)  # dummy
                r_idx = newmol.AddAtom(r_atom)
                newmol.AddBond(new_aidx, r_idx, Chem.BondType.SINGLE)

    # Sanitize to ensure aromaticity perception for canonical SMILES
    submol = newmol.GetMol()
    Chem.SanitizeMol(submol)
    return submol


def mine_ring_patterns(smiles_list: list[str]) -> dict[str, int]:
    """Mine ring patterns from a list of SMILES strings.

    Args:
        smiles_list (list[str]): list of SMILES strings to mine patterns from.

    Returns:
        dict[str, int]: dict mapping canonical SMILES of ring patterns to their counts.
    """
    counts = defaultdict(int)

    for i, smi in tqdm(enumerate(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Skipping invalid SMILES at index {i}: {smi}")
                continue
            Chem.SanitizeMol(mol)

            # Identify ring systems
            components = get_ring_system_components(mol)
            if not components:
                continue

            for comp in components:
                submol = build_ring_pattern_submol(mol, comp)
                key = Chem.MolToSmiles(submol, canonical=True, kekuleSmiles=False)
                counts[key] += 1
        except Exception:
            # print(f"Error processing SMILES at index {i}: {smi}. Skipping.")
            continue

    return dict(counts)


def generate_3d_coords_from_patterns(patterns, num_examples=100):
    """Generate 3D coordinates for a list of SMILES patterns with dummy atoms.

    Args:
        patterns (list[str]): list of SMILES strings that may contain '*' dummies.
        num_examples (int): maximum number of examples to generate 3D coordinates for.

    Returns:
        tuple: A tuple containing:
            - mols3d (list[Chem.Mol]): molecules with '*' dummies, explicit Hs, and a 3D conformer
            - dummy_indices_list (list[list[int]]): list of dummy index lists per molecule
    """
    mols3d = []
    dummy_indices_list = []
    counter = 0
    for i, smi in enumerate(patterns):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Skipping invalid pattern at {i}: {smi}")
            continue
        mol = Chem.AddHs(mol, addCoords=False)

        dummy_idx = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
        ]

        if len(dummy_idx) == 0:
            print(f"Pattern at index {i} has no dummy atoms: {smi}")
            continue

        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                newA = Chem.Atom(1)
                newA.SetFormalCharge(atom.GetFormalCharge())
                rw.ReplaceAtom(atom.GetIdx(), newA)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)

        # Generate a 3D conformer
        params = AllChem.ETKDGv3()
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            raise RuntimeError("Embedding failed")

        # Geometry optimization
        AllChem.UFFOptimizeMolecule(mol)

        mol.SetProp("R_group_indices", json.dumps(list(map(int, dummy_idx))))

        mols3d.append(mol)
        dummy_indices_list.append(dummy_idx)
        counter += 1
        if counter >= num_examples:
            break

    return mols3d, dummy_indices_list


if __name__ == "__main__":
    start_time = datetime.now()

    # (1) Load SMILES from GEOM training data
    data_path = os.path.join(ROOT, "data", "GEOM", "raw", "train_data.pickle")
    with open(data_path, "rb") as f:
        mol_list = pickle.load(f)
    smiles_list = [smiles for smiles, _ in mol_list]

    # (2) Mine ring patterns
    counts_per_pattern = mine_ring_patterns(smiles_list)
    # print("Pattern counts:")
    # for k, v in sorted(counts_per_pattern.items(), key=lambda kv: (-kv[1], kv[0])):
    #     print(f"{k}: {v}")
    patterns, vals = zip(
        *sorted(counts_per_pattern.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    print(f"Mined {len(patterns)} unique ring patterns.")

    # (3) Generate 3D coords for top 100 patterns
    mols3d, dummy_idx = generate_3d_coords_from_patterns(patterns)
    print(f"Generated 3D coordinates for top {len(mols3d)} patterns.")

    # (4) Write 3D coords to SDF
    w = SDWriter(os.path.join(PREFIXES_DIR, "prefixes.sdf"))
    for m in mols3d:
        w.write(m)
    w.close()
    print(f"Wrote prefixes to {os.path.join(PREFIXES_DIR, 'prefixes.sdf')}.")

    # (5) Convert to 2D for visualization
    mols2d = []
    for mol, dummy_indices in zip(mols3d, dummy_idx):
        if mol is None:
            mols2d.append(None)
        else:
            rw = Chem.RWMol(mol)
            for idx in dummy_indices:
                newA = Chem.Atom(0)
                rw.ReplaceAtom(idx, newA)

            mol = rw.GetMol()
            Chem.SanitizeMol(mol)

            mol = AllChem.RemoveHs(mol)
            rdDepictor.Compute2DCoords(mol)
            mols2d.append(mol)

    img = Draw.MolsToGridImage(
        mols2d,
        molsPerRow=5,
        subImgSize=(300, 300),
    )
    img.save(os.path.join(PREFIXES_DIR, "prefixes.png"))

    end_time = datetime.now()
    print(f"Total time: {end_time - start_time}")
