from __future__ import annotations

from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Model import Model
import logging
import numpy as np
from rdkit import Chem
import torch


RDKIT_BOND_TO_ID = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

ATOM_VOCABULARY = {
    1: 1,  # H
    5: 2,  # B
    6: 3,  # C
    7: 4,  # N
    8: 5,  # O
    9: 6,  # F
    13: 7,  # Al
    14: 8,  # Si
    15: 9,  # P
    16: 10,  # S
    17: 11,  # Cl
    33: 12,  # As
    35: 13,  # Br
    53: 14,  # I
    80: 15,  # Hg
    83: 16,  # Bi
}

# Standard 20 amino acids + unknown
AA_VOCABULARY = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}
AA_VOCABULARY_UNK = 20


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())

def _ligand_features(
    mol: Chem.Mol, get_charge: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger = logging.getLogger(__name__)
    try:
        n = mol.GetNumAtoms()
        x = torch.tensor(
            [ATOM_VOCABULARY[a.GetAtomicNum()] for a in mol.GetAtoms()],
            dtype=torch.long,
        )
        conf = mol.GetConformer()
        pos = torch.zeros((n, 3), dtype=torch.float32)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            pos[i, 0] = p.x
            pos[i, 1] = p.y
            pos[i, 2] = p.z
        if get_charge:
            charge = torch.tensor(
                [a.GetFormalCharge() for a in mol.GetAtoms()],
                dtype=torch.float32,
            )
            return x, pos, charge
        else:
            return x, pos
    except Exception as e:
        logger.warning(f"Ligand {mol}: cannot get features: {e}")
        if get_charge:
            return None, None, None
        else:
            return None, None
        
def _ligand_edges(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    logger = logging.getLogger(__name__)
    try:
        # 1. Create a copy to prevent in-place mutation of the input molecule
        mol_kekule = Chem.Mol(mol)

        # 2. Convert aromatic bonds to explicit single/double bonds
        Chem.Kekulize(mol_kekule, clearAromaticFlags=True)

        edge_index: list[tuple[int, int]] = []
        edge_labels: list[int] = []

        # 3. Iterate over the kekulized copy
        for bond in mol_kekule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_index.append((i, j))
            edge_index.append((j, i))

            bt = RDKIT_BOND_TO_ID.get(bond.GetBondType(), 0)
            edge_labels.append(bt)
            edge_labels.append(bt)

        if not edge_index:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_labels_t = torch.tensor(edge_labels, dtype=torch.long)

        return edge_index_t, edge_labels_t

    except Exception as e:
        logger.warning(f"Ligand {mol}: cannot get edges: {e}")
        return None, None
    
def _pdb_heavy_element_symbol(atom) -> str | None:
    e = (getattr(atom, "element", None) or "").strip().upper()
    if e:
        if len(e) > 1:
            return e
        return e
    name = atom.get_name().strip()
    if not name:
        return None
    if len(name) >= 2 and name[:2].upper() in ("FE", "ZN", "MG", "CA", "MN", "CO"):
        return name[:2].upper()
    c0 = name[0].upper()
    if c0 in "CNOSHP":
        return c0
    return None


def _encode_pocket_atom(symbol: str | None, vocabulary: dict[int, int]) -> int:
    if symbol is None or symbol.upper() == "H":
        return 0
    sym = symbol.strip()
    pt = Chem.GetPeriodicTable()
    try:
        if len(sym) == 1:
            n = pt.GetAtomicNumber(sym.upper())
        else:
            n = pt.GetAtomicNumber(sym[:1].upper() + sym[1:].lower())
    except Exception:
        return 0
    return int(vocabulary.get(n, 0))


def _get_pocket_features_from_biopython_model(
    pdb_model: Model,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # (atom_type, atom_coords, residue_type, residue_id)
    selected: list[tuple[int, np.ndarray, int, int]] = []
    residue_id = 0
    logger = logging.getLogger(__name__)
    try:
        for chain in pdb_model.get_chains():
            for residue in chain.get_residues():
                if not is_aa(residue.get_resname(), standard=True):
                    continue
                resname = residue.get_resname().strip().upper()
                residue_type = AA_VOCABULARY.get(resname, AA_VOCABULARY_UNK)

                heavy = [
                    a
                    for a in residue.get_atoms()
                    if _pdb_heavy_element_symbol(a) not in (None, "H")
                ]
                if not heavy:
                    continue

                for atom in heavy:
                    atom_type = _pdb_heavy_element_symbol(atom)
                    atom_type_encoded = _encode_pocket_atom(atom_type, ATOM_VOCABULARY)
                    atom_coords = np.asarray(atom.get_coord(), dtype=np.float32)
                    selected.append(
                        (atom_type_encoded, atom_coords, residue_id, residue_type)
                    )
                residue_id += 1

        if not selected:
            logger.warning(f"Pocket {pdb_model}: no atoms selected.")
            return None, None, None, None

        pocket_x = torch.tensor([t[0] for t in selected], dtype=torch.long)
        pocket_pos = torch.from_numpy(np.stack([t[1] for t in selected], axis=0))
        pocket_residue_id = torch.tensor([t[2] for t in selected], dtype=torch.long)
        pocket_residue_type = torch.tensor([t[3] for t in selected], dtype=torch.long)

        return pocket_x, pocket_pos, pocket_residue_id, pocket_residue_type

    except Exception as e:
        logger.warning(f"Pocket {pdb_model}: cannot get features: {e}")
        return None, None, None, None