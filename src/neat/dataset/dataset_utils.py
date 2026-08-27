from __future__ import annotations

import logging
import torch
from rdkit import Chem


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