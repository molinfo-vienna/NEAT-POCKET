

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path

import gdown
import networkx as nx
import numpy as np
import torch
import random
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

SEED = 0

RDKIT_BOND_TO_ID = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

LIGAND_VOCABULARY = {
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

# Standard 20 amino acids + unknown
AA_3_TO_IDX = {
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
AA_UNK_IDX = 20


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


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


class CrossDockedDataSet(InMemoryDataset):
    """CrossDocked dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            torch_geometric.data.Data object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an torch_geometric.data.Data object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in
            an torch_geometric.data.Data object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset. (default: :obj:`None`)
        split (str): One of 'train', or 'test' to specify the dataset split.
        pocket_dist_cutoff: Include a standard residue if any of its heavy atoms are
            within this distance (Å) of any ligand heavy atom.
    """

    CROSS_DOCKED_ID = "10KGuj15mxOJ2FBsduun2Lggzx0yPreEU"
    CROSS_DOCKED_SPLIT_ID = "1mycOKpphVBQjxEbpn1AwdpQs8tNVbxKY"

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        split: str = "train",
        pocket_dist_cutoff: float = 6.0,
    ):
        self.split = split
        self.pocket_dist_cutoff = pocket_dist_cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        split = split.lower()
        if split == "train":
            self.load(self.processed_paths[0])
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown split: {split}")

    def download(self) -> None:
        raw_path = os.path.join(self.root, "raw")
        os.makedirs(raw_path, exist_ok=True)
        data_file_path = os.path.join(raw_path, self.raw_file_names[0])
        split_file_path = os.path.join(raw_path, self.raw_file_names[1])
        data_extracted_path = os.path.join(raw_path, self.raw_file_names[2])

        if not os.path.exists(data_file_path):
            gdown.download(
                id=self.CROSS_DOCKED_ID,
                output=data_file_path,
                quiet=False,
            )
            print("Downloaded CrossDocked dataset.")

        if not os.path.exists(data_extracted_path):
            os.makedirs(data_extracted_path, exist_ok=True)
            with tarfile.open(data_file_path, "r:gz") as tar:
                tar.extractall(path=data_extracted_path)
            print("Extraction complete.")

        if not os.path.exists(split_file_path):
            gdown.download(
                id=self.CROSS_DOCKED_SPLIT_ID,
                output=split_file_path,
                quiet=False,
            )
            print("Downloaded CrossDocked split file.")

    @property
    def raw_file_names(self):
        return [
            "crossdocked_pocket10.tar.gz",
            "split_by_name.pt",
            "crossdocked_pocket10",
        ]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def process(self) -> None:
        split_path = self.raw_paths[1]
        try:
            data_split = torch.load(
                split_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            data_split = torch.load(split_path, map_location="cpu")

        # Take 5% of the training data for validation
        train_data = data_split["train"]
        random.shuffle(train_data)
        val_data = train_data[:int(len(train_data) * 0.05)]
        train_data = train_data[int(len(train_data) * 0.05):]
        data_split["train"] = train_data
        data_split["val"] = val_data

        datadir = Path(self.root, "raw", self.raw_file_names[2])

        split_to_path_idx = {"train": 0, "val": 1, "test": 2}
        for split_name, path_idx in split_to_path_idx.items():
            pairs = data_split.get(split_name)
            if pairs is None:
                logging.warning(
                    "Split %r not found in split file; writing empty processed split.",
                    split_name,
                )
                data_list: list[Data] = []
            else:
                data_list = self._process_split(pairs, datadir)

                if self.pre_filter is not None:
                    data_list = [d for d in data_list if self.pre_filter(d)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(d) for d in data_list]
                
                self.save(data_list, self.processed_paths[path_idx])
                print(f"Saved {len(data_list)} graphs to {self.processed_paths[path_idx]}")

    def _process_split(self, pairs, datadir: Path) -> list[Data]:
        data_list: list[Data] = []
        failed: int = 0
        pbar = tqdm(pairs)
        for pocket_fn, ligand_fn in pbar:
            pocket_path = datadir / pocket_fn
            ligand_path = datadir / ligand_fn
            try:
                data = self._process_pair(pocket_path, ligand_path)
                if data is not None:
                    data_list.append(data)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logging.debug("Skip pair %s %s: %s", pocket_fn, ligand_fn, e)
            pbar.set_postfix(failed=failed)
        if failed:
            logging.warning("Dropped %d pairs in this split.", failed)
        return data_list

    def _process_pair(self, pocket_path: Path, ligand_path: Path) -> Data | None:
        if not pocket_path.is_file() or not ligand_path.is_file():
            return None

        suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=False)
        if suppl is None or len(suppl) == 0:
            return None
        rdmol = suppl[0]
        if rdmol is None:
            return None

        rdmol = _largest_fragment(rdmol)
        if rdmol is None:
            return None

        if rdmol.GetNumAtoms() < 1:
            return None

        for atom in rdmol.GetAtoms():
            z = atom.GetAtomicNum()
            if z not in LIGAND_VOCABULARY:
                return None

        pdb_model = PDBParser(QUIET=True).get_structure("", str(pocket_path))[0]

        lig_x, lig_pos = self._ligand_features(rdmol)
        edge_index, edge_labels = self._ligand_edges(rdmol)
        if edge_index.numel() == 0:
            G = nx.Graph()
            G.add_nodes_from(range(rdmol.GetNumAtoms()))
        else:
            G = nx.Graph()
            for i, j in edge_index.t().tolist():
                G.add_edge(i, j)
        eccentricity = torch.tensor(
            [nx.eccentricity(G, n) for n in range(rdmol.GetNumAtoms())],
            dtype=torch.long,
        )

        pocket_x, pocket_pos, pocket_res_idx, pocket_res_type = self._pocket_features(
            pdb_model, lig_pos
        )
        if pocket_pos.shape[0] == 0:
            return None

        com = lig_pos.mean(dim=0, keepdim=True)
        lig_pos = lig_pos - com
        pocket_pos = pocket_pos - com

        smiles = Chem.MolToSmiles(rdmol, canonical=True)
        name = f"{pocket_path.stem}__{ligand_path.name}"

        return Data(
            x=lig_x,
            pos=lig_pos,
            edge_index=edge_index,
            edge_labels=edge_labels,
            eccentricity=eccentricity,
            smiles=smiles,
            name=name,
            pocket_x=pocket_x,
            pocket_pos=pocket_pos,
            pocket_residue_index=pocket_res_idx,
            pocket_residue_type=pocket_res_type,
        )

    def _ligand_features(self, mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
        n = mol.GetNumAtoms()
        x = torch.tensor(
            [LIGAND_VOCABULARY[a.GetAtomicNum()] for a in mol.GetAtoms()],
            dtype=torch.long,
        )
        conf = mol.GetConformer()
        pos = torch.zeros((n, 3), dtype=torch.float32)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            pos[i, 0] = p.x
            pos[i, 1] = p.y
            pos[i, 2] = p.z
        return x, pos

    def _ligand_edges(self, mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index: list[tuple[int, int]] = []
        edge_labels: list[int] = []
        for bond in mol.GetBonds():
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

    def _pocket_features(
        self, 
        pdb_model, 
        lig_pos: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cutoff = self.pocket_dist_cutoff
        lig = lig_pos.numpy()

        # (coords, atom_type_idx, aa_type_idx, residue_index)
        selected: list[tuple[np.ndarray, int, int, int]] = []
        residue_index = 0

        for chain in pdb_model.get_chains():
            for residue in chain.get_residues():
                if not is_aa(residue.get_resname(), standard=True):
                    continue
                resname = residue.get_resname().strip().upper()
                aa_idx = AA_3_TO_IDX.get(resname, AA_UNK_IDX)

                heavy = [
                    a
                    for a in residue.get_atoms()
                    if _pdb_heavy_element_symbol(a) not in (None, "H")
                ]
                if not heavy:
                    continue
                res_xyz = np.stack(
                    [np.asarray(a.get_coord(), dtype=np.float64) for a in heavy],
                    axis=0,
                )
                if (
                    np.linalg.norm(
                        res_xyz[:, None, :] - lig[None, :, :], axis=-1
                    ).min()
                    >= cutoff
                ):
                    continue

                for atom in heavy:
                    xyz = np.asarray(atom.get_coord(), dtype=np.float32)
                    sym = _pdb_heavy_element_symbol(atom)
                    enc = _encode_pocket_atom(sym, LIGAND_VOCABULARY)
                    selected.append((xyz, enc, aa_idx, residue_index))
                residue_index += 1

        if not selected:
            return (
                torch.empty(0, dtype=torch.long),
                torch.zeros(0, 3, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        pocket_pos = torch.from_numpy(np.stack([t[0] for t in selected], axis=0))
        pocket_x = torch.tensor([t[1] for t in selected], dtype=torch.long)
        pocket_residue_type = torch.tensor([t[2] for t in selected], dtype=torch.long)
        pocket_residue_index = torch.tensor([t[3] for t in selected], dtype=torch.long)

        return pocket_x, pocket_pos, pocket_residue_index, pocket_residue_type
