from __future__ import annotations

import copy
import logging
import os
import random
import tarfile
from pathlib import Path

import gdown
import networkx as nx
import numpy as np
import openbabel
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from dask.distributed import Client, LocalCluster, as_completed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
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


def _add_hydrogens_with_rdkit(mol: Chem.Mol, max_attempts: int = 50) -> Chem.Mol:
    try:
        mol_h = Chem.AddHs(mol)
        AllChem.ConstrainedEmbed(mol_h, mol, maxAttempts=max_attempts)
        return mol_h
    except Exception as e:
        return None


def _add_hydrogens_with_openbabel(mol: Chem.Mol) -> Chem.Mol:

    try:
        # Convert RDKit molecule to OpenBabel molecule
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("mol")
        ob_conversion.SetOutFormat("mol")

        # Generate a temporary MOL file from RDKit molecule
        mol_block = Chem.MolToMolBlock(mol)
        ob_mol = openbabel.OBMol()
        ob_conversion.ReadString(ob_mol, mol_block)

        # Add hydrogens using OpenBabel
        ob_mol.AddHydrogens()

        # Convert OpenBabel molecule back to RDKit molecule
        updated_mol_block = ob_conversion.WriteString(ob_mol)
        updated_rdkit_mol = Chem.MolFromMolBlock(updated_mol_block, removeHs=False)

        # Return the updated RDKit molecule
        return updated_rdkit_mol

    except Exception:
        return None


def _ligand_features(mol: Chem.Mol, get_charge: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    except Exception as e:
        logger.warning(f"Ligand {mol}: cannot get edges: {e}")
        return None, None


def _pocket_features(
    pdb_model,
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


def _process_pair(
    pocket_path: Path, ligand_path: Path, add_hydrogens: bool
) -> Data | None:

    logger = logging.getLogger(__name__)

    if not pocket_path.is_file() or not ligand_path.is_file():
        return None

    ### Load ligand and clean ligand ###

    suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=False)
    if suppl is None or len(suppl) == 0:
        logger.warning(f"Ligand {ligand_path}: cannot be loaded or sanitized by RDKit.")
        return None

    rdmol = suppl[0]

    rdmol = _largest_fragment(rdmol)
    if rdmol is None:
        logger.warning(f"Ligand {ligand_path}: largest fragment is None.")
        return None

    if rdmol.GetNumAtoms() < 1:
        logger.warning(f"Ligand {ligand_path}: number of atoms is less than 1.")
        return None

    for atom in rdmol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in ATOM_VOCABULARY:
            logger.warning(
                f"Ligand {ligand_path}: atomic number {z} not in vocabulary."
            )
            return None

    if add_hydrogens:
        rdmol = Chem.AddHs(rdmol, addCoords=True)
        if rdmol is None:
            logger.warning(
                f"Ligand {ligand_path}: adding hydrogen atoms with RDKit's default method failed."
            )
            return None

    ### Process ligand into graph ###

    lig_x, lig_pos = _ligand_features(rdmol)
    if lig_x is None or lig_pos is None:
        logger.warning(f"Ligand {ligand_path}: cannot get features.")
        return None
    edge_index, edge_labels = _ligand_edges(rdmol)
    if edge_index is None or edge_labels is None:
        logger.warning(f"Ligand {ligand_path}: cannot get edges.")
        return None
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
    smiles = Chem.MolToSmiles(rdmol, canonical=True)

    ### Load pocket ###

    pdb_model = PDBParser(QUIET=True).get_structure("", str(pocket_path))[0]

    ### Process pocket into graph ###

    pocket_x, pocket_pos, pocket_residue_id, pocket_residue_type = _pocket_features(
        pdb_model
    )
    if (
        pocket_x is None
        or pocket_pos is None
        or pocket_residue_id is None
        or pocket_residue_type is None
    ):
        return None

    com = lig_pos.mean(dim=0, keepdim=True)
    lig_pos = lig_pos - com
    pocket_pos = pocket_pos - com

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
        pocket_residue_id=pocket_residue_id,
        pocket_residue_type=pocket_residue_type,
    )


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
            self.split_names = torch.load(self.raw_paths[1])["train"]
        elif split == "val":
            self.load(self.processed_paths[1])
            self.split_names = torch.load(self.raw_paths[1])["train"]
        elif split == "test":
            self.load(self.processed_paths[2])
            self.split_names = torch.load(self.raw_paths[1])["test"]
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
            with tarfile.open(data_file_path, "r:gz") as tar:
                tar.extractall(path=raw_path)
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
        return [
            "train_data.pt",
            "val_data.pt",
            "test_data.pt",
            "log.txt",
        ]

    def process(self) -> None:
        # Set up logging
        log_path = self.processed_paths[3]
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.info("Processing CrossDocked dataset...")

        # Load split file
        split_file_path = self.raw_paths[1]
        data_split = torch.load(split_file_path, map_location="cpu", weights_only=False)

        # Set aside 5% of the training data for validation
        train_data = data_split["train"]
        random.shuffle(train_data)
        val_data = train_data[: int(len(train_data) * 0.05)]
        train_data = train_data[int(len(train_data) * 0.05) :]
        data_split["train"] = train_data
        data_split["val"] = val_data

        # Load and process data
        datadir = Path(self.root, "raw", self.raw_file_names[2])
        split_to_path_idx = {"train": 0, "val": 1, "test": 2}
        for split_name, path_idx in split_to_path_idx.items():
            pairs = data_split.get(split_name)
            if pairs is None:
                self.logger.warning(
                    "Split %r not found in split file; writing empty processed split.",
                    split_name,
                )
                data_list: list[Data] = []
            else:
                self.logger.info(
                    f"Processing {split_name} split with {len(pairs)} pairs..."
                )
                data_list = self._process_split(pairs, datadir, split_name)

                if self.pre_filter is not None:
                    data_list = [d for d in data_list if self.pre_filter(d)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(d) for d in data_list]

                self.save(data_list, self.processed_paths[path_idx])
                self.logger.info(
                    f"Saved {len(data_list)} graphs to {self.processed_paths[path_idx]}"
                )

    def _process_split(
        self, pairs, datadir: Path, split_name: str, num_workers=8
    ) -> list[Data]:
        data_list: list[Data] = []
        failed: int = 0

        # if split_name == "test":
        #     add_hydrogens = False
        # else:
        #     add_hydrogens = True

        add_hydrogens = True

        if num_workers is None or num_workers <= 1:
            pbar = tqdm(pairs)
            for pocket_fn, ligand_fn in pbar:
                pocket_path = datadir / pocket_fn
                ligand_path = datadir / ligand_fn
                try:
                    data = _process_pair(pocket_path, ligand_path, add_hydrogens)
                    if data is not None:
                        data_list.append(data)
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                pbar.set_postfix(failed=failed)
            if failed:
                self.logger.warning(f"Dropped {failed} pairs in {split_name} split.")
        else:
            cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
            client = Client(cluster)
            client.forward_logging(logger_name=__name__)
            print(f"Dask cluster initialized with {num_workers} workers.")

            # 1. Submit tasks eagerly as Futures (this doesn't build a massive graph)
            # Make sure '_process_pair' is the standalone function we discussed!
            futures = [
                client.submit(
                    _process_pair,
                    datadir / pocket_fn,
                    datadir / ligand_fn,
                    add_hydrogens,
                )
                for pocket_fn, ligand_fn in pairs
            ]

            data_list = []
            failed = 0

            # 2. Track progress in real-time as tasks finish
            with tqdm(total=len(futures), desc="Processing complexes") as pbar:
                # as_completed yields futures the exact second they finish
                for future in as_completed(futures):
                    try:
                        result = future.result()  # Grab the actual output
                        if result is not None:
                            data_list.append(result)
                        else:
                            failed += 1
                    except Exception as e:
                        # This catches worker crashes or unhandled code exceptions
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix(failed=failed)

            print(f"Done! Successfully processed {len(data_list)} complexes.")
            client.close()
            cluster.close()

        return data_list

    @staticmethod
    def collate_pocket_info(pocket_list, samples_per_pocket=1, device="cpu"):
        pocket_x = torch.cat(
            [
                data_point.pocket_x.to(device).repeat(samples_per_pocket)
                for data_point in pocket_list
            ]
        )
        pocket_pos = torch.cat(
            [
                data_point.pocket_pos.to(device).repeat(samples_per_pocket, 1)
                for data_point in pocket_list
            ]
        )
        pocket_residue_id = torch.cat(
            [
                data_point.pocket_residue_id.to(device).repeat(samples_per_pocket)
                for data_point in pocket_list
            ]
        )
        pocket_residue_type = torch.cat(
            [
                data_point.pocket_residue_type.to(device).repeat(samples_per_pocket)
                for data_point in pocket_list
            ]
        )
        resets = torch.cat(
            [
                torch.tensor([False], device=device),
                pocket_residue_id[1:] < pocket_residue_id[:-1],
            ]
        )
        pocket_batch = resets.long().cumsum(dim=0).to(device)
        pocket_info = {
            "pocket_x": pocket_x,
            "pocket_pos": pocket_pos,
            "pocket_residue_id": pocket_residue_id,
            "pocket_residue_type": pocket_residue_type,
            "pocket_batch": pocket_batch,
        }
        return pocket_info

    def get_pdb_code_from_data_point(self, data_point):
        name = data_point.name.split("__")[0]
        pdb_code = name.split("_")[0].upper()
        return pdb_code

    def get_pocket_path_from_data_point(self, data_point):
        name = data_point.name.split("__")[0]
        pocket_path = [
            split_name[0] for split_name in self.split_names if name in split_name[0]
        ][0]
        pdb_file = os.path.join(self.raw_paths[2], pocket_path)
        return pdb_file

    def get_ligand_path_from_data_point(self, data_point):
        name = data_point.name.split("__")[0]
        ligand_path = [
            split_name[1] for split_name in self.split_names if name in split_name[0]
        ][0]
        sdf_file = os.path.join(self.raw_paths[2], ligand_path)
        return sdf_file
