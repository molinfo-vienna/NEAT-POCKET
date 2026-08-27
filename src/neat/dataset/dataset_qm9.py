import logging
import os
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .dataset_utils import RDKIT_BOND_TO_ID

RDLogger.DisableLog("rdApp.*")
SEED = 0


class QM9DataSet(InMemoryDataset):
    """QM9 dataset.

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
    """

    VOCABULARY = {
        1: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
    }
    QM9_URL = "https://ndownloader.figshare.com/files/3195389"
    UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dsgdb9nsd.xyz.tar.bz2", "uncharacterized.txt"]

    @property
    def processed_file_names(self):
        return ["qm9.pt"]

    def download(self):
        raw_path = os.path.join(self.root, "raw")
        qm9_path = os.path.join(raw_path, self.raw_file_names[0])
        xyz_path = os.path.join(raw_path, "xyz_files")
        unchar_path = os.path.join(raw_path, self.raw_file_names[1])

        if not os.path.exists(unchar_path):
            subprocess.run(
                [
                    "wget",
                    "-O",
                    unchar_path,
                    self.UNCHARACTERIZED_URL,
                ],
                check=True,
            )
            print("Downloaded uncharacterized.txt file.")

        if not os.path.exists(qm9_path):
            subprocess.run(
                [
                    "wget",
                    "-O",
                    qm9_path,
                    self.QM9_URL,
                ],
                check=True,
            )
            print("Downloaded QM9 dataset.")

            with tarfile.open(qm9_path, mode="r:*") as tar:
                # Safe extraction to avoid path traversal
                def is_within_directory(directory, target):
                    directory = Path(directory).resolve()
                    target = Path(target).resolve()
                    return str(target).startswith(str(directory))

                for member in tar.getmembers():
                    target_path = Path(xyz_path) / member.name
                    if not is_within_directory(Path(xyz_path), target_path):
                        raise RuntimeError(f"Blocked unsafe path: {member.name}")
                tar.extractall(path=Path(xyz_path))
            print("Extracted QM9 dataset.")

    @staticmethod
    def normalize_numeric_token(tok: str) -> str:
        """
        Normalize nonstandard scientific notation to something float() accepts.

        Examples:
        '2.1997*^-6' -> '2.1997e-6'
        '+3.0*^2'    -> '+3.0e2'
        '2.0×10^-3'  -> '2.0e-3'

        Args:
            tok: string token representing a number.

        Returns:
            Normalized string token.
        """
        s = tok.strip()
        s = s.rstrip(",;")
        if "*^" in s:
            s = s.replace("*^", "e")
        s = re.sub(r"[×x]\s*10\^([+-]?\d+)", r"e\1", s)
        s = s.replace(" ", "")
        return s

    def safe_float(self, tok: str) -> float:
        s = self.normalize_numeric_token(tok)
        try:
            return float(s)
        except Exception as e:
            raise ValueError(
                f"Cannot parse float from token {tok!r} (normalized {s!r})"
            ) from e

    def parse_xyz_atoms(
        self, lines: List[str]
    ) -> Tuple[int, List[Tuple[str, float, float, float]], str]:
        """
        Parse the XYZ atom section.
        Returns (n_atoms, atoms, name_line) where atoms is a list of (symbol, x, y, z).
        Assumes:
        - Line 0: integer atom count
        - Line 1: comment/metadata (returned as name_line)
        - Lines 2..(2+n-1): atom lines: symbol x y z [extras...]
        Coordinates are normalized to handle tokens like '2.1997*^-6'.
        """
        if not lines:
            raise ValueError("Empty file")

        try:
            n = int(lines[0].strip())
        except Exception as e:
            raise ValueError(f"Could not read atom count from first line: {e}")

        name_line = lines[1].rstrip("\n") if len(lines) > 1 else ""

        atom_lines = lines[2 : 2 + n]
        if len(atom_lines) < n:
            raise ValueError(f"Expected {n} atom lines, found {len(atom_lines)}")

        atoms = []
        for i, line in enumerate(atom_lines, start=1):
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 4:
                raise ValueError(f"Atom line {i} malformed: {line!r}")
            symbol = parts[0]
            try:
                x = self.safe_float(parts[1])
                y = self.safe_float(parts[2])
                z = self.safe_float(parts[3])
            except Exception as e:
                raise ValueError(
                    f"Coordinates not parseable on atom line {i}: {line!r}"
                ) from e
            atoms.append((symbol, x, y, z))
        return n, atoms, name_line

    def build_rdkit_mol_from_atoms(
        self, n: int, atoms: List[Tuple[str, float, float, float]], name: str
    ) -> Chem.Mol:
        """
        Build an RDKit molecule with a conformer from a list of atoms and coordinates using an XYZ block,
        then infer bonds with rdDetermineBonds and sanitize the molecule.
        """
        xyz_block = [str(n), name]
        xyz_block += [f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, x, y, z in atoms]
        xyz_block = "\n".join(xyz_block)

        mol = Chem.rdmolfiles.MolFromXYZBlock(xyz_block)
        if mol is None:
            raise ValueError("RDKit failed to create Mol from XYZ block")

        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)
        return mol

    @staticmethod
    def extract_id_from_filename(path: str) -> Optional[int]:
        """Extract integer ID from filenames like dsgdb9nsd_133885.xyz."""
        m = re.search(r"_(\d+)\.xyz$", os.path.basename(path))
        return int(m.group(1)) if m else None

    @staticmethod
    def parse_uncharacterized_ids(txt_path: str) -> Set[int]:
        """
        Parse uncharacterized.txt and return a set of indices (ints) to exclude.
        """
        exclude = set()
        if not os.path.isfile(txt_path):
            return exclude
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Skip headers and separator lines
                if (
                    s.startswith("=")
                    or s.startswith("#")
                    or s.lower().startswith("list of")
                    or "smiles" in s.lower()
                ):
                    continue
                # Capture a leading integer (possibly with padding)
                m = re.match(r"^(\d+)", s)
                if not m:
                    # Sometimes the index is right-aligned: capture first integer anywhere
                    m = re.search(r"(^|\s)(\d+)", s)
                    if m:
                        idx = int(m.group(2))
                        exclude.add(idx)
                    continue
                idx = int(m.group(1))
                exclude.add(idx)
        return exclude

    def parse_xyz_file(self, path: str) -> Dict:
        """Parse a single .xyz file and return a record dict."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        n, atoms, name_line = self.parse_xyz_atoms(lines)
        mol = self.build_rdkit_mol_from_atoms(
            n, atoms, name_line or os.path.basename(path)
        )

        return mol

    def process(self):
        # Load uncharacterized IDs to exclude (3054 molecules)
        exclude_set = self.parse_uncharacterized_ids(self.raw_paths[1])

        data_list = []

        # Parse all .xyz files
        for root, _, files in os.walk(os.path.join(self.root, "raw", "xyz_files")):
            for file in tqdm(files):
                if file.lower().endswith(".xyz"):
                    path = os.path.join(root, file)
                    try:
                        mol_id = self.extract_id_from_filename(path)
                        if mol_id in exclude_set:
                            continue
                        mol = self.parse_xyz_file(path)
                        data = self.process_molecule(mol, mol_id)
                        if data is not None:
                            data_list.append(data)
                    except Exception as e:
                        logging.warning(f"Could not process {path}: {e}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def process_molecule(self, mol, mol_id):
        try:
            if len(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)) > 1:
                logging.warning(f"Molecule {mol_id} has multiple fragments; skipping.")
                return None

            # Canonical SMILES
            smiles = Chem.MolToSmiles(mol, canonical=True)

            # Atomic number OHE
            x = torch.tensor(
                [self.VOCABULARY[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
                dtype=torch.long,
            )

            # 3D coordinates centered at origin
            conf = mol.GetConformer()
            n = mol.GetNumAtoms()
            pos = torch.zeros((n, 3), dtype=torch.float32)
            for i in range(n):
                p = conf.GetAtomPosition(i)  # returns RDKit Point3D
                pos[i, 0] = p.x
                pos[i, 1] = p.y
                pos[i, 2] = p.z
            pos = pos - pos.mean(dim=0, keepdim=True)

            # Bond information for data augmentation during training
            # Note that the generation does not use bond information
            edge_index = []
            edge_labels = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append((i, j))
                edge_index.append((j, i))
                bt = RDKIT_BOND_TO_ID.get(bond.GetBondType(), 0)
                edge_labels.append(bt)
                edge_labels.append(bt)  # one label per directed edge

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_labels = torch.tensor(edge_labels, dtype=torch.long)

            # Initialize molecular graph
            G = nx.Graph()
            for i, j in edge_index.t().tolist():
                G.add_edge(i, j)

            # Compute eccentricities for all nodes
            eccentricities = nx.eccentricity(G)
            eccentricity_tensor = torch.tensor(
                [eccentricities[node] for node in range(len(G.nodes))], dtype=torch.long
            )

            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_labels=edge_labels,
                eccentricity=eccentricity_tensor,
                smiles=smiles,
            )

            return data

        except Exception as e:
            logging.error(f"Error processing molecule {mol_id}: {e}")
            return None

    def get_splits(
        self,
        n_train: int = 100000,
        test_frac: float = 0.10,
    ) -> Dict[str, np.ndarray]:
        """
        Create shuffled train/val/test splits of index integers.

        Parameters
        ----------
        n_train : int
            Target number of training examples (capped at available data minus test set).
        test_frac : float
            Fraction of the total reserved for the test set (0 <= test_frac < 1).

        Returns
        -------
        Dict[str, np.ndarray]
            Arrays of indices for "train", "val", and "test".
        """
        n_total = len(self)
        if not (0 <= test_frac < 1):
            raise ValueError("test_frac must be in [0, 1).")

        # Compute sizes with safeguards
        n_test = int(round(test_frac * n_total))
        n_train = min(n_train, max(0, n_total - n_test))
        n_val = n_total - n_train - n_test

        # Permute indices
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_total)

        # Slice indices
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        return {"train": train_idx, "val": val_idx, "test": test_idx}
