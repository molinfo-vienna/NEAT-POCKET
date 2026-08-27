import logging
import os
import pickle
import subprocess

import networkx as nx
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

SEED = 0

from .dataset_utils import RDKIT_BOND_TO_ID, ATOM_VOCABULARY


class GEOMDataSet(InMemoryDataset):
    """GEOM dataset.

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
        split (str): One of 'train', 'val', or 'test' to specify the dataset split.
        num_conformers (int): Number of conformers to use per molecule.
    """

    GEOM_URL = "https://bits.csb.pitt.edu/files/geom_raw"
    VOCABULARY = ATOM_VOCABULARY
    NUM_CONFORMERS = 5

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        split="train",
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        if split == "train":
            self.load(self.processed_paths[0])
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown split: {split}")

    def download(self):
        """Download the GEOM dataset if it doesn't exist already."""
        raw_path = os.path.join(self.root, "raw")
        os.makedirs(raw_path, exist_ok=True)

        for raw_file in self.raw_file_names:
            if not os.path.exists(os.path.join(raw_path, raw_file)):
                subprocess.run(
                    [
                        "wget",
                        "-r",
                        "-np",
                        "-nH",
                        "--cut-dirs=2",
                        "--reject",
                        "index.html*",
                        "-P",
                        raw_path,
                        os.path.join(self.GEOM_URL, raw_file),
                    ],
                    check=True,
                )

                print("Downloaded GEOM dataset.")

    @property
    def raw_file_names(self):
        return ["train_data.pickle", "val_data.pickle", "test_data.pickle"]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def process(self):
        """Process the raw GEOM dataset files."""
        for i, raw_path in enumerate(self.raw_paths):
            raw_path = self.raw_paths[i]
            with open(raw_path, "rb") as f:
                mol_list = pickle.load(f)

            print(f"Processing {len(mol_list)} molecules from {raw_path}...")

            data_list = []
            for smiles, conformers in tqdm(mol_list):
                mol_data = self.process_molecule(
                    smiles,
                    conformers,
                    self.VOCABULARY,
                    num_conformers=self.NUM_CONFORMERS,
                )
                if mol_data is not None:
                    data_list.extend(mol_data)

            data_list = [data for data in data_list if data is not None]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            self.save(data_list, self.processed_paths[i])

    def largest_fragment_by_size(
        self, mol: Chem.Mol, use_heavy_atoms: bool = True, sanitize_frags: bool = True
    ):
        """Return the largest fragment of a molecule by number of atoms.

        Args:
            mol: RDKit molecule.
            use_heavy_atoms: count only non-hydrogen atoms if True, else count all atoms.
            sanitize_frags: sanitize the fragment molecules.

        Returns:
            The largest fragment as an RDKit molecule.
        """
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=sanitize_frags)
        if not frags:
            return None
        key = (
            (lambda m: m.GetNumHeavyAtoms())
            if use_heavy_atoms
            else (lambda m: m.GetNumAtoms())
        )
        return max(frags, key=key)

    def process_molecule(
        self, smiles: str, conformers: list, vocabulary: dict, num_conformers: int = 5
    ) -> list[Data]:
        """Process a single molecule and its conformers into a list of Data objects.

        Args:
            smiles: SMILES string of the molecule.
            conformers: List of RDKit molecule conformers.
            vocabulary: Mapping from atomic numbers to indices.
            num_conformers: Number of conformers to process.

        Returns:
            List of Data objects for the conformers of the molecule.
        """
        try:
            conformer_list = []
            for mol in conformers:
                if len(conformer_list) >= num_conformers:
                    break

                mol = self.largest_fragment_by_size(mol, use_heavy_atoms=True)

                if mol is None:
                    logging.warning(
                        f"RDKit molecule is None for {smiles}, skipping conformer."
                    )
                    continue

                # Create a tensor for atomic numbers and hybridization states
                x = torch.tensor(
                    [vocabulary[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
                    dtype=torch.long,
                )
                charge = torch.tensor(
                    [a.GetFormalCharge() for a in mol.GetAtoms()],
                    dtype=torch.float32,
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
                    [eccentricities[node] for node in range(len(G.nodes))],
                    dtype=torch.long,
                )

                data = Data(
                    x=x,
                    pos=pos,
                    edge_index=edge_index,
                    edge_labels=edge_labels,
                    eccentricity=eccentricity_tensor,
                    smiles=smiles,
                    charge=charge,
                )

                conformer_list.append(data)

            return conformer_list

        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return None
