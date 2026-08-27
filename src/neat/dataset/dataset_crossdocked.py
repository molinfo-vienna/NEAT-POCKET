from __future__ import annotations

import logging
import os
from pathlib import Path
import random
import tarfile

from Bio.PDB import PDBParser
from dask.distributed import Client, LocalCluster, as_completed
import gdown
import networkx as nx
from rdkit import Chem, RDLogger
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .dataset_utils import (
    ATOM_VOCABULARY, 
    _get_pocket_features_from_biopython_model,
    _largest_fragment,
    _ligand_edges,
    _ligand_features,
)

RDLogger.DisableLog("rdApp.*")
SEED = 0


def _process_protein_ligand_complex(
    pocket_path: Path, ligand_path: Path, add_hydrogens: bool
) -> Data | None:
    """Construct PyG Data object from raw protein-ligand complex

    Args:
        pocket_path (Path): Path to the pocket PDB file.
        ligand_path (Path): Path to the ligand SDF file.
        add_hydrogens (bool): Whether to add hydrogen atoms.

    Returns:
        Data | None: The constructed PyG Data object or None if construction failed.
    """

    logger = logging.getLogger(__name__)

    if not pocket_path.is_file() or not ligand_path.is_file():
        return None

    # (1) Load ligand and do sanitization checks
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

    # (2) Add hydrogens with RDKit's default method
    if add_hydrogens:
        rdmol = Chem.AddHs(rdmol, addCoords=True)
        if rdmol is None:
            logger.warning(
                f"Ligand {ligand_path}: adding hydrogen atoms with RDKit's default method failed."
            )
            return None

    # (3) Construct ligand graph object
    lig_x, lig_pos = _ligand_features(rdmol, get_charge=False)
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

    # (4) Load pocket and construct point cloud object
    pdb_model = PDBParser(QUIET=True).get_structure("", str(pocket_path))[0]
    pocket_x, pocket_pos, pocket_residue_id, pocket_residue_type = _get_pocket_features_from_biopython_model(
        pdb_model
    )
    if (
        pocket_x is None
        or pocket_pos is None
        or pocket_residue_id is None
        or pocket_residue_type is None
    ):
        return None

    # (5) Center both to the ligand's COM, and store the complex name
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
    ):
        self.split = split
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
        """Parallelized dataset processing using Dask. Sequential fallback if num_workers <= 1."""
        data_list: list[Data] = []
        failed: int = 0

        add_hydrogens = True

        if num_workers is None or num_workers <= 1:
            pbar = tqdm(pairs)
            for pocket_fn, ligand_fn in pbar:
                pocket_path = datadir / pocket_fn
                ligand_path = datadir / ligand_fn
                try:
                    data = _process_protein_ligand_complex(pocket_path, ligand_path, add_hydrogens)
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

            # 1. Submit tasks as futures 
            futures = [
                client.submit(
                    _process_protein_ligand_complex,
                    datadir / pocket_fn,
                    datadir / ligand_fn,
                    add_hydrogens,
                )
                for pocket_fn, ligand_fn in pairs
            ]

            data_list = []
            failed = 0

            # 2. Track progress as tasks finish
            with tqdm(total=len(futures), desc="Processing complexes") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()  
                        if result is not None:
                            data_list.append(result)
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix(failed=failed)

            print(f"Done! Successfully processed {len(data_list)} complexes.")
            client.close()
            cluster.close()

        return data_list

    @staticmethod
    def collate_pocket_info(pocket_list, samples_per_pocket=1, device="cpu"):
        """PyG style batching of a list of pocket objects.

        Args:
            pocket_list (list): List of PyG data objects representing pockets.
            samples_per_pocket (int, optional): Number of repeats per pocket. Defaults to 1.
            device (str, optional): Device to move the tensors to. Defaults to "cpu".

        Returns:
            dict: Collated pocket information.
        """
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
