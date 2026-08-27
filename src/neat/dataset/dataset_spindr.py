from __future__ import annotations

import logging
import os
import subprocess
import tarfile
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import networkx as nx
import torch
from dask.distributed import Client, LocalCluster, as_completed
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .dataset_utils import (AA_VOCABULARY, ATOM_VOCABULARY,
                                  _largest_fragment, _ligand_edges,
                                  _ligand_features)

RDLogger.DisableLog("rdApp.*")
SEED = 0


def _process_protein_ligand_complex(pocket_path: Path, ligand_path: Path, split_name: str) -> Data | None:

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

    ### Process ligand into graph ###
    lig_x, lig_pos, lig_charge = _ligand_features(rdmol, get_charge=True)
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
    file = pdbx.CIFFile.read(str(pocket_path))
    cif_model = pdbx.get_structure(file, model=1)
    pt = Chem.GetPeriodicTable()
    pocket_x = torch.tensor(
        [
            ATOM_VOCABULARY.get(pt.GetAtomicNumber(element))
            for element in cif_model.element
        ],
        dtype=torch.long,
    )
    pocket_pos = torch.tensor(cif_model.coord)
    pocket_residue_id = torch.tensor(cif_model.res_id, dtype=torch.long)
    _, pocket_residue_id = torch.unique(pocket_residue_id, return_inverse=True)
    pocket_residue_type = torch.tensor(
        [AA_VOCABULARY.get(residue_type, 0) for residue_type in cif_model.res_name],
        dtype=torch.long,
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

    name = f"{pocket_path.stem}"

    return Data(
        x=lig_x,
        pos=lig_pos,
        charge=lig_charge,
        edge_index=edge_index,
        edge_labels=edge_labels,
        eccentricity=eccentricity,
        smiles=smiles,
        name=name,
        pocket_x=pocket_x,
        pocket_pos=pocket_pos,
        pocket_residue_id=pocket_residue_id,
        pocket_residue_type=pocket_residue_type,
        split=split_name,
    )


class SpindrDataSet(InMemoryDataset):
    """Spindr dataset.

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
    """

    SPINDR_ZENODO = "https://zenodo.org/records/15991057/files/spindr.tar.gz?download=1"

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
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown split: {split}")

    def download(self) -> None:
        target_dir = self.root
        file_name = "spindr.tar.gz"
        output_path = os.path.join(target_dir, file_name)
        command = ["wget", "-O", output_path, self.SPINDR_ZENODO]

        if not os.path.exists(Path(target_dir) / "raw" / "train"):
            try:
                print(f"Downloading file to {output_path}...")
                subprocess.run(command, check=True)
                print("Download completed successfully! Starting extracting...")

                with tarfile.open(output_path, "r:gz") as tar:
                    tar.extractall(path=self.root, filter="data")

                print("Extraction completed successfully!")

            except subprocess.CalledProcessError as e:
                print(f"Error occurred during download: {e}")
            except FileNotFoundError:
                print(f"Error: 'wget' is not installed or not in your system's PATH.")

        else:
            print(
                f"Raw directory already exists at {Path(target_dir) / 'raw'}. Skipping download and extraction."
            )

    @property
    def raw_file_names(self):
        return [
            "train",
            "val",
            "test",
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
        raw_folder = Path(self.root) / "raw"

        # Load and process data
        for i, split_name in enumerate(self.raw_file_names):
            raw_split_path = raw_folder / split_name
            complex_names = list(
                {file.stem for file in raw_split_path.iterdir() if file.is_file()}
            )
            pairs = [(f"{name}.cif", f"{name}.sdf") for name in complex_names]
            self.logger.info(
                f"Processing {split_name} split with {len(pairs)} pairs..."
            )
            data_list = self._process_split(pairs, raw_split_path, split_name)

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[i])
            self.logger.info(
                f"Saved {len(data_list)} graphs to {self.processed_paths[i]} for {split_name} split."
            )

    def _process_split(
        self, pairs, datadir: Path, split_name: str, num_workers=8
    ) -> list[Data]:
        data_list: list[Data] = []
        failed: int = 0

        if num_workers is None or num_workers <= 1:
            pbar = tqdm(pairs)
            for pocket_fn, ligand_fn in pbar:
                pocket_path = datadir / pocket_fn
                ligand_path = datadir / ligand_fn
                try:
                    data = _process_protein_ligand_complex(pocket_path, ligand_path, split_name)
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
                    _process_protein_ligand_complex,
                    datadir / pocket_fn,
                    datadir / ligand_fn,
                    split_name,
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
        cif_file = os.path.join(
            self.root, "raw", data_point.split, data_point.name + ".cif"
        )
        return cif_file

    def get_ligand_path_from_data_point(self, data_point):
        sdf_file = os.path.join(
            self.root, "raw", data_point.split, data_point.name + ".sdf"
        )
        return sdf_file
