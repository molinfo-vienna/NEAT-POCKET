import logging
import os
import gdown
from pathlib import Path
import tarfile
from collections import defaultdict
from time import time

import networkx as nx
from Bio.PDB import PDBParser
from numpy import random
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


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
        split (str): One of 'train', 'val', or 'test' to specify the dataset split.
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
        if split == "train":
            self.load(self.processed_paths[0])
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown split: {split}")

    def download(self):
        """Download the CrossDocked dataset."""
        raw_path = os.path.join(self.root, "raw")
        os.makedirs(raw_path, exist_ok=True)
        data_file_path = os.path.join(raw_path, self.raw_paths[0])
        split_file_path = os.path.join(raw_path, self.raw_paths[1])
        data_extracted_path = os.path.join(raw_path, self.raw_paths[2])

        if not os.path.exists(data_file_path):
            gdown.download(
                id=self.CROSS_DOCKED_ID,
                output=data_file_path,
                quiet=False,
            )
            print("Downloaded CrossDocked dataset.")
            
        if not os.path.exists(data_extracted_path):
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

    def process(self):
        data_split = torch.load(self.raw_paths[1])
        datadir = Path(self.root, "raw", "crossdocked_pocket10")

        # If there is no validation set, copy training examples (the validation set
        # is not very important in this application)
        if "val" not in data_split:
            random.shuffle(data_split["train"])
            data_split["val"] = data_split["train"][-self.val_size :]
            data_split["train"] = data_split["train"][: -self.val_size]

        failed = {}
        train_smiles = []

        n_samples_after = {}
        for split in data_split.keys():

            print(f"Processing {split} dataset...")

            ligands = defaultdict(list)
            pockets = defaultdict(list)

            tic = time()
            pbar = tqdm(data_split[split])
            for pocket_fn, ligand_fn in pbar:

                pbar.set_description(f"#failed: {len(failed)}")

                sdffile = os.path.join(datadir, ligand_fn)
                pdbfile = os.path.join(datadir, pocket_fn)

                try:
                    pdb_model = PDBParser(QUIET=True).get_structure("", pdbfile)[0]
                    rdmol = Chem.SDMolSupplier(str(sdffile))[0]

                    ligand, pocket = self.process_raw_pair(
                        pdb_model,
                        rdmol,
                        pocket_representation=args.pocket,
                        compute_nerf_params=args.flex,
                        compute_bb_frames=args.flex,
                        nma_input=pdbfile if args.normal_modes else None,
                    )

                except (
                    KeyError,
                    AssertionError,
                    FileNotFoundError,
                    IndexError,
                    ValueError,
                    AttributeError,
                ) as e:
                    failed[(split, sdffile, pdbfile)] = (type(e).__name__, str(e))
                    continue

                nerf_keys = [
                    "fixed_coord",
                    "atom_mask",
                    "nerf_indices",
                    "length",
                    "theta",
                    "chi",
                    "ddihedral",
                    "chi_indices",
                ]
                for k in (
                    ["x", "one_hot", "bonds", "bond_one_hot", "v", "nma_vec"]
                    + nerf_keys
                    + ["axis_angle"]
                ):
                    if k in ligand:
                        ligands[k].append(ligand[k])
                    if k in pocket:
                        pockets[k].append(pocket[k])

                pocket_file = pdbfile.name.replace("_", "-")
                ligand_file = (
                    Path(pocket_file).stem + "_" + Path(sdffile).name.replace("_", "-")
                )
                ligands["name"].append(ligand_file)
                pockets["name"].append(pocket_file)
                train_smiles.append(rdmol_to_smiles(rdmol))

                if split in {"val", "test"}:
                    pdb_sdf_dir = processed_dir / split
                    pdb_sdf_dir.mkdir(exist_ok=True)

                    # Copy PDB file
                    pdb_file_out = Path(pdb_sdf_dir, pocket_file)
                    shutil.copy(pdbfile, pdb_file_out)

                    # Copy SDF file
                    sdf_file_out = Path(pdb_sdf_dir, ligand_file)
                    shutil.copy(sdffile, sdf_file_out)

            data = {"ligands": ligands, "pockets": pockets}
            torch.save(data, Path(processed_dir, f"{split}.pt"))

            if split == "train":
                np.save(Path(processed_dir, "train_smiles.npy"), train_smiles)

            print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")

        # --------------------------------------------------------------------------
        # Compute statistics & additional information
        # --------------------------------------------------------------------------
        train_data = torch.load(Path(processed_dir, f"train.pt"))

        # Maximum molecule size
        max_ligand_size = max([len(x) for x in train_data["ligands"]["x"]])

        # Joint histogram of number of ligand and pocket nodes
        pocket_coords = train_data["pockets"]["x"]
        ligand_coords = train_data["ligands"]["x"]
        n_nodes = get_n_nodes(ligand_coords, pocket_coords)
        np.save(Path(processed_dir, "size_distribution.npy"), n_nodes)

        # Get histograms of ligand node types
        lig_one_hot = [x.numpy() for x in train_data["ligands"]["one_hot"]]
        ligand_hist = get_type_histogram(lig_one_hot, atom_encoder)
        np.save(Path(processed_dir, "ligand_type_histogram.npy"), ligand_hist)

        # Get histograms of ligand edge types
        lig_bond_one_hot = [x.numpy() for x in train_data["ligands"]["bond_one_hot"]]
        ligand_bond_hist = get_type_histogram(lig_bond_one_hot, bond_encoder)
        np.save(Path(processed_dir, "ligand_bond_type_histogram.npy"), ligand_bond_hist)

        # Write error report
        error_str = ""
        for k, v in failed.items():
            error_str += f"{'Split':<15}:  {k[0]}\n"
            error_str += f"{'Ligand':<15}:  {k[1]}\n"
            error_str += f"{'Pocket':<15}:  {k[2]}\n"
            error_str += f"{'Error type':<15}:  {v[0]}\n"
            error_str += f"{'Error msg':<15}:  {v[1]}\n\n"

        with open(Path(processed_dir, "errors.txt"), "w") as f:
            f.write(error_str)

        metadata = {"max_ligand_size": max_ligand_size}
        with open(Path(processed_dir, "metadata.yml"), "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def process_raw_pair(
        self,
        biopython_model,
        rdmol,
        dist_cutoff=None,
        pocket_representation="side_chain_bead",
        compute_nerf_params=False,
        compute_bb_frames=False,
        nma_input=None,
        return_pocket_pdb=False,
    ):

        # Process ligand
        ligand = self.prepare_ligand(rdmol, self.atom_encoder, self.bond_encoder)

        # Find interacting pocket residues based on distance cutoff
        pocket_residues = []
        for residue in biopython_model.get_residues():

            # Remove non-standard amino acids and HETATMs
            if not is_aa(residue.get_resname(), standard=True):
                continue

            res_coords = torch.from_numpy(
                np.array([a.get_coord() for a in residue.get_atoms()])
            )
            if (
                dist_cutoff is None
                or (
                    ((res_coords[:, None, :] - ligand["x"][None, :, :]) ** 2).sum(-1)
                    ** 0.5
                ).min()
                < dist_cutoff
            ):
                pocket_residues.append(residue)

        pocket, pocket_residues = self.prepare_pocket(
            pocket_residues,
            self.aa_encoder,
            self.residue_encoder,
            self.residue_bond_encoder,
            self.pocket_representation,
            self.compute_nerf_params,
            self.compute_bb_frames,
            self.nma_input,
        )

        if return_pocket_pdb:
            builder = StructureBuilder.StructureBuilder()
            builder.init_structure("")
            builder.init_model(0)
            pocket_struct = builder.get_structure()
            for residue in pocket_residues:
                chain = residue.get_parent().get_id()

                # init chain if necessary
                if not pocket_struct[0].has_id(chain):
                    builder.init_chain(chain)

                # add residue
                pocket_struct[0][chain].add(residue)

            pocket["pocket_pdb"] = pocket_struct
        # if return_pocket_pdb:
        #     pocket['residues'] = [prepare_internal_coord(res) for res in pocket_residues]

        return ligand, pocket

    def prepare_ligand(self, rdmol, atom_encoder, bond_encoder):

        # remove H atoms if not in atom_encoder
        if "H" not in atom_encoder:
            rdmol = Chem.RemoveAllHs(rdmol, sanitize=False)

        # Coordinates
        ligand_coord = rdmol.GetConformer().GetPositions()
        ligand_coord = torch.from_numpy(ligand_coord)

        # Features
        ligand_onehot = F.one_hot(
            torch.tensor([encode_atom(a, atom_encoder) for a in rdmol.GetAtoms()]),
            num_classes=len(atom_encoder),
        )

        # Bonds
        adj = (
            np.ones((rdmol.GetNumAtoms(), rdmol.GetNumAtoms())) * bond_encoder["NOBOND"]
        )
        for b in rdmol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            adj[i, j] = bond_encoder[str(b.GetBondType())]
            adj[j, i] = adj[i, j]  # undirected graph

        # molecular graph is undirected -> don't save redundant information
        bonds = np.stack(np.triu_indices(len(ligand_coord), k=1), axis=0)
        # bonds = np.stack(np.ones_like(adj).nonzero(), axis=0)
        bond_types = adj[bonds[0], bonds[1]].astype("int64")
        bonds = torch.from_numpy(bonds)
        bond_types = F.one_hot(
            torch.from_numpy(bond_types), num_classes=len(bond_encoder)
        )

        ligand = {
            "x": ligand_coord.to(dtype=FLOAT_TYPE),
            "one_hot": ligand_onehot.to(dtype=FLOAT_TYPE),
            "mask": torch.zeros(len(ligand_coord), dtype=INT_TYPE),
            "bonds": bonds.to(INT_TYPE),
            "bond_one_hot": bond_types.to(FLOAT_TYPE),
            "bond_mask": torch.zeros(bonds.size(1), dtype=INT_TYPE),
            "size": torch.tensor([len(ligand_coord)], dtype=INT_TYPE),
            "n_bonds": torch.tensor([len(bond_types)], dtype=INT_TYPE),
        }

        return ligand

    def prepare_pocket(
        biopython_residues,
        amino_acid_encoder,
        residue_encoder,
        residue_bond_encoder,
        pocket_representation="side_chain_bead",
        compute_nerf_params=False,
        compute_bb_frames=False,
        nma_input=None,
    ):

        assert (
            nma_input is None or pocket_representation == "CA+"
        ), "vector features are only supported for CA+ pockets"

        # sort residues
        biopython_residues = sorted(
            biopython_residues, key=lambda x: (x.parent.id, x.id[1])
        )

        if nma_input is not None:
            # preprocessed normal mode eigenvectors
            if isinstance(nma_input, dict):
                nma_dict = nma_input

            # PDB file
            else:
                nma_dict = pdb_to_normal_modes(str(nma_input))

        if pocket_representation == "side_chain_bead":
            ca_coords = np.zeros((len(biopython_residues), 3))
            ca_types = np.zeros(len(biopython_residues), dtype="int64")
            side_chain_coords = []
            side_chain_aa_types = []
            edges = []  # CA-CA and CA-side_chain
            edge_types = []
            last_res_id = None
            for i, res in enumerate(biopython_residues):
                aa = amino_acid_encoder[protein_letters_3to1[res.get_resname()]]
                ca_coords[i, :] = res["CA"].get_coord()
                ca_types[i] = aa
                side_chain_coord = get_side_chain_bead_coord(res)
                if side_chain_coord is not None:
                    side_chain_coords.append(side_chain_coord)
                    side_chain_aa_types.append(aa)
                    edges.append((i, len(ca_coords) + len(side_chain_coords) - 1))
                    edge_types.append(residue_bond_encoder["CA-SS"])

                # add edges between contiguous CA atoms
                if i > 0 and res.id[1] == last_res_id + 1:
                    edges.append((i - 1, i))
                    edge_types.append(residue_bond_encoder["CA-CA"])

                last_res_id = res.id[1]

            # Coordinates
            side_chain_coords = np.stack(side_chain_coords)
            pocket_coords = np.concatenate([ca_coords, side_chain_coords], axis=0)
            pocket_coords = torch.from_numpy(pocket_coords)

            # Features
            amino_acid_onehot = F.one_hot(
                torch.cat(
                    [
                        torch.from_numpy(ca_types),
                        torch.tensor(side_chain_aa_types, dtype=torch.int64),
                    ],
                    dim=0,
                ),
                num_classes=len(amino_acid_encoder),
            )
            side_chain_onehot = np.concatenate(
                [
                    np.tile(
                        np.eye(1, len(residue_encoder), residue_encoder["CA"]),
                        [len(ca_coords), 1],
                    ),
                    np.tile(
                        np.eye(1, len(residue_encoder), residue_encoder["SS"]),
                        [len(side_chain_coords), 1],
                    ),
                ],
                axis=0,
            )
            side_chain_onehot = torch.from_numpy(side_chain_onehot)
            pocket_onehot = torch.cat([amino_acid_onehot, side_chain_onehot], dim=1)

            vector_features = None
            nma_features = None

            # Bonds
            edges = torch.tensor(edges).T
            edge_types = F.one_hot(
                torch.tensor(edge_types), num_classes=len(residue_bond_encoder)
            )

        elif pocket_representation == "CA+":
            ca_coords = np.zeros((len(biopython_residues), 3))
            ca_types = np.zeros(len(biopython_residues), dtype="int64")

            v_dim = max([x for aa in aa_atom_index.values() for x in aa.values()]) + 1
            vec_feats = np.zeros((len(biopython_residues), v_dim, 3), dtype="float32")
            nf_nma = 5
            nma_feats = np.zeros((len(biopython_residues), nf_nma, 3), dtype="float32")

            edges = []  # CA-CA and CA-side_chain
            edge_types = []
            last_res_id = None
            for i, res in enumerate(biopython_residues):
                aa = amino_acid_encoder[protein_letters_3to1[res.get_resname()]]
                ca_coords[i, :] = res["CA"].get_coord()
                ca_types[i] = aa

                vec_feats[i] = get_side_chain_vectors(res, aa_atom_index, v_dim)
                if nma_input is not None:
                    nma_feats[i] = get_normal_modes(res, nma_dict)

                # add edges between contiguous CA atoms
                if i > 0 and res.id[1] == last_res_id + 1:
                    edges.append((i - 1, i))
                    edge_types.append(residue_bond_encoder["CA-CA"])

                last_res_id = res.id[1]

            # Coordinates
            pocket_coords = torch.from_numpy(ca_coords)

            # Features
            pocket_onehot = F.one_hot(
                torch.from_numpy(ca_types), num_classes=len(amino_acid_encoder)
            )

            vector_features = torch.from_numpy(vec_feats)
            nma_features = torch.from_numpy(nma_feats)

            # Bonds
            if len(edges) < 1:
                edges = torch.empty(2, 0)
                edge_types = torch.empty(0, len(residue_bond_encoder))
            else:
                edges = torch.tensor(edges).T
                edge_types = F.one_hot(
                    torch.tensor(edge_types), num_classes=len(residue_bond_encoder)
                )

        else:
            raise NotImplementedError(
                f"Pocket representation '{pocket_representation}' not implemented"
            )

        # pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in biopython_residues]

        pocket = {
            "x": pocket_coords.to(dtype=FLOAT_TYPE),
            "one_hot": pocket_onehot.to(dtype=FLOAT_TYPE),
            # 'ids': pocket_ids,
            "size": torch.tensor([len(pocket_coords)], dtype=INT_TYPE),
            "mask": torch.zeros(len(pocket_coords), dtype=INT_TYPE),
            "bonds": edges.to(INT_TYPE),
            "bond_one_hot": edge_types.to(FLOAT_TYPE),
            "bond_mask": torch.zeros(edges.size(1), dtype=INT_TYPE),
            "n_bonds": torch.tensor([len(edge_types)], dtype=INT_TYPE),
        }

        if vector_features is not None:
            pocket["v"] = vector_features.to(dtype=FLOAT_TYPE)

        if nma_input is not None:
            pocket["nma_vec"] = nma_features.to(dtype=FLOAT_TYPE)

        if compute_nerf_params:
            nerf_params = [get_nerf_params(r) for r in biopython_residues]
            nerf_params = {
                k: torch.stack([x[k] for x in nerf_params], dim=0)
                for k in nerf_params[0].keys()
            }
            pocket.update(nerf_params)

        if compute_bb_frames:
            n_xyz = torch.from_numpy(
                np.stack([r["N"].get_coord() for r in biopython_residues])
            )
            ca_xyz = torch.from_numpy(
                np.stack([r["CA"].get_coord() for r in biopython_residues])
            )
            c_xyz = torch.from_numpy(
                np.stack([r["C"].get_coord() for r in biopython_residues])
            )
            pocket["axis_angle"], _ = get_bb_transform(n_xyz, ca_xyz, c_xyz)

        return pocket, biopython_residues
