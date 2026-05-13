import glob
import logging
import os
from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import Mol, MolToSmiles, rdDetermineBonds, rdmolfiles
from tqdm import tqdm

from neat.model import BondPredictor

# Bond type mapping: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
RDKIT_BOND_TYPES = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


class MoleculeBuilder:
    """Build RDKit molecules from tensors.

    Args:
        vocab (str): The vocabulary to use. Options are "QM9" and "GEOM".
    """

    def __init__(self, vocab="QM9") -> None:
        super().__init__()
        self.vocab = vocab
        if vocab == "QM9":
            self.atom_type_to_element = {
                1: "H",
                2: "C",
                3: "N",
                4: "O",
                5: "F",
            }
        elif vocab == "GEOM" or vocab == "CrossDocked":
            self.atom_type_to_element = {
                1: "H",
                2: "B",
                3: "C",
                4: "N",
                5: "O",
                6: "F",
                7: "Al",
                8: "Si",
                9: "P",
                10: "S",
                11: "Cl",
                12: "As",
                13: "Br",
                14: "I",
                15: "Hg",
                16: "Bi",
            }
        else:
            raise ValueError(f"Unsupported vocabulary: {vocab}")

        # Atomic numbers for RDKit (element symbol -> atomic number)
        pt = Chem.GetPeriodicTable()
        self.atom_type_to_atomic_num = {
            k: pt.GetAtomicNumber(v) for k, v in self.atom_type_to_element.items()
        }
        # Precomputed tensor for fast lookup (used in bond predictor path)
        max_atom_type = max(self.atom_type_to_atomic_num.keys())
        self._atomic_num_lookup = torch.zeros(max_atom_type + 1, dtype=torch.long)
        for k, v in self.atom_type_to_atomic_num.items():
            self._atomic_num_lookup[k] = v
        self._atomic_num_lookup[0] = 1  # fallback for unknown

    def load_tensor_from_file(
        self, files_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load atom types, positions, and batch indices from a generated_mols.pt file.

        Args:
            files_path (str): The path to the generated_mols.pt file.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the atom types, positions, and batch indices.
        """
        generated_mols = (
            torch.load(
                os.path.join(files_path, "generated_mols.pt"), weights_only=False
            )
            .detach()
            .cpu()
        )
        return generated_mols.x, generated_mols.pos, generated_mols.batch

    def create_xyz_block(self, x: torch.Tensor, pos: torch.Tensor) -> str:
        """Create an XYZ block from a tensor of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).

        Returns:
            str: An XYZ block.
        """
        xyz_lines = []
        num_atoms = x.size(0)
        xyz_lines.append(f"{num_atoms}")
        xyz_lines.append("")

        for i in range(num_atoms):
            atom_type = x[i].item()
            element = self.atom_type_to_element.get(atom_type, "X")
            x_coord, y_coord, z_coord = pos[i].tolist()
            xyz_lines.append(f"{element}\t{x_coord:.4f}\t{y_coord:.4f}\t{z_coord:.4f}")

        return "\n".join(xyz_lines)

    def generate_rdkit_molecules_via_xyz2mol(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        progress_bar: bool = False,
        break_after_k_mols: int = None,
    ) -> list[Mol]:
        """Generate RDKit molecules from tensors of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).
            batch (torch.Tensor): A tensor of shape (n_atoms,).

        Returns:
            list[Mol]: A list of RDKit molecules.
        """
        mols = []
        unique_batches = batch.unique().tolist()

        iterator = unique_batches
        if progress_bar:
            iterator = tqdm(unique_batches, desc="Generating RDKit molecules")
        for batch_id in iterator:
            mask = batch == batch_id
            x_mol = x[mask]
            pos_mol = pos[mask]

            xyz_block = self.create_xyz_block(x_mol, pos_mol)
            mol = rdmolfiles.MolFromXYZBlock(xyz_block)
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0, maxIterations=100000)
            except Exception as e:
                logging.warning(
                    f"An error occurred while determining bonds for molecule in batch {batch_id}: {e}"
                )
                mol = None

            mols.append(mol)

            if break_after_k_mols is not None and len(mols) >= break_after_k_mols:
                break

        return mols

    def generate_rdkit_molecules_via_bond_predictor(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        bond_predictor_path: str,
        progress_bar: bool = False,
        break_after_k_mols: Optional[int] = None,
    ) -> list[Mol]:
        """Generate RDKit molecules from tensors using the bond predictor.

        Args:
            x: Atom types [n_atoms].
            pos: Coordinates [n_atoms, 3].
            batch: Batch indices [n_atoms].
            bond_predictor_path: Path to bond predictor checkpoint or directory
                containing bond_predictor_<dataset>.ckpt.
            progress_bar: Whether to show a progress bar.
            break_after_k_mols: Stop after generating this many molecules.

        Returns:
            list[Mol]: List of RDKit molecules (None for failed conversions).
        """
        ckpt_path = bond_predictor_path
        if os.path.isdir(ckpt_path):
            # Look for *.ckpt
            pattern = os.path.join(ckpt_path, f"*.ckpt")
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(
                    f"No bond predictor checkpoint found in {ckpt_path}."
                )
            ckpt_path = matches[0]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Bond predictor checkpoint not found: {ckpt_path}")

        bond_predictor = BondPredictor.load_from_checkpoint(ckpt_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, pos, batch = x.to(device), pos.to(device), batch.to(device)
        bond_predictor = bond_predictor.to(device).eval()

        with torch.no_grad():
            # predict_bonds returns (bond_types, pair_indices) for radius-graph edges
            bond_types, pair_indices = bond_predictor.predict_bonds(
                x,
                pos,
                batch,
                device,
                radius=getattr(bond_predictor.hparams, "radius", 2.5),
            )

        bond_types = bond_types.cpu()
        pair_indices = pair_indices.cpu()
        x, pos, batch = x.cpu(), pos.cpu(), batch.cpu()

        unique_batches, batch_counts = batch.unique(return_counts=True)
        local_idx = torch.zeros(x.shape[0], dtype=torch.long)
        for m, b in enumerate(unique_batches):
            mask = batch == b
            local_idx[mask] = torch.arange(batch_counts[m].item(), dtype=torch.long)
        atomic_nums = self._atomic_num_lookup[
            x.clamp(0, self._atomic_num_lookup.shape[0] - 1)
        ]

        mols = []
        num_mols = len(unique_batches)
        iterator = range(num_mols)
        if progress_bar:
            iterator = tqdm(iterator, desc="Building molecules (bond predictor)")

        for m in iterator:
            b = unique_batches[m].item()
            mol_mask = batch == b
            mol_atoms = torch.where(mol_mask)[0]
            n = mol_atoms.shape[0]

            # Edges with both endpoints in this molecule and bond_type > 0
            edge_mask = (batch[pair_indices[:, 0]] == b) & (
                batch[pair_indices[:, 1]] == b
            )
            edge_mask = edge_mask & (bond_types > 0)
            if not edge_mask.any():
                bonded_pairs = pair_indices.new_empty(0, 2)
                bonded_types = bond_types.new_empty(0)
            else:
                bonded_pairs = pair_indices[edge_mask]
                bonded_types = bond_types[edge_mask]
                # Deduplicate: radius graph has both (i,j) and (j,i); keep only i < j
                keep = bonded_pairs[:, 0] < bonded_pairs[:, 1]
                bonded_pairs = bonded_pairs[keep]
                bonded_types = bonded_types[keep]

            i_local = local_idx[bonded_pairs[:, 0]]
            j_local = local_idx[bonded_pairs[:, 1]]
            mol_atomic_nums = atomic_nums[mol_mask]

            rwmol = Chem.RWMol()
            for i in range(n):
                rwmol.AddAtom(Chem.Atom(mol_atomic_nums[i].item()))
            for idx in range(bonded_pairs.shape[0]):
                bt = bonded_types[idx].item()
                rdkit_bt = RDKIT_BOND_TYPES[bt]
                if rdkit_bt is not None:
                    try:
                        rwmol.AddBond(
                            i_local[idx].item(), j_local[idx].item(), rdkit_bt
                        )
                    except Exception:
                        pass

            try:
                mol = rwmol.GetMol()

                # Attach 3D coordinates from `pos` as a conformer to get a 3D molecule
                mol_pos = pos[mol_mask]
                conf = Chem.Conformer(n)
                for i in range(n):
                    x_coord, y_coord, z_coord = mol_pos[i].tolist()
                    conf.SetAtomPosition(
                        i, (float(x_coord), float(y_coord), float(z_coord))
                    )
                mol.AddConformer(conf, assignId=True)

                Chem.SanitizeMol(mol)
                mols.append(mol)
            except Exception:
                mols.append(None)

            if break_after_k_mols is not None and len(mols) >= break_after_k_mols:
                break

        return mols
