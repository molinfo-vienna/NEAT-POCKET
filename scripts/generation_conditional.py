"""Conditional pocket-conditioned molecule generation with NEAT.

Workflow:
  1. Load a trained checkpoint and the CrossDocked test set.
  2. For each pocket chunk: center the pocket/ligand, optionally load a
     precomputed BRICS fragment, and run model.generate().
  3. Save generated_mols.pt per pocket under output/.../conditional/.

Fragment-based (FBDD) generation is controlled by `fragment_type` in
config_generation_conditional.yaml:
  - null: standard pocket-conditioned generation (no fragment seed)
  - largest | second_largest | smallest: seed generation from that
    precomputed fragment (see scripts/fragments_from_crossdocked.py)
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch_geometric
import yaml
from lightning import seed_everything
from rdkit import Chem
from torch_geometric.data import Batch

from neat.dataset import DataModule
from neat.dataset.dataset_crossdocked import (
    _largest_fragment,
    _ligand_features,
)
from neat.model import NEAT
from neat.model.bond_predictor import BondPredictor
from neat.model.bond_predictor import BondPredictor
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils import center_pdb, cif_2_pdb, save_molecules_to_sdf

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch_geometric.seed_everything(42)
seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()
FRAGMENTS_DIR = os.path.join(ROOT, "fragments")
FRAGMENT_TYPES = ("largest", "second_largest", "smallest")


def prepare_fragment_info(
    fragment_list: list[Chem.Mol], num_molecules: int
) -> dict:
    """Build batched fragment tensors for model.generate().

    For each pocket fragment in the chunk, atom types and positions are
    repeated `num_molecules` times so every sample for that pocket starts
    from the same fragment seed. Batch indices encode
    (pocket_in_chunk, sample_within_pocket).

    Args:
        fragment_list: One RDKit mol per pocket in the current chunk.
        num_molecules: Number of molecules to generate per pocket.

    Returns:
        Dict with fragment_x, fragment_pos, and fragment_batch tensors.
    """
    x_list = []
    pos_list = []
    batch_list = []

    for i, fragment in enumerate(fragment_list):
        x, pos = _ligand_features(fragment, get_charge=False)
        # Repeat the fragment once for each molecule generated for this pocket.
        x_list.append(torch.cat([x for _ in range(num_molecules)], dim=0))
        pos_list.append(torch.cat([pos for _ in range(num_molecules)], dim=0))
        batch_list.append(
            torch.cat(
                [torch.ones(len(x), dtype=torch.long) * j for j in range(num_molecules)],
                dim=0,
            )
            + i * num_molecules
        )

    return {
        "fragment_x": torch.hstack(x_list),
        "fragment_pos": torch.vstack(pos_list),
        "fragment_batch": torch.hstack(batch_list),
    }


def load_fragment(pdb_code: str, dataset: str, fragment_type: str) -> Chem.Mol:
    """Load a precomputed BRICS fragment SDF for a pocket.

    Fragments are written by scripts/fragments_from_crossdocked.py under
    fragments/{fragment_type}/{pdb_code}.sdf.

    Args:
        pdb_code: PDB code identifying the pocket.
        dataset: Dataset name (e.g., "CrossDocked", "SPINDR").
        fragment_type: One of largest, second_largest, smallest.

    Returns:
        Fragment molecule with 3D coordinates (pocket-/COM-centered).
    """
    if fragment_type not in FRAGMENT_TYPES:
        raise ValueError(
            f"Unknown fragment_type {fragment_type!r}. "
            f"Expected one of {FRAGMENT_TYPES}."
        )
    fragment_path = os.path.join(FRAGMENTS_DIR, dataset, fragment_type, f"{pdb_code}.sdf")
    if not os.path.exists(fragment_path):
        raise FileNotFoundError(
            f"Fragment file not found: {fragment_path}."
        )
    supplier = Chem.SDMolSupplier(fragment_path, removeHs=False, sanitize=False)
    fragment = supplier[0]
    if fragment is None:
        raise ValueError(f"Failed to read fragment from {fragment_path}.")
    return fragment


def generate(args: argparse.Namespace) -> None:
    """Run pocket-conditioned (optionally fragment-seeded) generation.

    Args:
        args: Must provide optional config_file; defaults to
            scripts/config_generation_conditional.yaml.
    """
    # --- Config & model ---
    if args.config_file is not None:
        CONFIG_FILE_PATH = args.config_file
        print(f"Using config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(
            ROOT, "scripts", "config_generation_conditional.yaml"
        )
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    # Load the checkpoint of the generative model
    checkpoints_dir = os.path.join(ROOT, params["checkpoints_path"], "checkpoints")
    pt_files = [
        f
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".ckpt") and f.startswith("best-val-loss")
    ]
    if not pt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

    checkpoints_path = os.path.join(checkpoints_dir, pt_files[0])
    print(f"Using checkpoint file: {checkpoints_path}")

    MODEL = NEAT
    model = MODEL.load_from_checkpoint(checkpoints_path, map_location=DEVICE)
    
    # Load bond predictor if available
    bond_predictor_dir = params.get("bond_predictor_dir", None)
    
    if bond_predictor_dir is not None:
        bond_predictor_dir = os.path.join(ROOT, bond_predictor_dir, "checkpoints")
        pt_files = [
            f
            for f in os.listdir(bond_predictor_dir)
            if f.endswith(".ckpt")
        ]
        if not pt_files:
            raise FileNotFoundError(f"No .ckpt files found in {bond_predictor_dir}")

        bond_predictor_path = os.path.join(bond_predictor_dir, pt_files[0])
        print(f"Using checkpoint file: {bond_predictor_path}")
        bond_predictor = BondPredictor.load_from_checkpoint(bond_predictor_path, map_location=DEVICE)
    else:
        bond_predictor = None

    # --- Data ---
    dataset = params["data_set"].upper()
    datamodule = DataModule(
        os.path.join(ROOT, "data"),
        dataset,
    )
    datamodule.setup()
    test_data = datamodule.test_data

    num_molecules = params["num_molecules"]
    chunk_size = params["chunk_size"]
    fragment_type = params.get("fragment_type", None)
    # null / None => standard conditional generation; otherwise FBDD with that rank.
    if fragment_type is not None:
        print(f"Using fragment type: {fragment_type}")

    # Process pockets in chunks to bound GPU memory.
    for chunk_start_idx in range(0, len(test_data), chunk_size):
        chunk_end_idx = min(chunk_start_idx + chunk_size, len(test_data))
        # Indices / fragments actually used for generation in this chunk
        # (FBDD may drop pockets with no precomputed fragment).
        included_data_indices: list[int] = []
        fragment_list: list[Chem.Mol] = []

        # Prepare centered pocket/ligand files (and collect fragments if FBDD).
        for data_idx in range(chunk_start_idx, chunk_end_idx):
            pdb_code = test_data.get_pdb_code_from_data_point(test_data[data_idx])

            if fragment_type is not None:
                try:
                    fragment = load_fragment(pdb_code, dataset, fragment_type)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Skipping pocket {pdb_code}: {e}")
                    continue

            out_dir = os.path.join(
                ROOT, params["output_path"], "conditional", f"pocket_{pdb_code}"
            )
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Center pocket at the origin and write pocket.pdb for later evaluation.
            in_pdb_file = test_data.get_pocket_path_from_data_point(test_data[data_idx])
            out_pdb_file = os.path.join(out_dir, "pocket.pdb")
            if in_pdb_file.endswith(".cif"):
                pocket_center = cif_2_pdb(in_pdb_file, out_pdb_file, return_center=True)
            elif in_pdb_file.endswith(".pdb"):
                pocket_center = center_pdb(in_pdb_file, out_pdb_file, return_center=True)
            else:
                raise ValueError(f"Unsupported pocket file format: {in_pdb_file}")

            # Load the reference ligand, keep the largest connected component,
            # add hydrogens, and shift into the same pocket-centered frame.
            in_sdf_file = test_data.get_ligand_path_from_data_point(test_data[data_idx])
            out_sdf_file = os.path.join(out_dir, "ligand.sdf")
            supplier = Chem.SDMolSupplier(in_sdf_file, removeHs=False, sanitize=False)
            rdmol = supplier[0]
            rdmol = _largest_fragment(rdmol)
            # TODO: decide which hydrogenation method to use
            rdmol = Chem.AddHs(rdmol, addCoords=True)

            conformer = rdmol.GetConformer()
            for i in range(rdmol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                new_pos = np.array([pos.x, pos.y, pos.z]) - pocket_center
                conformer.SetAtomPosition(i, new_pos)

            writer = Chem.SDWriter(out_sdf_file)
            writer.write(rdmol)
            writer.close()

            included_data_indices.append(data_idx)
            if fragment_type is not None:
                fragment_list.append(fragment)

        if not included_data_indices:
            print(
                f"Skipping chunk {chunk_start_idx}–{chunk_end_idx - 1}: "
                "no pockets with usable fragments."
            )
            continue

        fragment_info = None
        if fragment_type is not None:
            fragment_info = prepare_fragment_info(
                fragment_list, num_molecules
            )

        # Collate pocket features for the pockets kept in this chunk.
        data_point_list = [test_data[i] for i in included_data_indices]
        pocket_info = datamodule.test_data.collate_pocket_info(
            data_point_list, samples_per_pocket=num_molecules, device=DEVICE
        )
        pocket_start_time = datetime.now()
        with torch.no_grad():
            model.eval()
            cfg_factor = params.get("cfg_factor", 0.)
            generated_mols = model.generate(
                batch_size=pocket_info["pocket_batch"].max().item() + 1,
                max_atoms=params["max_atoms"],
                num_time_steps=params["num_time_steps"],
                time_step_spacing=params["time_step_spacing"],
                integration_method=params["integration_method"],
                pocket_info=pocket_info,
                fragment_info=fragment_info,
                cfg_factor=cfg_factor,
            )

        # Split the chunk batch back into per-pocket tensors and save.
        for data_idx_in_chunk, data_idx in enumerate(included_data_indices):
            pdb_code = test_data.get_pdb_code_from_data_point(test_data[data_idx])
            out_dir = os.path.join(
                ROOT, params["output_path"], "conditional", f"pocket_{pdb_code}"
            )
            subset_mask = torch.isin(
                generated_mols.batch,
                torch.arange(
                    data_idx_in_chunk * num_molecules,
                    (data_idx_in_chunk + 1) * num_molecules,
                    device=generated_mols.batch.device,
                ),
            )
            generated_mols_subset = Batch(
                x=generated_mols.x[subset_mask],
                pos=generated_mols.pos[subset_mask],
                batch=generated_mols.batch[subset_mask],
            )
            # Renormalize batch ids to start at 0 within this pocket file.
            generated_mols_subset.batch -= generated_mols_subset.batch.min()
            torch.save(
                generated_mols_subset, os.path.join(out_dir, "generated_mols.pt")
            )
            # Generate mols with bond and charge predictors and save as SDF for later evaluation.
            builder = MoleculeBuilder(vocab=params["data_set"])
            if bond_predictor is not None:
                rdkit_mols = builder.generate_rdkit_molecules_via_bond_predictor(
                    generated_mols_subset.x,
                    generated_mols_subset.pos,
                    generated_mols_subset.batch,
                    bond_predictor=bond_predictor,
                    progress_bar=True,
                )
            else:
                rdkit_mols = builder.generate_rdkit_molecules_via_xyz2mol(
                    generated_mols_subset.x,
                    generated_mols_subset.pos,
                    generated_mols_subset.batch,
                    progress_bar=True
                )
            save_molecules_to_sdf(rdkit_mols, os.path.join(out_dir, "generated_mols.sdf"))

        seed_end_time = datetime.now()
        print(
            f"Generation time for {len(included_data_indices)} pocket(s) "
            f"in chunk {chunk_start_idx}–{chunk_end_idx - 1}: "
            f"{seed_end_time - pocket_start_time}"
        )


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Generate molecules conditioned on CrossDocked pockets."
    )

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for generation.",
    )

    args = parser.parse_args()

    generate(args)

    end_time = datetime.now()
    print(f"Total generation time: {end_time - start_time}")
