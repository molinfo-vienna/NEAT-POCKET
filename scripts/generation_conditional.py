import argparse
import os
from datetime import datetime

from rdkit import Chem
import torch
import torch_geometric
import yaml
from lightning import seed_everything
from torch_geometric.data import Batch
import numpy as np
from rdkit.Chem import BRICS

from neat.dataset import DataModule
from neat.dataset.dataset_crossdocked import _add_hydrogens_with_rdkit, _largest_fragment, _ligand_features
from neat.model import NEAT
from neat.utils import center_pdb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch_geometric.seed_everything(42)
seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()

import matplotlib.pyplot as plt
from typing import List

def plot_fragment_statistics(full_fragment_list: List[Chem.Mol], full_ligand_list: List[Chem.Mol]):
    if len(full_fragment_list) != len(full_ligand_list):
        raise ValueError("The fragment and ligand lists must be of the same length.")
        
    # 1. Extract the heavy atom counts (excluding hydrogens for standard structural analysis)
    # If you want to include hydrogens, use mol.GetNumAtoms(onlyExplicit=False) instead.
    frag_atom_counts = np.array([mol.GetNumHeavyAtoms() for mol in full_fragment_list])
    ligand_atom_counts = np.array([mol.GetNumHeavyAtoms() for mol in full_ligand_list])
    
    # Calculate the relative size ratio
    # Adding a tiny epsilon to avoid potential division by zero errors on empty/malformed mols
    relative_sizes = frag_atom_counts / (ligand_atom_counts + 1e-9)
    
    # 2. Setup the matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Absolute Atom Count Frequency ---
    max_atoms = int(np.max(frag_atom_counts)) if len(frag_atom_counts) > 0 else 10
    # Set bin edges exactly at integer boundaries [0, 1, 2, ..., max_atoms + 1]
    abs_bins = np.arange(0, max_atoms + 2)
    
    ax1.hist(frag_atom_counts, bins=abs_bins, edgecolor='black', alpha=0.75, rwidth=0.8, align='left')
    ax1.set_xlim(-0.5, max_atoms + 0.5)
    ax1.set_xticks(np.arange(0, max_atoms + 1, max(1, max_atoms // 10))) # Dynamic tick scaling
    ax1.set_title("Fragment Absolute Size Distribution")
    ax1.set_xlabel("Number of Heavy Atoms")
    ax1.set_ylabel("Frequency")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # --- Plot 2: Relative Atom Count Frequency ---
    # 10% bin width means exactly 10 bins from 0.0 to 1.0
    rel_bins = np.linspace(0.0, 1.0, 21)
    
    ax2.hist(relative_sizes, bins=rel_bins, edgecolor='black', alpha=0.75, rwidth=0.8)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticks(rel_bins)
    # Format labels as nice percentages
    ax2.set_xticklabels([f"{int(x*100)}%" for x in rel_bins])
    
    ax2.set_title("Fragment Size Relative to Full Ligand")
    ax2.set_xlabel("Relative Size (% of Total Heavy Atoms)")
    ax2.set_ylabel("Frequency")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def prepare_fragment_info(fragment_list: list[Chem.Mol], num_molecules: int, device=DEVICE) -> dict:
    x_list = []
    pos_list = []
    batch_list = []

    for i, fragment in enumerate(fragment_list):
        x, pos = _ligand_features(fragment)
        x_list.append(torch.cat([x for _ in range(num_molecules)], dim=0))
        pos_list.append(torch.cat([pos for _ in range(num_molecules)], dim=0))
        batch_list.append(torch.cat([torch.ones(len(x), dtype=torch.long) * j for j in range(num_molecules)], dim=0) + i * num_molecules)

    return {
        "fragment_x": torch.hstack(x_list),
        "fragment_pos": torch.vstack(pos_list),
        "fragment_batch": torch.hstack(batch_list)
    }

def get_fragment_with_brics(mol: Chem.Mol) -> Chem.Mol:
    """
    Centers a molecule to its center of mass (unweighted),
    fragments it using BRICS, and returns the largest fragment 
    with BRICS dummy atoms stripped.
    """
    # 1. Center the molecule to its unweighted Center of Mass (COM)
    # Ensure the molecule has at least one conformer
    if mol.GetNumConformers() == 0:
        raise ValueError("The input molecule must have a 3D conformation to calculate the center of mass.")
    
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    
    # Extract coordinates of all atoms
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    
    # Calculate unweighted Center of Mass (mean of positions)
    com = np.mean(coords, axis=0)
    
    # Shift all atom positions to center them at (0, 0, 0)
    for i in range(num_atoms):
        original_pos = conf.GetAtomPosition(i)
        centered_pos = original_pos - com
        conf.SetAtomPosition(i, centered_pos)
        
    # 2. Fragment the molecule using BRICS
    # BreakBRICSBonds breaks the bonds and adds dummy atoms at the cut points
    fragmented_mol = BRICS.BreakBRICSBonds(mol)
    
    # 3. Get individual disconnected fragments
    fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
    
    if not fragments or len(fragments) <= 1:
        return mol # Return original if no fragments were generated
    
    # Get largest fragment
    clean_frags = [Chem.DeleteSubstructs(frag, Chem.MolFromSmiles('*')) for frag in fragments]
    num_atoms = np.array([frag.GetNumHeavyAtoms() for frag in clean_frags])
    largest_idx = np.argsort(num_atoms)[-1]
    largest_fragment = clean_frags[largest_idx]
    
    return largest_fragment

def generate(args: argparse.Namespace) -> None:
    """Generate molecules using the NEAT model.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
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

    datamodule = DataModule(
        os.path.join(ROOT, "data"),
        params["data_set"].upper(),
    )
    datamodule.setup()
    test_data = datamodule.test_data

    num_molecules = params["num_molecules"]
    chunk_size = params["chunk_size"]
    fbdd = params.get("fbdd", False)
    full_fragment_list = []
    full_ligand_list = []

    for chunk_start_idx in range(0, len(test_data), chunk_size):
        # Here we store the metadata data in the appropriate directories
        if fbdd:
            fragment_list = []
        for data_idx_in_chunk, data_idx in enumerate(
            range(chunk_start_idx, min(chunk_start_idx + chunk_size, len(test_data)))
        ):
            out_dir = os.path.join(
                ROOT, params["output_path"], "conditional", f"pocket_{data_idx}"
            )
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            in_pdb_file = test_data.get_pocket_path_from_data_point(test_data[data_idx])
            out_pdb_file = os.path.join(out_dir, "pocket.pdb")
            pocket_center = center_pdb(in_pdb_file, out_pdb_file, return_center=True)
            in_sdf_file = in_pdb_file.replace("_pocket10.pdb", ".sdf")
            out_sdf_file = os.path.join(out_dir, "ligand.sdf")
            supplier = Chem.SDMolSupplier(in_sdf_file, removeHs=False)
            rdmol = supplier[0]
            rdmol = _largest_fragment(rdmol)
            rdmol = _add_hydrogens_with_rdkit(rdmol)
            
            conformer = rdmol.GetConformer()
            for i in range(rdmol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                # Convert Point3D to numpy array, subtract center, and update
                new_pos = np.array([pos.x, pos.y, pos.z]) - pocket_center
                conformer.SetAtomPosition(i, new_pos)

            # 4. Save the modified molecule back to an SDF file
            writer = Chem.SDWriter(out_sdf_file)
            writer.write(rdmol)
            writer.close()
            
            full_ligand_list.append(rdmol)
            if fbdd:
                fragment = get_fragment_with_brics(rdmol)
                fragment_list.append(fragment)
                full_fragment_list.append(fragment)

        if fbdd:
            fragment_info = prepare_fragment_info(fragment_list, num_molecules, device=DEVICE)
        
        # Here we generate the molecules
        data_point_list = list(
            test_data[chunk_start_idx : chunk_start_idx + chunk_size]
        )
        pocket_info = datamodule.test_data.collate_pocket_info(
            data_point_list, samples_per_pocket=num_molecules, device=DEVICE
        )
        pocket_start_time = datetime.now()
        with torch.no_grad():
            model.eval()
            generated_mols = model.generate(
                batch_size=pocket_info["pocket_batch"].max().item() + 1,
                max_atoms=params["max_atoms"],
                num_time_steps=params["num_time_steps"],
                time_step_spacing=params["time_step_spacing"],
                integration_method=params["integration_method"],
                pocket_info=pocket_info,
                fragment_info=fragment_info if fbdd else None,
            )

        # Here we store the generated data in the appropriate directories
        for data_idx_in_chunk, data_idx in enumerate(
            range(chunk_start_idx, min(chunk_start_idx + chunk_size, len(test_data)))
        ):
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
            generated_mols_subset.batch -= generated_mols_subset.batch.min()
            out_dir = os.path.join(
                ROOT, params["output_path"], "conditional", f"pocket_{data_idx}"
            )
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                generated_mols_subset, os.path.join(out_dir, "generated_mols.pt")
            )

        seed_end_time = datetime.now()
        print(
            f"Generation time for pockets {chunk_start_idx} to {min(chunk_start_idx + chunk_size, len(test_data)) - 1}: {seed_end_time - pocket_start_time}"
        )
        
    # plot_fragment_statistics(full_fragment_list, full_ligand_list)


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()

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