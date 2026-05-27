import argparse
import os
from datetime import datetime

import torch
import torch_geometric
import yaml
from lightning import seed_everything
from torch_geometric.data import Batch

from neat.dataset import DataModule
from neat.model import NEAT

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch_geometric.seed_everything(42)
seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()

import numpy as np
from Bio.PDB import PDBParser, PDBIO


def center_pdb(input_path, output_path):
    # 1. Initialize the parser and load the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_path)

    # 2. Collect all atom coordinates
    # We use a list comprehension to gather coordinates of all atoms in the structure
    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])

    if len(coords) == 0:
        raise ValueError("No atoms found in the provided PDB file.")

    # 3. Calculate the geometric center (mean of all coordinates)
    # Note: If you want the center of mass instead, you would need to weight
    # these by element masses, but geometric center is the standard for "centering coordinates".
    geometric_center = coords.mean(axis=0)

    # 4. Subtract the center from each atom's coordinates
    # This shifts the structure so that its new geometric center is at (0, 0, 0)
    for atom in structure.get_atoms():
        new_coord = atom.get_coord() - geometric_center
        atom.set_coord(new_coord)

    # 5. Write the modified structure to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"Successfully centered structure and saved to {output_path}")


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
    split_names = torch.load(test_data.raw_paths[1])["test"]

    num_molecules = params["num_molecules"]
    batch_size = params["batch_size"]
    num_batches = (num_molecules + batch_size - 1) // batch_size
    if (num_molecules % batch_size) == 0:
        num_mols_per_batch = [batch_size] * num_batches
    else:
        num_mols_per_batch = [batch_size] * (num_batches - 1) + [
            num_molecules % batch_size
        ]

    for test_data_idx, test_data_point in enumerate(test_data):
        data_point_list = [test_data_point]
        pocket_info = datamodule.test_data.collate_pocket_info(
            data_point_list, samples_per_pocket=batch_size, device=DEVICE
        )
        pocket_start_time = datetime.now()
        generated_batches = []
        for batch_idx in range(num_batches):
            num_mols_batch = num_mols_per_batch[batch_idx]
            with torch.no_grad():
                model.eval()
                generated_batch = model.generate(
                    batch_size=num_mols_batch,
                    max_atoms=params["max_atoms"],
                    num_time_steps=params["num_time_steps"],
                    time_step_spacing=params["time_step_spacing"],
                    integration_method=params["integration_method"],
                    pocket_info=pocket_info,
                )
            generated_batches.append(generated_batch)

        generated_mols = Batch.from_data_list(generated_batches)
        out_dir = os.path.join(
            ROOT, params["output_path"], "conditional", f"pocket_{test_data_idx}"
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        name = test_data_point.name.split("__")[0]
        pocket_path = [
            split_name[0] for split_name in split_names if name in split_name[0]
        ][0]
        in_pdb_file = os.path.join(test_data.raw_paths[2], pocket_path)
        out_pdb_file = os.path.join(out_dir, "pocket.pdb")
        center_pdb(in_pdb_file, out_pdb_file)
        torch.save(generated_mols, os.path.join(out_dir, "generated_mols.pt"))

        seed_end_time = datetime.now()
        print(
            f"Generation time for pocket {test_data_idx}: {seed_end_time - pocket_start_time}"
        )


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
