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

from neat.dataset import DataModule
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

    for chunk_start_idx in range(0, len(test_data), chunk_size):
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
            )

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
            mol = supplier[0]
            conformer = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                # Convert Point3D to numpy array, subtract center, and update
                new_pos = np.array([pos.x, pos.y, pos.z]) - pocket_center
                conformer.SetAtomPosition(i, new_pos)

            # 4. Save the modified molecule back to an SDF file
            writer = Chem.SDWriter(out_sdf_file)
            writer.write(mol)
            writer.close()

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
            torch.save(
                generated_mols_subset, os.path.join(out_dir, "generated_mols.pt")
            )

        seed_end_time = datetime.now()
        print(
            f"Generation time for pockets {chunk_start_idx} to {min(chunk_start_idx + chunk_size, len(test_data)) - 1}: {seed_end_time - pocket_start_time}"
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
