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
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_generation_conditional.yaml")
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
    batch_size = params["batch_size"]
    num_batches = (num_molecules + batch_size - 1) // batch_size
    if (num_molecules % batch_size) == 0:
        num_mols_per_batch = [batch_size] * num_batches
    else:
        num_mols_per_batch = [batch_size] * (num_batches - 1) + [
            num_molecules % batch_size
        ]

    for test_data_idx, test_data in enumerate(test_data):
        pocket_x = torch.cat([test_data.pocket_x.to(DEVICE) for _ in range(batch_size)])
        pocket_pos = torch.cat([test_data.pocket_pos.to(DEVICE) for _ in range(batch_size)])
        pocket_residue_id = torch.cat([test_data.pocket_residue_id.to(DEVICE) for _ in range(batch_size)])
        pocket_residue_type = torch.cat([test_data.pocket_residue_type.to(DEVICE) for _ in range(batch_size)])
        res_id = pocket_residue_id
        resets = torch.cat([torch.tensor([False], device=DEVICE), res_id[1:] < res_id[:-1]])
        pocket_batch = resets.long().cumsum(dim=0).to(DEVICE)
        pocket_info = {
            "pocket_x": pocket_x,
            "pocket_pos": pocket_pos,
            "pocket_residue_id": pocket_residue_id,
            "pocket_residue_type": pocket_residue_type,
            "pocket_batch": pocket_batch.to(DEVICE),
        }
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
        torch.save(generated_mols, os.path.join(out_dir, "generated_mols.pt"))

        seed_end_time = datetime.now()
        print(f"Generation time for pocket {test_data_idx}: {seed_end_time - pocket_start_time}")


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
