"""Train NEAT model.

Workflow:
  1. Load the config file.
  2. Load the data module.
  3. Initialize the model.
  4. Load the bond predictor if provided.
  5. Load the pretrained model if provided.
  6. Initialize the callbacks.
  7. Train the model.
"""

import argparse
import os

import torch
import torch_geometric
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from neat.dataset import DataModule
from neat.model import NEAT, GenerationMonitor, UnfreezeModelCallback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(42)
seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()


def train(args: argparse.Namespace) -> None:
    """Train NEAT model.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """

    # (1) Load configuration
    if args.config_file is not None:
        config_file_path = args.config_file
        print(f"Using config file: {config_file_path}")
    else:
        config_file_path = os.path.join(ROOT, "scripts", "config_training.yaml")
        print(f"Using default config file: {config_file_path}")

    MODEL = NEAT
    params = yaml.load(
        open(config_file_path, "r"),
        Loader=yaml.FullLoader,
    )

    # (2) Load data module
    datamodule = DataModule(
        os.path.join(ROOT, "data"),
        params["data_set"],
        batch_size=params["batch_size"],
        flow_matching_noise_std=params["noise_std"],
        source_set_perturbation_fraction=params["perturbation_fraction"],
        source_set_perturbation_std=params["perturbation_std"],
        num_workers=8,
    )
    datamodule.setup()

    # (3) Initialize the model
    accumulate_grad_batches = params.pop("accumulate_grad_batches")
    params["vocab_size"] = datamodule.vocab_size
    model = MODEL(**params)

    # (4) If path to bond predictor is provided, get checkpoints to pass them to the GenerationMonitor callback
    bond_predictor_dir = params.get("bond_predictor_dir", None)
    if bond_predictor_dir is not None:
        bond_predictor_dir = os.path.join(ROOT, bond_predictor_dir, "checkpoints")
        pt_files = [f for f in os.listdir(bond_predictor_dir) if f.endswith(".ckpt")]
        if not pt_files:
            raise FileNotFoundError(f"No .ckpt files found in {bond_predictor_dir}")
        bond_predictor_path = os.path.join(bond_predictor_dir, pt_files[0])
        print(f"Using checkpoint file: {bond_predictor_path}")

    # (5) If path to pretrained unconditional model is provided, load weights and transfer them to the conditional model
    pretrained_model_path = params.get("pretrained_model_path", None)
    if pretrained_model_path is not None:
        checkpoints_dir = os.path.join(ROOT, pretrained_model_path, "checkpoints")
        pt_files = [
            f
            for f in os.listdir(checkpoints_dir)
            if f.endswith(".ckpt") and f.startswith("best-val-loss")
        ]
        if not pt_files:
            raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

        checkpoints_path = os.path.join(checkpoints_dir, pt_files[0])
        print(f"Using checkpoint file: {checkpoints_path}")
        pretrained_model = MODEL.load_from_checkpoint(
            checkpoints_path, map_location="cpu"
        )
        model.initialize_from_pretrained_model(pretrained_model)

    # (6) Initialize the callbacks
    tb_logger = TensorBoardLogger(
        os.path.join(ROOT, "logs"),
        name=f"{MODEL.__name__}",
        default_hp_metric=False,
    )

    checkpoint_val_loss = ModelCheckpoint(
        monitor="val/val_loss",
        mode="min",
        filename="best-val-loss-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=10,
    )

    generate_every_n_epochs = 10
    checkpoint_validity = ModelCheckpoint(
        monitor="val/validity",
        mode="max",
        filename="best-val-validity-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=generate_every_n_epochs,
    )

    checkpoint_perodic = ModelCheckpoint(
        filename="periodic-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=10,
    )

    unfreeze_callback = UnfreezeModelCallback(
        unfreeze_epoch=params.get("unfreeze_epoch", 0)
    )

    callbacks = [
        GenerationMonitor(
            num_samples=400,
            every_n_epochs=generate_every_n_epochs,
            dataset=params["data_set"],
            bond_predictor_path=bond_predictor_path,
        ),
        checkpoint_val_loss,
        checkpoint_validity,
        checkpoint_perodic,
        LearningRateMonitor(logging_interval="epoch"),
        unfreeze_callback,
    ]

    # (7) Model training and logging
    trainer = Trainer(
        devices=[0],
        max_epochs=params["max_epochs"],
        accelerator="gpu",
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for training.",
    )

    args = parser.parse_args()

    train(args)
