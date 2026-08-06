"""Train bond predictor models on QM9 and GEOM datasets.

Uses DataModule with task="bond_prediction" which precomputes radius-graph edges
and edge labels. The BondPredictor predicts bond type per edge.

Usage:
    python scripts/training_bond_predictor.py --config scripts/config_bond_predictor.yaml
"""

import argparse
import os

import torch
import torch_geometric
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from neat.dataset import DataModule
from neat.model import BondPredictor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(42)
seed_everything(42)

ROOT = os.getcwd()


def train(args: argparse.Namespace) -> None:
    """Train bond predictor models on QM9 and GEOM datasets.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    config_path = args.config_file or os.path.join(
        ROOT, "scripts", "config_bond_predictor.yaml"
    )
    print(f"Using config: {config_path}")

    params = yaml.load(
        open(config_path, "r"),
        Loader=yaml.FullLoader,
    )

    dataset_name = str(params.get("data_set")).upper()
    data_dir = os.path.join(ROOT, "data")
    print(f"Loading {dataset_name}...")

    datamodule = DataModule(
        data_dir=data_dir,
        data_set=dataset_name,
        batch_size=params.get("batch_size", 64),
        num_workers=params.get("num_workers", 8),
        task="bond_prediction",
        bond_predictor_radius=params.get("radius", 2.5),
        bond_predictor_noise_ratio=params.get("noise_ratio", 0.0),
    )
    datamodule.setup()
    vocab_size = datamodule.vocab_size
    print(
        f"Train: {len(datamodule.training_data)}, Val: {len(datamodule.validation_data)}"
    )

    model_params = {
        "vocab_size": vocab_size,
        "n_embd": params.get("n_embd", 256),
        "n_conv_layers": params.get("n_conv_layers", 4),
        "dropout": params.get("dropout", 0.1),
        "learning_rate": params.get("learning_rate", 1e-3),
        "weight_decay": params.get("weight_decay", 1e-6),
        "max_epochs": params.get("max_epochs", 100),
        "lr_warmup_epochs": params.get("lr_warmup_epochs", 5),
        "lr_min_ratio": params.get("lr_min_ratio", 0.1),
        "radius": params.get("radius", 2.5),
        "noise_ratio": params.get("noise_ratio", 0.05),
        "predict_charge": params.get("predict_charge", False),
    }

    model = BondPredictor(**model_params)

    log_dir = os.path.join(ROOT, "logs", "BondPredictor")
    tb_logger = TensorBoardLogger(log_dir, name=dataset_name, default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        filename=f"bond_predictor_{dataset_name}-{{epoch:02d}}",
        save_top_k=1,
        every_n_epochs=1,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=params.get("early_stopping_patience", 5),
        min_delta=params.get("early_stopping_min_delta", 1e-3),
    )

    callbacks = [
        checkpoint_callback,
        early_stopping,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        devices=[0] if torch.cuda.is_available() else "auto",
        max_epochs=model_params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    trainer.fit(model=model, datamodule=datamodule)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


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
