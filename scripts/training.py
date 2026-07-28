import argparse
import os

import torch
import torch_geometric
import yaml
from lightning import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from neat.dataset import DataModule
from neat.model import NEAT, GenerationMonitor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(42)
seed_everything(42)

ROOT = os.getcwd()


class UnfreezeModelCallback(Callback):
    def __init__(self, unfreeze_epoch: int):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Check if we have hit the target epoch and haven't unfrozen yet
        if trainer.current_epoch >= self.unfreeze_epoch and not self._unfrozen:
            print(f"\n[Callback] Epoch {trainer.current_epoch}: Unfreezing all layers and updating optimizer.")
            
            # 1. Unfreeze all parameters in the model
            for param in pl_module.parameters():
                param.requires_grad = True
            
            # 2. Update the optimizer(s) so they track the newly unfrozen parameters
            # We clear the old parameter groups and re-add all parameters.
            for optimizer in trainer.optimizers:
                optimizer.param_groups.clear()
                # If you use multiple parameter groups (e.g., different learning rates),
                # you would need to recreate that specific structure here.
                optimizer.add_param_group({"params": pl_module.parameters()})
            
            # Mark as done so we don't repeat this every epoch after
            self._unfrozen = True


def train(args: argparse.Namespace) -> None:
    """Train NEAT model.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """

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

    # Initialize the model
    accumulate_grad_batches = params.pop("accumulate_grad_batches")
    params["vocab_size"] = datamodule.vocab_size
    model = MODEL(**params)

    # If path to pretrained weights is provided, load them
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
    unfreeze_callback = UnfreezeModelCallback(unfreeze_epoch=params.get("unfreeze_epoch", 0))

    callbacks = [
        GenerationMonitor(
            num_samples=400,
            every_n_epochs=generate_every_n_epochs,
            dataset=params["data_set"],
        ),
        checkpoint_val_loss,
        checkpoint_validity,
        LearningRateMonitor(logging_interval="epoch"),
        unfreeze_callback,
    ]

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
