import torch
from lightning import Callback, LightningModule, Trainer
from rdkit import Chem
from torch_geometric.nn import global_mean_pool

from neat.model.molecule_builder import MoleculeBuilder
from neat.utils.edm_metrics import edm_metrics


class GenerationMonitor(Callback):
    """Callback to monitor molecule generation during training.

    Args:
        num_samples: Number of molecules to generate for evaluation.
        every_n_epochs: Frequency (in epochs) to perform generation and evaluation.
        dataset: Dataset name, either "QM9" or "GEOM" or "CrossDocked".
    """

    def __init__(
        self,
        num_samples: int = 1000,
        every_n_epochs: int = 50,
        dataset: str = "QM9",
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.dataset = dataset

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        pl_module.log(
            "val/validity",
            -torch.inf,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        pl_module.log(
            "val/uniqueness",
            -torch.inf,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if (
            trainer.current_epoch % self.every_n_epochs != 0
            or trainer.current_epoch == 0
        ):
            return

        generated_mols = None
        if self.dataset == "QM9":
            generated_mols = pl_module.generate(
                batch_size=self.num_samples, integration_method="euler"
            )

        elif self.dataset == "GEOM":
            generated_mols = pl_module.generate(
                batch_size=self.num_samples, integration_method="euler_maruyama"
            )
        elif self.dataset == "CrossDocked":
            val_data = trainer.val_dataloaders.dataset
            res_id = val_data.pocket_residue_id
            resets = torch.cat([torch.tensor([False]), res_id[1:] < res_id[:-1]])
            graph_ids = resets.long().cumsum(dim=0)
            mean_pos = global_mean_pool(val_data.pocket_pos, graph_ids)
            pos = val_data.pocket_pos - mean_pos[graph_ids]
            mask = graph_ids < self.num_samples
            pocket_info = {
                "pocket_x": val_data.pocket_x[mask].to(pl_module.device),
                "pocket_pos": pos[mask].to(pl_module.device),
                "pocket_residue_id": val_data.pocket_residue_id[mask].to(
                    pl_module.device
                ),
                "pocket_residue_type": val_data.pocket_residue_type[mask].to(
                    pl_module.device
                ),
                "pocket_batch": graph_ids[mask].to(pl_module.device),
            }
            generated_mols = pl_module.generate(
                batch_size=self.num_samples,
                integration_method="euler_maruyama",
                pocket_info=pocket_info,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        builder = MoleculeBuilder(vocab=pl_module.hparams.data_set)
        mols = builder.generate_rdkit_molecules_via_xyz2mol(
            generated_mols.x, generated_mols.pos, generated_mols.batch
        )
        n_valid = self.compute_validity(mols)
        n_unique = self.compute_uniqueness(mols)
        frac_valid = n_valid / self.num_samples
        frac_unique = n_unique / n_valid if n_valid > 0 else 0.0

        pl_module.log(
            "val/validity",
            frac_valid,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        pl_module.log(
            "val/uniqueness",
            frac_unique,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def compute_validity(
        self,
        mols: list[Chem.Mol],
    ) -> int:
        """Compute the number of valid molecules in a list.

        Args:
            mols (list[Chem.Mol]): List of RDKit molecules.

        Returns:
            num_valid (int): Number of valid molecules.
        """
        num_valid = 0
        for mol in mols:
            if mol is not None:
                num_valid += 1
        return num_valid

    def compute_uniqueness(self, mols: list[Chem.Mol]) -> int:
        """Compute the number of unique molecules in a list.

        Args:
            mols (list[Chem.Mol]): List of RDKit molecules.

        Returns:
            num_unique (int): Number of unique molecules.
        """
        unique_smiles = set()
        for mol in mols:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                unique_smiles.add(smiles)
        return len(unique_smiles)
