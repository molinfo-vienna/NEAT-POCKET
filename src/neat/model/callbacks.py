from tempfile import NamedTemporaryFile

import torch
from lightning import Callback, LightningModule, Trainer
from rdkit import Chem
import numpy as np

from neat.utils.sbdd_metrics import ClashEvaluator
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils import center_pdb


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
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        generated_mols = None
        pocket_paths = None
        mols_per_pocket = None
        if str(self.dataset).upper() == "QM9":
            generated_mols = pl_module.generate(
                batch_size=self.num_samples, integration_method="euler"
            )

        elif str(self.dataset).upper() == "GEOM":
            generated_mols = pl_module.generate(
                batch_size=self.num_samples, integration_method="euler_maruyama"
            )
        elif str(self.dataset).upper() == "CROSSDOCKED":
            num_pockets = 10
            val_data = list(trainer.val_dataloaders.dataset[:num_pockets])
            mols_per_pocket = self.num_samples // num_pockets
            pocket_info = trainer.val_dataloaders.dataset.collate_pocket_info(
                val_data, samples_per_pocket=mols_per_pocket, device=pl_module.device
            )
            generated_mols = pl_module.generate(
                batch_size=self.num_samples,
                integration_method="euler_maruyama",
                pocket_info=pocket_info,
            )
            pocket_paths = [
                trainer.val_dataloaders.dataset.get_pocket_path_from_data_point(
                    data_point
                )
                for data_point in val_data
            ]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        builder = MoleculeBuilder(vocab=str(pl_module.hparams.data_set).upper())
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
        if str(self.dataset).upper() == "CROSSDOCKED":
            clash_scores_mean = []
            clash_scores_sum = []
            for i, pocket_path in enumerate(pocket_paths):
                try:
                    with NamedTemporaryFile(delete=True, suffix=".pdb") as temp_file:
                        # Convert the string path to a pathlib.Path object
                        center_pdb(pocket_path, temp_file.name)
                        mols_for_pocket = mols[
                            i * mols_per_pocket : (i + 1) * mols_per_pocket
                        ]
                        clash_evaluator = ClashEvaluator()
                        for mol in mols_for_pocket:
                            if mol is None:
                                continue
                            clash_results = clash_evaluator.evaluate_mol(
                                mol, temp_file.name
                            )
                            clash_score_mean = clash_results["clash_score_between_mean"]
                            clash_score_sum = clash_results["clash_score_between_sum"]
                            clash_scores_mean.append(clash_score_mean)
                            clash_scores_sum.append(clash_score_sum)
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue

            if len(clash_scores_mean) > 0:
                mean_clash_score_mean = np.array(clash_scores_mean).mean()
                mean_clash_score_sum = np.array(clash_scores_sum).mean()
            else:
                mean_clash_score_mean = 0.0
                mean_clash_score_sum = 0.0

            pl_module.log(
                "val/clashes_mean",
                mean_clash_score_mean,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                "val/clashes_sum",
                mean_clash_score_sum,
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
