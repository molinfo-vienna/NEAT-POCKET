import functools
import os

import torch
from lightning import LightningDataModule
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.transforms import Distance
from torch_geometric.utils import coalesce

from .augmentation import RandomRotationAugmentation
from .dataset_crossdocked import CrossDockedDataSet
from .dataset_geom import GEOMDataSet
from .dataset_qm9 import QM9DataSet
from .dataset_spindr import SpindrDataSet
from .splitting import SourceTargetSplitter


def bond_prediction_batch_transform(
    batch: Batch,
    radius: float,
    noise_ratio: float = 0.0,
) -> Batch:
    """Transform a batch of graphs for bond prediction.

    Args:
        batch (Batch): Batch of graphs.
        radius (float): Radius for edge computation.
        noise_ratio (float, optional): Ratio of noise to add to coordinates. Defaults to 0.0.

    Returns:
        Batch: Transformed batch of graphs.
    """
    # (0) Coordinate noise
    if noise_ratio > 0:
        noise_std = noise_ratio * radius
        added_noise = noise_std * torch.randn_like(batch.pos)
        batch.pos = batch.pos + added_noise

    # (1) Compute radius edges
    rad_edge_index = radius_graph(batch.pos, r=radius, batch=batch.batch, loop=False)
    rad_edge_labels = torch.zeros(
        rad_edge_index.size(1),
        device=batch.edge_labels.device,
        dtype=batch.edge_labels.dtype,
    )

    # (2) Combine original molecular edges and new radius edges
    combined_edge_index = torch.cat([batch.edge_index, rad_edge_index], dim=1)
    combined_edge_labels = torch.cat([batch.edge_labels, rad_edge_labels], dim=0)

    # (3) Deduplicate (coalesce) edges, keeping original labels (max reduction)
    new_edge_index, new_edge_labels = coalesce(
        combined_edge_index,
        combined_edge_labels,
        num_nodes=batch.num_nodes,
        reduce="max",
    )

    batch.edge_index = new_edge_index
    batch.edge_labels = new_edge_labels

    # (4) Add distance attributes
    batch = Distance(norm=False)(batch)

    return batch


def bond_prediction_collate_fn(
    batch: list,
    radius: float,
    noise_ratio: float = 0.0,
) -> Batch:
    batch = Batch.from_data_list(batch)
    return bond_prediction_batch_transform(batch, radius, noise_ratio)


def source_target_split_batch_transform(
    batch: Batch,
    source_target_split: str,
    noise_std: float,
    source_set_perturbation_std: float,
    source_set_perturbation_fraction: float,
) -> Batch:
    """Transform a batch of graphs by:

        1. applying random rotation augmentation,
        2. creating source-target splits,
        3. initializing stop tokens, and
        4. coupling positions in the target set with random positions.

    Args:
        batch (Batch): Batch of graphs.
        source_target_split (str): Source-target split mode.
        noise_std (float): Standard deviation of the initial Gaussian noise in the flow matching process.
        source_set_perturbation_std (float): Standard deviation of the Gaussian noise added to the source set positions.
        source_set_perturbation_fraction (float): Fraction of the source set positions to perturb.

    Returns:
        Batch: Transformed batch of graphs.
    """
    batch_size = batch.batch.max().item() + 1

    # (1) Apply random rotation augmentation (ligand + pocket share one rigid transform)
    rotation_augmentation = RandomRotationAugmentation()
    if (
        hasattr(batch, "pocket_pos")
        and batch.pocket_pos is not None
        and batch.pocket_pos.numel() > 0
        and hasattr(batch, "pocket_pos_batch")
    ):
        batch.pos, batch.pocket_pos = (
            rotation_augmentation.rotate_ligand_and_pocket_randomly(
                batch.pos,
                batch.batch,
                batch.pocket_pos,
                batch.pocket_pos_batch,
            )
        )
    else:
        batch.pos = rotation_augmentation.rotate_molecule_randomly(
            batch.pos, batch.batch
        )

    # (2.1) Create source-target split
    splitter = SourceTargetSplitter(splitting_mode=source_target_split)
    source_ptr, target_ptr = splitter.create_source_target_split(batch)

    # (2.2) Modify a random subset in the batch for start token prediction
    # For the first token, we have an empty source set and a full target set
    start_token_mask = torch.rand(batch_size) < 0.025
    corresponding_indices = torch.arange(batch.x.size(0), device=batch.x.device)[
        start_token_mask[batch.batch]
    ]
    mask = torch.isin(source_ptr, corresponding_indices)
    source_ptr = source_ptr[~mask]
    target_ptr = torch.unique(
        torch.cat((target_ptr, corresponding_indices)), sorted=True
    )

    # (2.3) Store source and target pointers in the batch object
    batch.source_ptr = source_ptr
    batch.target_ptr = target_ptr
    batch.start_token_mask = start_token_mask

    # (2.4) Perturb source set positions by adding Gaussian noise
    if source_set_perturbation_fraction is not None and source_set_perturbation_std is not None:
        source_set_noise = source_set_perturbation_std * torch.randn_like(
            batch.pos[batch.source_ptr]
        )
        dropout_mask = torch.rand_like(batch.source_ptr.float()) > source_set_perturbation_fraction
        source_set_noise[dropout_mask] *= 0.0
        batch.pos[batch.source_ptr] += source_set_noise

    # (2.5) Recenter positions w.r.t. the source set atoms
    mean_pos = global_mean_pool(
        batch.pos[batch.source_ptr], batch.batch[batch.source_ptr], size=batch_size
    )
    batch.pos = batch.pos - mean_pos[batch.batch]
    if (
        hasattr(batch, "pocket_pos")
        and batch.pocket_pos is not None
        and batch.pocket_pos.numel() > 0
        and hasattr(batch, "pocket_pos_batch")
    ):
        batch.pocket_pos = batch.pocket_pos - mean_pos[batch.pocket_pos_batch]

    # (3) Determine source sets with empty target sets, these have stop tokens
    target_set_mask = torch.zeros_like(
        batch.batch, device=batch.batch.device, dtype=torch.bool
    )
    target_set_mask[target_ptr] = 1
    batch_target = batch.batch[target_set_mask]
    stop_tokens = ~(torch.isin(torch.arange(0, len(batch)), torch.unique(batch_target)))
    batch.stop_tokens = stop_tokens

    # (4) Couple positions in the target set with random positions via linear sum assignment
    pos_target = batch.pos[target_set_mask]
    batch_target = batch_target.long()
    pos_random = noise_std * torch.randn_like(pos_target)
    target_idx = torch.unique(batch_target)
    for idx in target_idx:
        cost_matrix = torch.cdist(
            pos_target[batch_target == idx], pos_random[batch_target == idx], p=2
        )
        _, prior_idx = linear_sum_assignment(cost_matrix.cpu())

        # Reorder prior according to optimal assignment
        pos_random[batch_target == idx] = pos_random[batch_target == idx][prior_idx]
    batch.pos_random = pos_random

    return batch


def source_target_split_collate_fn(
    batch: list,
    source_target_split: str,
    noise_std: float,
    source_set_perturbation_std: float,
    source_set_perturbation_fraction: float,
    follow_batch: list[str] | None = None,
) -> Batch:
    fb = follow_batch or []
    if fb:
        batch = Batch.from_data_list(batch, follow_batch=fb)
    else:
        batch = Batch.from_data_list(batch)
    return source_target_split_batch_transform(
        batch,
        source_target_split,
        noise_std,
        source_set_perturbation_std,
        source_set_perturbation_fraction,
    )


class DataModule(LightningDataModule):
    """DataModule for loading and transforming the data for the NEAT model.

    Args:
        data_dir (str): Directory containing the data.
        data_set (str): Dataset to use ("QM9", "GEOM", "CROSSDOCKED", or "SPINDR"). Default is "QM9".
        batch_size (int): Batch size for the data loader. Default is 32.
        num_workers (int): Number of workers for the data loader. Default is 1.
        task (str): Task to perform ("neat" or "bond_prediction"). Default is "neat".
        flow_matching_noise_std (float): Standard deviation of the initial Gaussian noise in the flow matching process. Default is 1.4.
        source_target_split (str): Source-target split mode ("neighborhood" or "random"). Default is "neighborhood".
        source_set_perturbation_fraction (float): Fraction of source set nodes that receive random perturbation to their positions during training. Default is None.
        source_set_perturbation_std (float): Standard deviation of the Gaussian noise added to the perturbed source set positions during training. Default is None.
        bond_predictor_radius (float): Radius for the radius graph for bond prediction. Default is 2.5.
        bond_predictor_noise_ratio (float): For bond_prediction, fraction of radius for isotropic coordinate
            noise during training (e.g. 0.05 = 5%). 0 disables.

    Returns:
        DataModule for loading and transforming the data for the NEAT model.
    """

    def __init__(
        self,
        data_dir: str,
        data_set: str = "QM9",
        batch_size: int = 32,
        num_workers: int = 1,
        task: str = "neat",
        flow_matching_noise_std: float = 1.4,
        source_target_split: str = "neighborhood",
        source_set_perturbation_fraction: float = None,
        source_set_perturbation_std: float = None,
        bond_predictor_radius: float = 2.5,
        bond_predictor_noise_ratio: float = 0.0,
    ) -> None:
        super(DataModule, self).__init__()
        self.data_set = data_set.upper()
        self.data_path = os.path.join(data_dir, self.data_set)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.flow_matching_noise_std = flow_matching_noise_std
        self.source_target_split = source_target_split
        self.source_set_perturbation_fraction = source_set_perturbation_fraction
        self.source_set_perturbation_std = source_set_perturbation_std
        self.bond_predictor_radius = bond_predictor_radius
        self.bond_predictor_noise_ratio = bond_predictor_noise_ratio

        follow_pocket_batch = False
        if self.data_set == "QM9":
            self.vocab_size = len(QM9DataSet.VOCABULARY) + 1
            self.vocab = QM9DataSet.VOCABULARY
        elif self.data_set == "GEOM":
            self.vocab_size = len(GEOMDataSet.VOCABULARY) + 1
            self.vocab = GEOMDataSet.VOCABULARY
        elif self.data_set == "CROSSDOCKED":
            self.vocab_size = len(GEOMDataSet.VOCABULARY) + 1
            self.vocab = GEOMDataSet.VOCABULARY
            follow_pocket_batch = True
        elif self.data_set == "SPINDR":
            self.vocab_size = len(GEOMDataSet.VOCABULARY) + 1
            self.vocab = GEOMDataSet.VOCABULARY
            follow_pocket_batch = True
        else:
            raise ValueError(f"Unknown data_set: {self.data_set}")

        if self.task == "neat":
            self.source_target_split = source_target_split
            self.flow_matching_noise_std = flow_matching_noise_std
            self.source_target_split_fn = functools.partial(
                source_target_split_collate_fn,
                source_target_split=self.source_target_split,
                noise_std=self.flow_matching_noise_std,
                source_set_perturbation_std=self.source_set_perturbation_std,
                source_set_perturbation_fraction=self.source_set_perturbation_fraction,
                follow_batch=(["pocket_pos"] if follow_pocket_batch else None),
            )

        elif self.task == "bond_prediction":
            self.bond_prediction_train_fn = functools.partial(
                bond_prediction_collate_fn,
                radius=self.bond_predictor_radius,
                noise_ratio=self.bond_predictor_noise_ratio,
            )
            self.bond_prediction_eval_fn = functools.partial(
                bond_prediction_collate_fn,
                radius=self.bond_predictor_radius,
                noise_ratio=0.0,
            )

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            if str(self.data_set).upper() == "QM9":
                print("Using QM9 dataset.")
                self.full_data = QM9DataSet(self.data_path)
                splits = self.full_data.get_splits()
                self.training_data = self.full_data[splits["train"]]
                self.validation_data = self.full_data[splits["val"]]
                self.test_data = self.full_data[splits["test"]]
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")

            elif str(self.data_set).upper() == "GEOM":
                print("Using GEOM dataset.")
                self.training_data = GEOMDataSet(self.data_path, split="train")
                self.validation_data = GEOMDataSet(self.data_path, split="val")
                self.test_data = GEOMDataSet(self.data_path, split="test")
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")

            elif str(self.data_set).upper() == "CROSSDOCKED":
                print("Using CrossDocked dataset.")
                self.training_data = CrossDockedDataSet(self.data_path, split="train")
                self.validation_data = CrossDockedDataSet(self.data_path, split="val")
                self.test_data = CrossDockedDataSet(self.data_path, split="test")
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")

            elif str(self.data_set).upper() == "SPINDR":
                print("Using Spindr dataset.")
                self.training_data = SpindrDataSet(self.data_path, split="train")
                self.validation_data = SpindrDataSet(self.data_path, split="val")
                self.test_data = SpindrDataSet(self.data_path, split="test")
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")

            else:
                raise ValueError(f"Unknown data_set: {self.data_set}")

    def train_dataloader(self, shuffle_data=True) -> DataLoader:
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=shuffle_data,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=(
                self.source_target_split_fn
                if self.task == "neat"
                else self.bond_prediction_train_fn
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
            collate_fn=(
                self.source_target_split_fn
                if self.task == "neat"
                else self.bond_prediction_eval_fn
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=(
                self.source_target_split_fn
                if self.task == "neat"
                else self.bond_prediction_eval_fn
            ),
        )

    def full_dataloader(self) -> DataLoader:
        return DataLoader(
            self.full_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=(
                self.source_target_split_fn
                if self.task == "neat"
                else self.bond_prediction_eval_fn
            ),
        )
