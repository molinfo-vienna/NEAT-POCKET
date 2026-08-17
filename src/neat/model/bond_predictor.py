"""Bond predictor GNN: given atom types and coordinates, predict bond types for edges."""

import math

import torch
import torch.nn as nn
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, radius_graph
from torch_geometric.transforms import Distance

# Bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
NUM_BOND_TYPES = 4


class BondPredictor(LightningModule):
    """GNN to predict bond types for edges in a molecular graph."""

    def __init__(self, **params) -> None:
        super().__init__()
        self.hparams.setdefault("data_set", "SPINDR")
        self.save_hyperparameters()

        n_embd = self.hparams.n_embd
        n_conv_layers = self.hparams.n_conv_layers

        self.atom_type_embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=n_embd
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(1, n_embd // 2),
            nn.ReLU(),
            nn.Linear(n_embd // 2, n_embd),
        )

        self.net = SimpleNetwork(
            irreps_in=f"{n_embd}x0e",
            irreps_out=f"{n_embd}x0e",
            max_radius=2.5,
            layers=n_conv_layers,
            num_neighbors=20,
            num_nodes=5.0,
            pool_nodes=False,
        )

        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.hparams.dropout)

        self.bond_mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, NUM_BOND_TYPES),
        )

    def _get_edge_attr(self, data: Data) -> Tensor:
        """Get edge attributes (distances). Compute from pos if not in data."""
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            return edge_attr
        # Fallback: compute distances from positions
        data = Distance(norm=False)(data)
        return data.edge_attr

    def forward(self, data: Data) -> Tensor:
        """Forward pass.

        Args:
            data: PyG Batch with x, edge_index, edge_attr, edge_labels.

        Returns:
            bond_logits: [num_edges, 5] logits per edge.
        """
        if data.edge_index.shape[1] == 0:
            return torch.zeros(0, NUM_BOND_TYPES, device=data.x.device)

        # (1) Node features
        x = self.atom_type_embedding(data.x)
        x = self.dropout(x)

        # (2) Edge features
        edge_dist = self._get_edge_attr(data)
        edge_attr = self.edge_encoder(edge_dist)

        # (3) GNN with E(3) invariant features
        x = self.net({"x": x, "pos": data.pos, "batch": data.batch})

        # (4) Bond prediction
        src, dst = data.edge_index[0], data.edge_index[1]
        edge_features = x[src] + x[dst] + edge_attr
        bond_logits = self.bond_mlp(edge_features)

        return bond_logits

    @torch.no_grad()
    def predict_bonds(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        device: torch.device | None = None,
        radius: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict bond types for inference. Builds radius graph from pos/batch.

        Returns:
            bond_types: [num_edges] predicted class (0-4) per edge.
            pair_indices: [num_edges, 2] (src, dst) for each edge.
        """
        if device is None:
            device = x.device
        radius = radius or getattr(self.hparams, "radius", 2.5)

        data = Data(x=x, pos=pos)
        if batch is not None:
            data.batch = batch
        else:
            data.batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        data = data.to(device)

        data.edge_index = radius_graph(data.pos, r=radius, batch=data.batch, loop=False)

        logits = self(data)
        bond_probs = F.softmax(logits, dim=-1)
        no_bonds_mask = bond_probs[:, 0] > 0.99
        bond_probs = bond_probs[~no_bonds_mask]
        pair_indices = data.edge_index.t()
        pair_indices = pair_indices[~no_bonds_mask]

        return bond_probs, pair_indices

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        bond_logits = self(batch)
        loss = F.cross_entropy(bond_logits, batch.edge_labels.long(), reduction="mean")
        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        bond_logits = self(batch)
        loss = F.cross_entropy(bond_logits, batch.edge_labels.long(), reduction="mean")
        self.log("val/loss", loss, prog_bar=True, on_step=True)
        pred_bonds = bond_logits.argmax(dim=1)
        acc_bonds = (pred_bonds == batch.edge_labels).float().mean()
        self.log("val/acc_bonds", acc_bonds, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """AdamW; cosine LR schedule with warmup."""
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in self.named_parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.hparams.learning_rate)

        def lr_lambda(epoch):
            # Linear warmup, then cosine decay down to lr_min_ratio * base_lr
            if epoch < self.hparams.lr_warmup_epochs:
                return (epoch + 1) / (self.hparams.lr_warmup_epochs + 1)
            progress = (epoch - self.hparams.lr_warmup_epochs) / (
                self.hparams.max_epochs - self.hparams.lr_warmup_epochs
            )
            progress = min(progress, 1.0)
            return self.hparams.lr_min_ratio + (1 - self.hparams.lr_min_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
