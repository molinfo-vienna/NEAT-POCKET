"""Bond predictor GNN: given atom types and coordinates, predict bond types for edges."""

import math

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, radius_graph
from torch_geometric.transforms import Distance

# Bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
NUM_BOND_TYPES = 5


class BondPredictor(LightningModule):
    """GNN to predict bond types for edges in a molecular graph."""

    def __init__(self, **params) -> None:
        super().__init__()
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

        # GINEConv layers: message passing with edge features
        self.conv_layers = nn.ModuleList()
        for _ in range(n_conv_layers):
            nn_module = nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.ReLU(),
                nn.Dropout(self.hparams.dropout),
                nn.Linear(n_embd * 2, n_embd),
            )
            self.conv_layers.append(
                GINEConv(nn=nn_module, eps=0.0, train_eps=True, edge_dim=n_embd)
            )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.hparams.dropout)

        # Bond prediction head: [h_src; h_dst; dist] -> 5-way logits
        self.bond_mlp = nn.Sequential(
            nn.Linear(n_embd * 2 + 1, n_embd),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, NUM_BOND_TYPES),
        )
        if self.hparams.predict_charge:
            nn_module = nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.ReLU(),
                nn.Dropout(self.hparams.dropout),
                nn.Linear(n_embd * 2, n_embd),
            )
            self.charge_conv_layer = GINEConv(nn=nn_module, eps=0.0, train_eps=True, edge_dim=5)
            self.charge_mlp = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.ReLU(),
                nn.Dropout(self.hparams.dropout),
                nn.Linear(n_embd, 3),
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

        # (3) Message passing with GINEConv
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, edge_attr) + x
            x = F.relu(x)
        x = self.layer_norm(x)

        # (4) Bond prediction: [h_src; h_dst; dist]
        src, dst = data.edge_index[0], data.edge_index[1]
        h_src = x[src]
        h_dst = x[dst]
        edge_dist_scalar = self._get_edge_attr(data)  # [num_edges, 1]
        edge_features = torch.cat([h_src, h_dst, edge_dist_scalar], dim=-1)
        bond_logits = self.bond_mlp(edge_features)

        return bond_logits
        
    def forward_charges(self, data: Data) -> Tensor:
        """Forward pass for charge prediction."""

        atom_embedding = self.atom_type_embedding(data.x)
        atom_embedding = self.dropout(atom_embedding)
        edge_mask = [data.edge_labels != 0]
        edge_index = data.edge_index[:, edge_mask[0]]
        edge_labels = F.one_hot(data.edge_labels[edge_mask[0]], num_classes=5).float()
        charge_embedding = self.charge_conv_layer(atom_embedding, edge_index, edge_labels) + atom_embedding
        charge_embedding = F.relu(charge_embedding)
        charge_embedding = self.layer_norm(charge_embedding)
        charge_logits = self.charge_mlp(charge_embedding)
        
        return charge_logits  

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
        # edge_attr not set; _get_edge_attr will compute from pos

        logits = self(data)
        bond_types = logits.argmax(dim=1)
        pair_indices = data.edge_index.t()

        return bond_types, pair_indices
    
    @torch.no_grad()
    def predict_charges(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        bond_types: Tensor | None = None,
        pair_indices: Tensor | None = None,
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

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            
        data = Data(x=x, pos=pos, edge_labels=bond_types, edge_index=pair_indices.T, batch=batch)
        data = data.to(device)

        logits = self.forward_charges(data)
        charges = logits.argmax(dim=1) - 1

        return charges

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        bond_logits = self(batch)
        loss = F.cross_entropy(bond_logits, batch.edge_labels.long(), reduction="mean")
        self.log("train/bond_loss", loss, prog_bar=True, on_step=True)
        
        if self.hparams.predict_charge:
            charge_logits = self.forward_charges(batch)
            charge_loss = F.cross_entropy(charge_logits, batch.charge.long() + 1, reduction="mean")
            self.log("train/charge_loss", charge_loss, prog_bar=True, on_step=True)   
            loss = loss + charge_loss
            self.log("train/loss", loss, prog_bar=True, on_step=True)
        
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        bond_logits = self(batch)
        loss = F.cross_entropy(bond_logits, batch.edge_labels.long(), reduction="mean")
        self.log("val/bond_loss", loss, prog_bar=True, on_step=True)
        pred_bonds = bond_logits.argmax(dim=1)
        acc_bonds = (pred_bonds == batch.edge_labels).float().mean()
        self.log("val/acc_bonds", acc_bonds, prog_bar=True)
        
        if self.hparams.predict_charge:
            charge_logits = self.forward_charges(batch)
            charge_loss = F.cross_entropy(charge_logits, batch.charge.long() + 1, reduction="mean")
            self.log("val/charge_loss", charge_loss, prog_bar=True, on_step=True)   
            loss = loss + charge_loss
            self.log("val/loss", loss, prog_bar=True, on_step=True)
            pred_charges = charge_logits.argmax(dim=1) - 1
            acc_charges = (pred_charges == batch.charge).float().mean()
            self.log("val/acc_charges", acc_charges, prog_bar=True)
        
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
