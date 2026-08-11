"""Taken and modified from the nanoGPT repository:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm
from rdkit import Chem

from ..dataset.dataset_crossdocked import ATOM_VOCABULARY
from ..dataset.augmentation import RandomRotationAugmentation
from .attention import BidirectionalAttentionBlock
from .positional_encoding import FourierPositionEncoding
from .simple_mlp import SimpleMLPAdaLN


def _dataset_enables_cross_attention(data_set: str | None) -> bool:
    return str(data_set).upper() == "CROSSDOCKED" or str(data_set).upper() == "SPINDR"


class NEAT(LightningModule):
    """NEAT model for molecular generation using continuous flow matching."""

    def __init__(self, **params) -> None:
        super(NEAT, self).__init__()
        self.hparams.setdefault("noise_std", 1.0)
        self.hparams.setdefault("global_cond_proj", False)
        self.hparams.setdefault("cross_attn_with_null_token", False)
        self.hparams.setdefault("residue_pooling", "sum")
        self.hparams.setdefault("pocket_n_layer_atom_level", 1)
        self.hparams.setdefault("clash_penalty", 0)
        self.hparams.setdefault("clash_penalty_margin", 1.2)
        self.hparams.setdefault("freeze_pretrained", False)

        self.save_hyperparameters()

        # Atom type embedding layer
        self.atom_type_embedding_layer = nn.Embedding(
            num_embeddings=self.hparams.vocab_size,
            embedding_dim=self.hparams.n_embd,
        )

        # Learnable start token embedding
        self.start_token_embedding = nn.Parameter(
            torch.randn(1, self.hparams.n_embd) * 0.02
        )

        # Pocket residue type embedding layer
        self.hparams.setdefault("pocket_residue_vocab_size", 21)
        self.pocket_residue_type_embedding_layer = nn.Embedding(
            num_embeddings=self.hparams.pocket_residue_vocab_size,
            embedding_dim=self.hparams.n_embd,
        )

        # Fourier features for embedding of Cartesian coordinates
        self.fourier_embedding_layer = FourierPositionEncoding(
            out_dim=self.hparams.n_embd
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.hparams.dropout)

        # Transformer blocks for the ligand stream
        self.enable_cross_attention = _dataset_enables_cross_attention(self.hparams.data_set)
        scale_shift_weights = False if self.hparams.global_cond_proj else True
        self.transformer_blocks = nn.ModuleList(
            [
                BidirectionalAttentionBlock(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.dropout,
                    self.hparams.bias,
                    enable_cross_attention=self.enable_cross_attention,
                    scale_shift_weights=scale_shift_weights,
                )
                for _ in range(self.hparams.n_layer)
            ]
        )

        # Layer normalization after the transformer blocks
        self.layer_norm_after_transformer_blocks = nn.LayerNorm(
            self.hparams.n_embd, bias=False
        )

        if self.enable_cross_attention:
            self.atom_type_embedding_layer_pocket = nn.Embedding(
                num_embeddings=self.hparams.vocab_size,
                embedding_dim=self.hparams.n_embd,
            )

            # Fourier features for embedding of Cartesian coordinates
            self.fourier_embedding_layer_pocket = FourierPositionEncoding(
                out_dim=self.hparams.n_embd
            )

            # Transformer blocks for the pocket stream
            self.hparams.setdefault("pocket_n_layer", self.hparams.n_layer)
            self.atom_level_pocket_transformer_blocks = nn.ModuleList(
                [
                    BidirectionalAttentionBlock(
                        self.hparams.n_embd,
                        self.hparams.n_head,
                        self.hparams.dropout,
                        self.hparams.bias,
                    )
                    for _ in range(self.hparams.pocket_n_layer_atom_level)
                ]
            )

            self.residue_level_pocket_transformer_blocks = nn.ModuleList(
                [
                    BidirectionalAttentionBlock(
                        self.hparams.n_embd,
                        self.hparams.n_head,
                        self.hparams.dropout,
                        self.hparams.bias,
                    )
                    for _ in range(self.hparams.pocket_n_layer)
                ]
            )

            self.atom_level_pocket_transformer_blocks_2 = nn.ModuleList(
                [
                    BidirectionalAttentionBlock(
                        self.hparams.n_embd,
                        self.hparams.n_head,
                        self.hparams.dropout,
                        self.hparams.bias,
                    )
                    for _ in range(self.hparams.pocket_n_layer_atom_level)
                ]
            )

            # Layer normalization after the pocket transformer blocks
            self.layer_norm_after_atom_level_pocket_transformer_blocks = nn.LayerNorm(
                self.hparams.n_embd, bias=False
            )
            self.layer_norm_after_residue_level_pocket_transformer_blocks = (
                nn.LayerNorm(self.hparams.n_embd, bias=False)
            )
            self.layer_norm_after_atom_level_pocket_transformer_blocks_2 = nn.LayerNorm(
                self.hparams.n_embd, bias=False
            )
            self.null_condition_embedding = nn.Parameter(
                torch.zeros(1, self.hparams.n_embd)
            )
            if self.hparams.global_cond_proj:
                self.global_condition_projection = nn.Sequential(
                    nn.Linear(self.hparams.n_embd, self.hparams.n_embd, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.hparams.n_embd, self.hparams.n_embd, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.hparams.n_embd, self.hparams.n_embd * 8, bias=False),
                )

        # Linear prediction head for atom type prediction
        self.atom_type_prediction_head = nn.Linear(
            self.hparams.n_embd,
            self.hparams.vocab_size,
            bias=self.hparams.bias,
        )

        # Init all weights (taken from the nanoGPT repository)
        self.apply(self._init_weights)
        # Apply special scaled initialization to the residual projections
        # (taken from the nanoGPT repository)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.c_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.hparams.n_layer)
                )
            if (
                pn.endswith("attn_cross.c_proj.weight")
                or pn.endswith("scale_shift.2.weight")
                or pn.endswith("global_condition_projection.4.weight")
            ):
                nn.init.constant_(p, 0)

        # This is the Diffusion MLP with AdaLN conditioning.s
        # It was used in the original diffusion loss paper, and QUETZAL also uses it.
        self.ada_mlp = SimpleMLPAdaLN(
            model_channels=self.hparams.n_embd_fm,  # model hidden width
            condition_channels=self.hparams.n_embd,  # dimension of conditioning vector c
            fourier_features_channels=512,  # number of Fourier channels for coord embedding
            fourier_features_bandwidth=20.0,  # frequency bandwidth for Fourier features
            n_layer_mlp=self.hparams.n_layers_fm,  # number of residual blocks)
        )
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights as in NanoGPT"""
        if isinstance(module, nn.Linear):
            # std is chosen w.r.t. sqrt(embd_dim)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def initialize_from_pretrained_model(self, pretrained_model: nn.Module):
        """
        Initializes a structurally modified conditional model using weights from a
        pretrained unconditional model, freezing the pretrained parameters while keeping
        new conditioning layers and LoRA modules trainable.
        """
        # 1. Extract state dicts
        pretrained_state_dict = pretrained_model.state_dict()
        current_state_dict = self.state_dict()

        matched_keys = []
        unmatched_keys = []

        # 2. Load matching weights
        with torch.no_grad():
            for key, value in pretrained_state_dict.items():
                if key in current_state_dict:
                    if value.shape == current_state_dict[key].shape:
                        current_state_dict[key].copy_(value)
                        matched_keys.append(key)
                    else:
                        print(
                            f"[Warning] Shape mismatch for key '{key}': "
                            f"Pretrained {value.shape} vs Conditional {current_state_dict[key].shape}. Skipping."
                        )
                        unmatched_keys.append(key)
                else:
                    unmatched_keys.append(key)

        print(f"Successfully transferred {len(matched_keys)} parameter tensors.")

        if self.hparams.freeze_pretrained:
            unfrozen_layers = [
                "atom_type_embedding_layer_pocket",
                "fourier_embedding_layer_pocket",
                "atom_level_pocket_transformer_blocks",
                "residue_level_pocket_transformer_blocks",
                "layer_norm_after_atom_level_pocket_transformer_blocks",
                "layer_norm_after_residue_level_pocket_transformer_blocks",
                "null_condition_embedding",
                "global_condition_projection",
                "pocket_residue_type_embedding_layer",
                "attn_cross",
                "ln_2",
                "scale_shift",
            ]
            for name, param in self.named_parameters():
                is_new_layer = any(layer in name for layer in unfrozen_layers)

                if is_new_layer:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(
            f"Model configured: {trainable_params:,} trainable params | {frozen_params:,} frozen params."
        )

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the NEAT model.

        Args:
            data: Batch of data.

        Returns:
            Tuple of total loss, atom type prediction loss, and flow matching loss.
        """
        device = data.x.device
        batch_size = data.batch.max().item() + 1

        # We split the molecular data into source and target atom sets.
        # The indexing tensors point to the same molecules as in the original batch.
        # The source set contains at least one atom, and at most all atoms.
        # If it contains all atoms, then the target set will be empty.
        # The stop tokens mask indicates which molecules have empty target sets.

        # (1) Compute the representation of the source atom sets with the transformer
        if self.enable_cross_attention:
            pocket_info = {
                "pocket_x": data.pocket_x,
                "pocket_pos": data.pocket_pos,
                "pocket_residue_id": data.pocket_residue_id,
                "pocket_residue_type": data.pocket_residue_type,
                "pocket_batch": data.pocket_pos_batch,
            }
        else:
            pocket_info = None
            
        # Create mask for CFG dropout
        cfg_dropout = self.hparams.get("cfg_dropout", None)
        if self.training and pocket_info is not None and cfg_dropout is not None:
            cfg_mask = torch.rand(batch_size, device=device) < cfg_dropout
        else:
            cfg_mask = None

        source_set_representation = self.compute_source_set_representation(
            data.x[data.source_ptr],
            data.pos[data.source_ptr],
            data.batch[data.source_ptr],
            batch_size,
            device,
            pocket_info if self.enable_cross_attention else None,
            cfg_mask=cfg_mask,
        )  # [batch_size, n_embd]

        # (2) Compute the logits for the atom type prediction
        logits = self.atom_type_prediction_head(
            source_set_representation
        )  # [n_target_sets, vocab_size]

        # (3) Calculate a cross-entropy loss for atom type prediction
        loss_ce = self.compute_atom_type_loss(
            logits,
            data.x[data.target_ptr],
            data.batch[data.target_ptr],
            data.stop_tokens,
            device,
        )

        # (4) Calculate a flow matching loss for the target atom positions
        loss_fm, clash_penalty = self.compute_flow_matching_loss(
            data.x[data.target_ptr],
            data.pos[data.target_ptr],
            data.pos_random,
            data.batch[data.target_ptr],
            source_set_representation,
            data.start_token_mask,
            device,
            pocket_info,
            cfg_mask=cfg_mask,
        )

        # (5) Add the losses together
        if clash_penalty is not None:
            loss = loss_ce + loss_fm + clash_penalty

            return loss, loss_ce, loss_fm, clash_penalty
        else:
            loss = loss_ce + loss_fm

            return loss, loss_ce, loss_fm, None

    def compute_source_set_representation(
        self,
        x_source: Tensor,
        pos_source: Tensor,
        batch_source: Tensor,
        batch_size: int,
        device: torch.device,
        pocket_info: dict[str, Tensor] | None = None,
        cfg_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the representation of the source atom sets.

        Args:
            x_source (Tensor): The atom types of the source atoms. shape: [n_source_atoms]
            pos_source (Tensor): The positions of the source atoms. shape: [n_source_atoms, 3]
            batch_source (Tensor): The batch indices of the source atoms. shape: [n_source_atoms]
            device (torch.device): The device to use for computations.
            pocket_info (dict[str, Tensor] | None): Information about the pocket atoms.

        Returns:
            Tensor: The representation of the source atom sets. shape: [batch_size, n_embd]
        """
        x_source = x_source.to(device).long()
        pos_source = pos_source.to(device).float()
        batch_source = batch_source.to(device).long()

        # (0) Process the pocket data if it is provided
        if pocket_info is not None:
            pocket_x = pocket_info["pocket_x"].to(device).long()
            pocket_pos = pocket_info["pocket_pos"].to(device).float()
            pocket_residue_id = pocket_info["pocket_residue_id"].to(device).long()
            pocket_residue_type = pocket_info["pocket_residue_type"].to(device).long()
            pocket_batch = pocket_info["pocket_batch"].to(device).long()

            # Make the pocket_residue_id continuous across the batch
            # e.g. [0,0,0,1,1,2,2,2,0,0,1,1] -> [0,0,0,1,1,2,2,2,3,3,4,4]
            num_res_per_graph = torch.zeros(batch_size, device=device, dtype=torch.long)
            for graph_idx in range(batch_size):
                mask_g = pocket_batch == graph_idx
                if mask_g.any():
                    num_res_per_graph[graph_idx] = (
                        int(pocket_residue_id[mask_g].max().item()) + 1
                    )
            id_offsets = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    torch.cumsum(num_res_per_graph[:-1], dim=0),
                ]
            )
            pocket_residue_id = pocket_residue_id + id_offsets[pocket_batch]

        # (1) Compute atom counts of the source sets
        atom_count_source = torch.bincount(
            batch_source, minlength=batch_size
        )  # [batch_size]

        # (2) Reshape the input to [batch_size, max_atom_count, n_embd].
        # This could also be done with sequence packing, but for now we keep it simple.
        # The output tensor is padded with zeros for all source sets with less atoms
        # than the largest source atom set in the batch. The atom mask keeps track of
        # which entries correspond to atoms and padding.
        dim = [len(atom_count_source), atom_count_source.max(), self.hparams.n_embd]
        x = torch.zeros(dim, device=device)  # [batch_size, max_atom_count, n_embd]
        context_range = torch.arange(
            atom_count_source.max(), device=atom_count_source.device
        ).unsqueeze(0)
        atom_mask = context_range < atom_count_source.unsqueeze(
            1
        )  # [batch_size, max_atom_count]

        # (3) Embed the atom types and positions
        atom_type_embedding = self.atom_type_embedding_layer(
            x_source
        )  # [n_source_atoms, n_embd]
        positional_embedding = self.fourier_embedding_layer(
            pos_source
        )  # [n_source_atoms, n_embd]

        # (4) Combine the atom type embedding and the positional embedding
        input_embedding = (
            atom_type_embedding + positional_embedding
        )  # [n_source_atoms, n_embd]

        # (5) Apply the dropout layer
        input_embedding = self.dropout_layer(
            input_embedding
        )  # [n_source_atoms, n_embd]

        # (6) Apply the atom mask to the input embedding
        x[atom_mask] = input_embedding  # [batch_size, max_atom_count, n_embd]

        # (7) Concatenate start token embedding to the transformer input.
        x = torch.cat(
            [self.start_token_embedding.expand(batch_size, 1, -1), x], dim=1
        )  # [batch_size, max_atom_count + 1, n_embd]
        atom_mask = torch.cat(
            [torch.ones(batch_size, 1, device=device, dtype=torch.bool), atom_mask],
            dim=1,
        )  # [batch_size, max_atom_count + 1]

        # (8) Create the attention mask for the transformer blocks.
        # The attention mask is used in the transformer blocks and is the outer product of the atom mask.
        attn_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(
            2
        )  # [batch_size, max_atom_count, max_atom_count]
        attn_mask = attn_mask.unsqueeze(1).expand(
            -1, self.hparams.n_head, -1, -1
        )  # [batch_size, n_head, max_atom_count, max_atom_count]

        if pocket_info is not None:
            # (9) Encode the pocket residues with a lightweight transformer
            x_residues, residue_mask = self.compute_residue_representations(
                batch_size=batch_size,
                device=device,
                pocket_x=pocket_x,
                pocket_pos=pocket_pos,
                pocket_residue_id=pocket_residue_id,
                pocket_residue_type=pocket_residue_type,
                pocket_batch=pocket_batch,
            )
            ada_ln_condition = x_residues.sum(dim=1)  # [batch_size, n_embd]

            if cfg_mask is not None:
                x_residues[cfg_mask] = 0
                residue_mask[cfg_mask] = 0
                ada_ln_condition[cfg_mask] = self.null_condition_embedding
                if self.hparams.cross_attn_with_null_token:
                    x_residues[cfg_mask, 0] = self.null_condition_embedding
                    residue_mask[cfg_mask, 0] = 1

            # (10) Create a cross-attention mask for the pocket residues
            cross_attn_mask = residue_mask.unsqueeze(1) * atom_mask.unsqueeze(
                2
            )  # [batch_size, max_residue_count, max_atom_count]
            cross_attn_mask = cross_attn_mask.unsqueeze(1).expand(
                -1, self.hparams.n_head, -1, -1
            )  # [batch_size, n_head, max_residue_count, max_atom_count]
        elif self.enable_cross_attention:
            x_residues = None
            cross_attn_mask = None
            ada_ln_condition = self.null_condition_embedding.expand(batch_size, -1)
            if self.hparams.cross_attn_with_null_token:
                x_residues = self.null_condition_embedding.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [batch_size, 1, n_embd]
                residue_mask = torch.ones(
                    (batch_size, 1), device=device, dtype=torch.bool
                )  # [batch_size, 1]
                # (10) Create a cross-attention mask for the pocket residues
                cross_attn_mask = residue_mask.unsqueeze(1) * atom_mask.unsqueeze(
                    2
                )  # [batch_size, max_residue_count, max_atom_count]
                cross_attn_mask = cross_attn_mask.unsqueeze(1).expand(
                    -1, self.hparams.n_head, -1, -1
                )  # [batch_size, n_head, max_residue_count, max_atom_count]
        else:
            x_residues = None
            cross_attn_mask = None
            ada_ln_condition = None

        if self.hparams.global_cond_proj:
            ada_ln_condition = self.global_condition_projection(
                ada_ln_condition
            )  # [batch_size, n_embd * 8]

        # (12) Pass through the transformer blocks
        for block in self.transformer_blocks:
            x = block(
                x,
                attn_mask=attn_mask,
                cross_attn_input=x_residues,
                cross_attn_mask=cross_attn_mask,
                ada_ln_condition=ada_ln_condition,
            )  # [batch_size, max_atom_count + 1, n_embd]

        x = self.layer_norm_after_transformer_blocks(
            x
        )  # [batch_size, max_atom_count + 1, n_embd]

        # (10) Apply the atom mask to the input embedding (not really needed, could be removed)
        x = x * atom_mask.unsqueeze(-1)  # [batch_size, max_atom_count + 1, n_embd]

        # (11) Pool the atom embeddings into a molecule embedding
        source_set_representation = x.sum(dim=1)  # [batch_size, n_embd]

        return source_set_representation

    def compute_residue_representations(
        self,
        batch_size: int,
        device: torch.device,
        pocket_x: Tensor | None = None,
        pocket_pos: Tensor | None = None,
        pocket_residue_id: Tensor | None = None,
        pocket_residue_type: Tensor | None = None,
        pocket_batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode pocket atoms into contextualized residue embeddings.

        Returns padded tokens and token mask with shape [B, R_max, C] and [B, R_max].
        """

        # 0. Early exit if the batch is empty or if the pocket data is missing.
        if batch_size == 0:
            return (
                torch.zeros(0, 0, self.hparams.n_embd, device=device),
                torch.zeros(0, 0, dtype=torch.bool, device=device),
            )

        if (
            pocket_x is None
            or pocket_pos is None
            or pocket_residue_id is None
            or pocket_residue_type is None
            or pocket_batch is None
            or pocket_x.numel() == 0
        ):
            return (
                torch.zeros(batch_size, 0, self.hparams.n_embd, device=device),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            )

        # (1) Atom-level embeddings
        atom_embedding = self.atom_type_embedding_layer_pocket(pocket_x)
        pos_embedding = self.fourier_embedding_layer_pocket(pocket_pos)
        residue_type_embedding = self.pocket_residue_type_embedding_layer(
            pocket_residue_type
        )
        atom_tokens = self.dropout_layer(
            atom_embedding + pos_embedding + residue_type_embedding
        )  # shape: [n_atoms, n_embd]

        # (2) Create attention mask on a residue level
        atom_count_per_residue = torch.bincount(pocket_residue_id)
        dim = [
            len(atom_count_per_residue),
            atom_count_per_residue.max(),
            self.hparams.n_embd,
        ]
        x_atom = torch.zeros(
            dim, device=device
        )  # [num_residues, max_atom_count_per_residue, n_embd]
        context_range = torch.arange(
            atom_count_per_residue.max(), device=atom_count_per_residue.device
        ).unsqueeze(0)
        atom_mask = context_range < atom_count_per_residue.unsqueeze(1)
        attn_mask_atom = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(
            2
        )  # [num_residues, max_atom_count_per_residue, max_atom_count_per_residue]
        attn_mask_atom = attn_mask_atom.unsqueeze(1).expand(
            -1, self.hparams.n_head, -1, -1
        )  # [num_residues, n_head, max_atom_count_per_residue, max_atom_count_per_residue]

        # (3) Pass throught atom-level transformer blocks
        x_atom[atom_mask] = (
            atom_tokens  # [num_residues, max_atom_count_per_residue, n_embd]
        )
        for block in self.atom_level_pocket_transformer_blocks:
            x_atom = block(
                x_atom, attn_mask=attn_mask_atom, pos=None
            )  # [num_residues, max_atom_count_per_residue, n_embd]
        x_atom = self.layer_norm_after_atom_level_pocket_transformer_blocks(
            x_atom
        )  # [num_residues, max_atom_count_per_residue, n_embd]
        x_atom = x_atom * atom_mask.unsqueeze(
            -1
        )  # [num_residues, max_atom_count_per_residue, n_embd]

        # (4) Pool atom-level tokens into residue-level tokens via sum pooling
        if self.hparams.residue_pooling == "mean":
            residue_tokens = x_atom.sum(dim=1) / atom_count_per_residue.unsqueeze(-1)
        elif self.hparams.residue_pooling == "sum":
            residue_tokens = x_atom.sum(dim=1)  # [num_residues, n_embd]
        else:
            raise ValueError(
                f"Invalid residue_pooling: {self.hparams.residue_pooling}. Must be 'mean' or 'sum'."
            )

        # (5) Now we need a residue-level attention mask
        ptr = torch.cat(
            [
                torch.tensor([0], device=atom_count_per_residue.device),
                torch.cumsum(atom_count_per_residue[:-1], dim=0),
            ]
        )
        residue_idx = pocket_batch[ptr]
        residue_count = torch.bincount(residue_idx)

        dim = [len(residue_count), residue_count.max(), self.hparams.n_embd]
        x_residues = torch.zeros(
            dim, device=device
        )  # [batch_size, max_residue_count, n_embd]
        context_range_residues = torch.arange(
            residue_count.max(), device=residue_count.device
        ).unsqueeze(0)
        residue_mask = context_range_residues < residue_count.unsqueeze(1)
        attn_mask_residues = residue_mask.unsqueeze(1) * residue_mask.unsqueeze(
            2
        )  # [num_residues, max_atom_count_per_residue, max_atom_count_per_residue]
        attn_mask_residues = attn_mask_residues.unsqueeze(1).expand(
            -1, self.hparams.n_head, -1, -1
        )  # [num_residues, n_head, max_atom_count_per_residue, max_atom_count_per_residue]
        x_residues[residue_mask] = (
            residue_tokens  # [batch_size, max_residue_count, n_embd]
        )

        # (6) Pass through the residue-level transformer blocks
        for block in self.residue_level_pocket_transformer_blocks:
            x_residues = block(
                x_residues,
                attn_mask=attn_mask_residues,
            )  # [batch_size, max_residue_count, n_embd]

        x_residues = self.layer_norm_after_residue_level_pocket_transformer_blocks(
            x_residues
        )  # [batch_size, max_residue_count, n_embd]
        x_residues = x_residues * residue_mask.unsqueeze(
            -1
        )  # [batch_size, max_residue_count, n_embd]

        # (7) Now we go back to residue-level attention on the atom level
        x_atom[atom_mask] = (
            x_atom[atom_mask] + x_residues[residue_mask][pocket_residue_id]
        )

        # (8) Pass through another round of atom-level transformer blocks
        for block in self.atom_level_pocket_transformer_blocks_2:
            x_atom = block(
                x_atom,
                attn_mask=attn_mask_atom,
            )  # [num_residues, max_atom_count_per_residue, n_embd]
        x_atom = self.layer_norm_after_atom_level_pocket_transformer_blocks_2(
            x_atom
        )  # [num_residues, max_atom_count_per_residue, n_embd]
        x_atom = x_atom * atom_mask.unsqueeze(
            -1
        )  # [num_residues, max_atom_count_per_residue, n_embd]

        # Now we need to reshape the atom-level tokens according to the batch size
        atom_count_per_residue = torch.bincount(pocket_batch)
        dim = [
            len(atom_count_per_residue),
            atom_count_per_residue.max(),
            self.hparams.n_embd,
        ]
        x_final = torch.zeros(
            dim, device=device
        )  # [num_residues, max_atom_count_per_residue, n_embd]
        context_range = torch.arange(
            atom_count_per_residue.max(), device=atom_count_per_residue.device
        ).unsqueeze(0)
        atom_mask_final = context_range < atom_count_per_residue.unsqueeze(1)

        x_final[atom_mask_final] = x_atom[atom_mask]

        return x_final, atom_mask_final.clone()

    def compute_atom_type_loss(
        self,
        logits: Tensor,
        x_target: Tensor,
        batch_target: Tensor,
        stop_tokens: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Compute the atom type prediction loss.

        Args:
            logits (Tensor): The logits of the atom type predictions. shape: [n_target_sets, vocab_size]
            x_target (Tensor): The target atom types. shape: [n_target_atoms]
            batch_target (Tensor): The batch indices of the target atoms. shape: [n_target_atoms]
            stop_tokens (Tensor): The stop tokens. shape: [batch_size]
            device (torch.device): The device to use for computations.

        Returns:
            Tensor: The atom type prediction loss. shape: [1]

        """
        logits = logits.to(device)
        x_target = x_target.to(device)
        batch_target = batch_target.to(device)
        stop_tokens = stop_tokens.to(device)

        # Atom type prediction is done with a cross-entropy loss.
        # Importantly, since we can have multiple atoms in the target set per source set,
        # we are modelling a target type *distribution*. This distribution is the mean
        # over the one-hot encodings of the target atom types.

        # (1) Map target atom indices to contiguous indices to avoid errors in the aggregation step.
        _, batch_target_contiguous = torch.unique(
            batch_target.clone(), return_inverse=True
        )  # [n_target_atoms]
        # (2) Take the mean over the one-hot encodings of the target atom types
        x_target_prob = F.one_hot(
            x_target.long(), self.hparams.vocab_size
        ).float()  # [n_target_atoms, vocab_size]
        x_target_prob = global_mean_pool(
            x_target_prob.float(), batch_target_contiguous
        )  # [n_target_sets, vocab_size]

        # (3) Incorporate the stop tokens into the target type distributions
        combined_prob = torch.zeros(
            (stop_tokens.shape[0], self.hparams.vocab_size),
            dtype=torch.float,
            device=device,
        )
        combined_prob[stop_tokens, 0] = 1.0
        combined_prob[~stop_tokens] = x_target_prob

        # (4) Compute the cross-entropy loss between predicted logits and target type distributions
        loss_ce = F.cross_entropy(
            logits,
            combined_prob,
            reduction="mean",
        )  # [1]

        return loss_ce

    def compute_flow_matching_loss(
        self,
        x_target: Tensor,
        pos_target: Tensor,
        pos_random: Tensor,
        batch_target: Tensor,
        source_set_representation: Tensor,
        start_token_mask: Tensor,
        device: torch.device,
        pocket_info: dict[str, Tensor] | None = None,
        cfg_mask: Tensor | None = None,
        resampling=4,
    ) -> Tensor:
        """Compute the flow matching loss.

        Args:
            x_target (Tensor): The target atom types. shape: [n_target_atoms]
            pos_target (Tensor): The target positions. shape: [n_target_atoms, 3]
            batch_target (Tensor): The batch indices of the target atoms. shape: [n_target_atoms]
            stop_tokens (Tensor): The stop tokens. shape: [batch_size]
            source_set_representation (Tensor): The representation of the source sets. shape: [batch_size, n_embd]
            device (torch.device): The device to use for computations.
            resampling (int): The number of resampling steps.

        Returns:
            Tensor: The flow matching loss. shape: [1]
        """
        x_target = x_target.to(device)
        pos_target = pos_target.to(device)
        batch_target = batch_target.to(device)
        source_set_representation = source_set_representation.to(device)
        pos_random = pos_random.to(device)
        batch_target = batch_target.long()

        n_paths = pos_target.shape[0]

        # Note: Coupling via linear sum assignment is done in the DataLoader

        # (1) Interpolation: t = 0 --> pos_random, t=1 --> target_pos
        interpolation = pos_target - pos_random  # [n_paths, 3]

        # (2) For each path, draw k random time steps
        resampling = self.hparams.time_step_resampling
        if self.hparams.time_step_sampling == "uniform":
            time_step = self.sample_timesteps_uniform(
                n_paths * resampling, device=device
            )  # [n_paths * k]
        elif self.hparams.time_step_sampling == "logit_normal":
            time_step = 0.98 * self.sample_timesteps_logit_normal(
                n_paths * resampling, device=device, m=0.8, s=1.7
            ) + 0.02 * self.sample_timesteps_uniform(
                n_paths * resampling, device=device
            )  # [n_paths * k]
        else:
            raise ValueError(
                f"Invalid time_step_sampling: {self.hparams.time_step_sampling}. Must be 'uniform' or 'logit_normal'."
            )

        # (3) Since we sample k time steps per path, we need to expand all other tensors accordingly
        x_target = torch.cat([x_target for _ in range(resampling)], dim=0)
        pos_random = torch.cat([pos_random for _ in range(resampling)], dim=0)
        pos_target = torch.cat([pos_target for _ in range(resampling)], dim=0)
        interpolation = torch.cat([interpolation for _ in range(resampling)], dim=0)
        source_set_representations = source_set_representation[batch_target]
        source_set_representations = torch.cat(
            [source_set_representations for _ in range(resampling)], dim=0
        )
        start_token_mask = start_token_mask[batch_target]
        start_token_mask = torch.cat(
            [start_token_mask for _ in range(resampling)], dim=0
        )
        scaling_factor = torch.ones_like(start_token_mask, dtype=torch.float)
        scaling_factor[start_token_mask] = 0.1

        # (4) Calculate k interpolated positions per path given the sampled time steps
        interpolated_pos = pos_random + interpolation * time_step.unsqueeze(
            1
        )  # [n_paths * k, 3]

        # (5) Compute the vector field output of the flow network at the
        # interpolated positions and time steps.
        output_fm = self.compute_vector_field(
            x_target,
            interpolated_pos,
            time_step,
            source_set_representations,
            device,
        )  # [n_paths * k, 3]

        # (7) Compute the flow matching loss.
        # This is the MSE between the predicted vector field and
        # the interpolation (pos_1 - pos_0) for each path.
        loss_fm = torch.mean((output_fm - interpolation) ** 2, dim=1)  # [n_paths * k]
        loss_fm = loss_fm * scaling_factor  # [n_paths * k]

        def batch_to_dense(x: Tensor, batch: Tensor, batch_size: int, n_embd: int) -> tuple[Tensor, Tensor]:
            atom_count_source = torch.bincount(
                batch, minlength=batch_size
            )  # [batch_size]
            dim = [len(atom_count_source), atom_count_source.max(), n_embd]
            dense = torch.zeros(dim, device=device)  # [batch_size, max_atom_count, n_embd]
            context_range = torch.arange(
                atom_count_source.max(), device=atom_count_source.device
            ).unsqueeze(0)
            atom_mask = context_range < atom_count_source.unsqueeze(
                1
            )  # [batch_size, max_atom_count]
            dense[atom_mask] = x  # [batch_size, max_atom_count, n_embd]
            return dense, atom_mask

        if pocket_info is not None and cfg_mask is not None and self.hparams.clash_penalty > 0:
            # Prepare pocket information
            _periodic_table = Chem.GetPeriodicTable()
            atom_types = list(ATOM_VOCABULARY.keys())
            atom_radii = torch.tensor([_periodic_table.GetRvdw(atom_type) for atom_type in atom_types], device=device)
            pocket_x = torch.cat([pocket_info["pocket_x"] for _ in range(resampling)], dim=0)
            pocket_pos = torch.cat([pocket_info["pocket_pos"] for _ in range(resampling)], dim=0)
            batch_size = pocket_info["pocket_batch"].max().item() + 1
            pocket_batch = torch.cat([pocket_info["pocket_batch"] + i * batch_size for i in range(resampling)], dim=0)
            batch_target = torch.cat([batch_target + i * batch_size for i in range(resampling)], dim=0)
            cfg_mask = torch.cat([cfg_mask for _ in range(resampling)], dim=0)
            
            # Compute the projected positions of the target atoms at the sampled time steps
            pos_projected = interpolated_pos + output_fm * (1 - time_step).unsqueeze(1)
            pocket_atom_radii = atom_radii[pocket_x - 1]
            x_next_atom_radii = atom_radii[x_target - 1]
            
            # For computing the distances, we need to convert to dense format
            pos_projected_dense, _ = batch_to_dense(pos_projected, batch_target, batch_size*resampling, 3)
            pos_pocket_dense, _ = batch_to_dense(pocket_pos, pocket_batch, batch_size*resampling, 3)
            dist = (pos_projected_dense.unsqueeze(1) - pos_pocket_dense.unsqueeze(2)).norm(dim=3)
            
            # We also need to compute the pairwise sum of vdw radii
            margin = self.hparams.clash_penalty_margin
            x_next_atom_radii_dense, atom_mask = batch_to_dense(x_next_atom_radii.unsqueeze(1), batch_target, batch_size*resampling, 1)
            pocket_atom_radii_dense, pocket_mask = batch_to_dense(pocket_atom_radii.unsqueeze(1), pocket_batch, batch_size*resampling, 1)
            vdw_radii_sum = (x_next_atom_radii_dense.unsqueeze(1) + pocket_atom_radii_dense.unsqueeze(2) + margin).squeeze(-1)
            mask = (atom_mask.unsqueeze(1) & pocket_mask.unsqueeze(2)).squeeze(-1)
            
            # Now we can compute the penalty for steric clashes
            penalty = torch.clamp(vdw_radii_sum - dist, min=0.0)**2
            penalty[~mask] = 0.0  # Ignore distances that are not valid (i.e., where there is no atom)
            penalty = penalty.sum(dim=1) # Add up contributions per atom
            penalty = penalty[atom_mask]
            non_masked_idx = torch.nonzero(~cfg_mask).flatten()
            final_mask = torch.isin(batch_target, non_masked_idx)
            penalty = penalty[final_mask]
            penalty = penalty.mean() * self.hparams.clash_penalty
            
            return loss_fm.mean(), penalty
        else:
            # (9) Return the mean loss over all paths and time steps.
            return loss_fm.mean(), None  

    def sample_timesteps_uniform(
        self, num_samples: int, device: torch.device
    ) -> Tensor:
        """Sample timesteps from a uniform distribution.

        Args:
            num_samples (int): The number of timesteps to sample.
            device (torch.device): The device to use for computations.

        Returns:
            Tensor: The sampled timesteps. shape: [num_samples]
        """
        return torch.rand(num_samples, device=device)

    def sample_timesteps_logit_normal(
        self, num_samples: int, device: torch.device, m: float = 0.8, s: float = 1.7
    ) -> Tensor:
        """Sample timesteps from a logit-normal distribution.

        Adapated from https://arxiv.org/pdf/2403.03206.pdf

        Args:
            num_samples (int): The number of timesteps to sample.
            device (torch.device): The device to use for computations.
            m (float): The mean of the logit-normal distribution.
            s (float): The standard deviation of the logit-normal distribution.

        Returns:
            Tensor: The sampled timesteps. shape: [num_samples]
        """
        u = torch.randn(num_samples, device=device) * s + m
        t = 1 / (1 + torch.exp(-u))
        return t

    def compute_vector_field(
        self,
        x: Tensor,
        pos_t: Tensor,
        time_step: Tensor,
        source_set_representation: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Compute the vector field of the flow matching network.

        Args:
            x (Tensor): The atom types of the noisy atoms. shape: [n_atoms, 1]
            pos_t (Tensor): The noisy positions at time t. shape: [n_atoms, 3]
            time_step (Tensor): The current time step. shape: [n_atoms], values in [0, 1]
            source_set_representation (Tensor): Learned representation of the source sets.
                shape: [n_atoms, n_embd]
            device (torch.device): cuda or cpu.

        Returns:
            Tensor: Vector field of shape [n_atoms, 3]
        """
        x = x.to(device)
        pos_t = pos_t.to(device)
        time_step = time_step.to(device)
        source_set_representation = source_set_representation.to(device)

        # CFM paths are conditioned on the type of the respective target atoms,
        # so we need to include this information in the flow matching condition.
        target_atom_type_embeddings = self.atom_type_embedding_layer(
            x
        )  # [n_target_atoms, n_embd]

        condition = (
            target_atom_type_embeddings + source_set_representation
        )  # [n_target_atoms, n_embd]

        output_fm = self.ada_mlp(
            pos_t, time_step, condition
        )  # [n_target_atoms, n_embd]

        return output_fm

    def configure_optimizers(self, betas=(0.9, 0.999)) -> Optimizer:
        """Same configurations as in NanoGPT"""
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == "cuda"
        # extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=betas,
            # fused=True,
        )

        def lr_lambda(epoch):
            # 1) linear warmup for warmup_iters steps
            warmup_epochs = self.hparams.lr_warmup_epochs
            min_lr = self.hparams.lr_min_ratio
            lr_decay_epochs = self.hparams.lr_decay_epochs
            if epoch < warmup_epochs:
                return (epoch + 1) / (warmup_epochs + 1)
            # 2) if it > lr_decay_iters, return min learning rate
            if epoch > lr_decay_epochs:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (epoch - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return min_lr + coeff * (1.0 - min_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int = None
    ) -> None:
        """Compute the gradient norm before clipping.

        Args:
            optimizer (Optimizer): The optimizer to use.
            optimizer_idx (int): The index of the optimizer.

        Returns:
            None
        """
        grad_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        self.log(
            "train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Shared step for training and validation"""
        loss, loss_ce, loss_fm, clash_penalty = self(batch)

        return loss, loss_ce, loss_fm, clash_penalty

    def on_train_start(self) -> None:
        """Initialization of the logger"""
        self.logger.log_hyperparams(
            self.hparams,
            {"train/train_loss": torch.inf, "val/val_loss": torch.inf},
        )

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
        loss, loss_ce, loss_fm, clash_penalty = self.shared_step(batch, batch_idx)

        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )
        self.log(
            "train/train_loss_ce",
            loss_ce,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )
        self.log(
            "train/train_loss_fm",
            loss_fm,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )
        if clash_penalty is not None:
            self.log(
                "train/clash_penalty",
                clash_penalty,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                batch_size=len(batch),
                reduce_fx="mean",
            )

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Validation step and logging"""
        loss, loss_ce, loss_fm, clash_penalty = self.shared_step(batch, batch_idx)

        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "val/val_loss_fm",
            loss_fm,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "val/val_loss_ce",
            loss_ce,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        if clash_penalty is not None:
            self.log(
                "val/clash_penalty",
                clash_penalty,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )

        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_atoms: int = 100,
        num_time_steps: int = 60,
        device: torch.device = torch.device("cuda"),
        prefix_x: Tensor = None,
        prefix_pos: Tensor = None,
        time_step_spacing: str = "linear",
        integration_method: str = "euler_maruyama",
        cfg_factor: float = 0.0,
        pocket_info: dict | None = None,
        fragment_info: dict | None = None,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a molecule using the flow matching network.

        Args:
            batch_size (int): Number of molecules to generate.
            max_atoms (int): Maximum number of atoms to generate.
            num_time_steps (int): Number of time steps to use for the flow matching.
            device (torch.device): Device to use for computations.
            prefix_x (Tensor): Optional prefix atom types to condition the generation on.
            prefix_pos (Tensor): Optional prefix positions to condition the generation on.
            time_step_spacing (str): Spacing of the time steps. Options: 'linear', 'logarithmic', 'quadratic'.
            integration_method (str): Integration method to use. Options: 'euler', 'euler_maruyama'.
            cfg_factor (float): Factor for conditional generation (CrossDocked only).
            pocket_info: For CrossDocked, dict with pocket_x, pocket_pos, pocket_residue_id,
                pocket_residue_type, pocket_batch. Ignored for QM9 and GEOM.
            fragment_info: For CrossDocked, dict with fragment_x, fragment_pos, fragment_batch.
        Returns:
            tuple[Tensor, Tensor, Tensor]: The atom types, their positions, and the batch indices of the generated molecules.
        """
        
        if prefix_x is not None and prefix_pos is not None:
            # (1) Initialize starting atom types, positions, and batch source with the provided prefix
            x = torch.cat([prefix_x for _ in range(batch_size)]).to(device)
            pos = torch.cat([prefix_pos for _ in range(batch_size)], dim=0).to(device)
            batch_source = torch.cat(
                [torch.ones_like(prefix_x) * i for i in range(batch_size)]
            ).to(device)
            rotation_augmentation = RandomRotationAugmentation()
            pos = rotation_augmentation.rotate_molecule_randomly(pos, batch_source)
        
        elif fragment_info is not None:
            # (1) Initialize starting atom types, positions, and batch source with the provided fragment
            x = fragment_info["fragment_x"].to(device)
            pos = fragment_info["fragment_pos"].to(device)
            batch_source = fragment_info["fragment_batch"].to(device)
            mean_pos = global_mean_pool(pos, batch_source)
            pos = pos - mean_pos[batch_source]
            if pocket_info is not None:
                pocket_info["pocket_pos"] = (
                    pocket_info["pocket_pos"]
                    - mean_pos[pocket_info["pocket_batch"]]
                )
    
        else:
            # (1) Initialize empty starting atom types, positions, and batch source (unconditional generation)
            x = torch.empty(0, dtype=torch.long, device=device)
            pos = torch.empty(0, 3, device=device)
            batch_source = torch.empty(0, device=device, dtype=torch.long)

        # (2) Create a mask for the stop tokens that will be used to track which molecules have a stop token
        stop_token_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

        # (3) Create a tensor of molecule indices that do not have a stop token
        active_mol_idx = torch.arange(batch_size, device=device)[~stop_token_mask]

        # (4) Iterate over the maximum number of atoms to generate
        with tqdm(range(max_atoms)) as pbar:
            for i in pbar:
                # (4.1) Compute source set representation
                active_batch_size = len(active_mol_idx)
                expanded_mask = torch.isin(batch_source, active_mol_idx)
                masked_x = x[expanded_mask]
                masked_pos = pos[expanded_mask]
                masked_batch_source = batch_source[expanded_mask]
                _, batch_source_remapped = torch.unique(
                    masked_batch_source.clone(), return_inverse=True
                )
                if pocket_info is not None:
                    expanded_mask_pocket = torch.isin(
                        pocket_info["pocket_batch"], active_mol_idx
                    )
                    masked_pocket_batch = pocket_info["pocket_batch"][
                        expanded_mask_pocket
                    ]
                    _, pocket_batch_remapped = torch.unique(
                        masked_pocket_batch.clone(), return_inverse=True
                    )
                    pocket_info_masked = {
                        "pocket_x": pocket_info["pocket_x"][expanded_mask_pocket],
                        "pocket_pos": pocket_info["pocket_pos"][expanded_mask_pocket],
                        "pocket_residue_id": pocket_info["pocket_residue_id"][
                            expanded_mask_pocket
                        ],
                        "pocket_residue_type": pocket_info["pocket_residue_type"][
                            expanded_mask_pocket
                        ],
                        "pocket_batch": pocket_batch_remapped,
                    }
                else:
                    pocket_info_masked = None

                source_set_representation = self.compute_source_set_representation(
                    masked_x,
                    masked_pos,
                    batch_source_remapped,
                    active_batch_size,
                    device,
                    pocket_info=pocket_info_masked,
                )  # [active_mol_count, n_embd]
                
                # (4.2) Compute logits
                logits = self.atom_type_prediction_head(
                    source_set_representation
                )  # [active_mol_count, vocab_size]

                # (4.3) If using pocket information, the source set representations are
                # conditioned on the pocket.
                if pocket_info is not None and cfg_factor > 0.0:
                    source_set_representation_unconditioned_for_cfg = (
                        self.compute_source_set_representation(
                            masked_x,
                            masked_pos,
                            batch_source_remapped,
                            active_batch_size,
                            device,
                        )
                    )  # [active_mol_count, n_embd]
                    
                    logits_unconditioned = self.atom_type_prediction_head(
                        source_set_representation_unconditioned_for_cfg
                    )  # [active_mol_count, vocab_size]

                    logits = (
                        1 + cfg_factor
                    ) * logits - cfg_factor * logits_unconditioned

                else:
                    source_set_representation_unconditioned_for_cfg = None

                
                logits = logits / temperature  # Apply temperature scaling to logits
                #temperature *= 0.9  # Decay temperature over time to encourage exploration early and exploitation later

                # (4.4) Compute probabilities
                probabilities = F.softmax(
                    logits, dim=-1
                )  # [active_mol_count, vocab_size]

                # (4.5) Sample next atom types from the resulting distribution
                x_next = torch.argmax(probabilities, dim=1)
                #x_next_0_mask = x_next == 0  # Used in hybrid sampling
                # x_next_1_mask = x_next == 1  # Used in hybrid sampling
                #x_next = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # Used in hybrid sampling
                #x_next[x_next_0_mask] = 0  # Used in hybrid sampling
                # x_next[x_next_1_mask] = 1  # Used in hybrid sampling

                # (4.6) Create a mask on the active molecules given the newly predicted atom types
                x_next_mask = x_next == 0  # [active_mol_count]

                pbar.set_postfix_str(
                    f"Generating atom {i + 2} for {(~x_next_mask).sum()} molecules."
                )
                pbar.refresh()

                # (4.7) Update the stop token mask with the newly predicted stop tokens
                stop_token_mask[active_mol_idx] += x_next_mask  # [batch_size]

                # (4.8) Count the number of stop tokens and break if all molecules
                # have a stop token also update the active molecule indices and count
                n_stop_tokens = stop_token_mask.sum()
                active_mol_idx = torch.arange(batch_size, device=device)[
                    ~stop_token_mask
                ]  # [active_mol_count] carefull, this might be shorter than before, if stop tokens were predicted!
                if n_stop_tokens == batch_size:
                    break

                # (4.9) Select only the source set representations for the active molecules
                x_next = x_next[~x_next_mask]
                source_set_representation = source_set_representation[~x_next_mask]
                if source_set_representation_unconditioned_for_cfg is not None:
                    source_set_representation_unconditioned_for_cfg = source_set_representation_unconditioned_for_cfg[~x_next_mask]

                # (4.10) Calculate the positions of the newly predicted atoms with flow matching
                pos_next = self.calculate_positions(
                    x_next,
                    source_set_representation,
                    num_time_steps,
                    device,
                    time_step_spacing,
                    integration_method,
                    cfg_factor,
                    source_set_representation_unconditioned_for_cfg,
                )

                # (4.11) Update the positions with the calculated new positions
                x, pos, batch_source = self.update_batch_with_new_atoms(
                    x,
                    pos,
                    batch_source,
                    x_next,
                    pos_next,
                    active_mol_idx,
                    batch_size,
                    device,
                )

                # (4.12) Recenter the ligand (and pocket) w.r.t. the COM
                mean_pos = global_mean_pool(pos, batch_source)
                pos = pos - mean_pos[batch_source]
                if pocket_info is not None:
                    pocket_info["pocket_pos"] = (
                        pocket_info["pocket_pos"]
                        - mean_pos[pocket_info["pocket_batch"]]
                    )
        if pocket_info is not None:
            pocket_center = global_mean_pool(
                pocket_info["pocket_pos"], pocket_info["pocket_batch"]
            )
            pos = pos - pocket_center[batch_source]

        return Batch(x=x, pos=pos, batch=batch_source)
    
    def update_batch_with_new_atoms(
        self,
        x: Tensor,
        pos: Tensor,
        batch_source: Tensor,
        x_next: Tensor,
        pos_next: Tensor,
        active_mol_idx: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        
        # 1. Prepare the new elements
        # Reshape x_next and batch assignments for the new atoms to ensure correct dimensions
        x_next_flat = x_next.view(-1)
        pos_next_flat = pos_next.view(-1, 3)
        new_batch_elements = active_mol_idx.view(-1)

        # 2. Compute sorting indices to group everything correctly by molecule ID
        # Combine the existing batch assignments with the new ones
        combined_batch = torch.cat([batch_source, new_batch_elements], dim=0)
        
        # argsort is highly optimized on GPU. It brings all elements of molecule 0 together, 
        # then molecule 1, etc., perfectly preserving or re-ordering them cleanly.
        sort_indices = torch.argsort(combined_batch, stable=True)

        # 3. Concatenate existing data with new data and apply the sorted order
        updated_x = torch.cat([x, x_next_flat], dim=0)[sort_indices]
        updated_pos = torch.cat([pos, pos_next_flat], dim=0)[sort_indices]
        updated_batch = combined_batch[sort_indices]

        return updated_x, updated_pos, updated_batch



    def calculate_positions(
        self,
        x_next: Tensor,
        source_set_representation: Tensor,
        num_time_steps: int,
        device: torch.device,
        time_step_spacing: str = "linear",
        integration_method: str = "euler_maruyama",
        cfg_factor: float = 1.5,
        source_set_representation_unconditioned_for_cfg: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the positions of the newly predicted atoms with flow matching.

        Args:
            x_next (Tensor): Atom types of the newly predicted atoms. shape: [n_atoms, 1]
            source_set_representation_unconditioned (Tensor): Representation of the unconditioned source sets. shape: [n_atoms, n_embd]
            source_set_representation_conditioned (Tensor): Representation of the conditioned source sets. shape: [n_atoms, n_embd]
            num_time_steps (int): Number of time steps to use for the flow matching.
            device (torch.device): Device to use for computations.
            time_step_spacing (str): Spacing of the time steps. Options: 'linear', 'logarithmic', 'quadratic'.
            integration_method (str): Integration method to use for the flow matching. Options: 'euler', 'euler_maruyama'.
            cfg_factor (float): Classifier-free guidance strength when apply_classifier_free_guidance is True.
            apply_classifier_free_guidance (bool): If False, no classifier-free guidance is applied (QM9 / GEOM).

        Returns:
            Tensor: The positions of the newly predicted atoms. shape: [n_atoms, 3]
        """
        # (1) Initialize next atoms' position with a random position
        pos_next = self.hparams.noise_std * torch.randn(
            x_next.shape[0], 3, device=device
        )

        # (2) Pick a time step spacing strategy
        if time_step_spacing == "linear":
            time_steps = torch.linspace(0, 1, num_time_steps, device=device)
        elif time_step_spacing == "logarithmic":
            time_steps = 1.0 - torch.logspace(
                -2, 0, num_time_steps + 1, device=device
            ).flip(0)
            time_steps = time_steps - torch.min(time_steps)
            time_steps = time_steps / torch.max(time_steps)
        elif time_step_spacing == "quadratic":
            dts = (
                torch.arange(
                    -num_time_steps // 2,
                    num_time_steps // 2 + 1,
                    1,
                    device=device,
                    dtype=torch.long,
                )
            ) ** 2 + num_time_steps * 2
            dts = dts.float() / dts.sum()
            time_steps = torch.cumsum(dts, dim=0)
            time_steps = torch.cat([torch.tensor([0], device=device), time_steps])
        else:
            raise ValueError(
                f"Invalid time_step_spacing: {time_step_spacing}. Must be 'linear', 'logarithmic', or 'quadratic'."
            )
        dts = time_steps[1:] - time_steps[:-1]

        # (3) Find position of the atoms through integration of the time trajectory
        for dt, time_step in zip(dts, time_steps[:-1]):
            # (3.1) Compute the velocity at the current time step
            time_step = time_step.expand(x_next.shape[0])
            velocity = self.compute_vector_field(
                x_next,
                pos_next,
                time_step,
                source_set_representation,
                device=device,
            )

            # (3.2) IF CFG: Compute the velocity of the unconditioned model and apply classifier-free guidance
            if source_set_representation_unconditioned_for_cfg is not None:
                velocity_unconditioned_for_cfg = self.compute_vector_field(
                    x_next,
                    pos_next,
                    time_step,
                    source_set_representation_unconditioned_for_cfg,
                    device=device,
                )
                velocity = (
                    1 + cfg_factor
                ) * velocity - cfg_factor * velocity_unconditioned_for_cfg

            # (3.3) Update the positions with the computed velocity using the specified integration method
            if integration_method == "euler":
                delta_pos = dt * velocity
            elif integration_method == "euler_maruyama":
                # Following: https://github.com/apple/ml-simplefold/blob/0f44c59b1664e58acf2c72145b3f88c9c16dd6c4/src/simplefold/model/torch/sampler.py
                delta_pos = self.compute_euler_maruyama_step(
                    pos_next, velocity, time_step[0], dt
                )
            else:
                raise ValueError(
                    f"Invalid integration_method: {integration_method}. Must be 'euler' or 'euler_maruyama'."
                )
            pos_next += delta_pos

        return pos_next

    def compute_euler_maruyama_step(
        self,
        pos_next: Tensor,
        velocity: Tensor,
        time_step: Tensor,
        dt: float,
        tau: float = 0.3,
    ) -> Tensor:
        """Compute a single Euler-Maruyama integration step.

        Args:
            pos_next (Tensor): Current positions. shape: [n_atoms, 3]
            velocity (Tensor): Velocity field at current positions. shape: [n_atoms, 3]
            time_step (Tensor): Current time step. shape: [n_atoms]
            dt (float): Time step size.
            tau (float): Noise scale parameter.

        Returns:
            Tensor: Position update. shape: [n_atoms, 3]
        """
        eps = torch.randn_like(pos_next)
        score = self.compute_score_from_velocity(velocity, pos_next, time_step)
        diff_coeff = self.diffusion_coefficient(time_step)
        drift = velocity + diff_coeff * score
        delta_pos = drift * dt + torch.sqrt(2.0 * diff_coeff * dt * tau) * eps
        return delta_pos

    def compute_score_from_velocity(
        self,
        v_t: Tensor,
        y_t: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute the score function from the velocity field.

        Args:
            v_t (Tensor): Velocity field at time t. shape: [n_atoms, 3]
            y_t (Tensor): Noisy positions at time t. shape: [n_atoms, 3]
            t (Tensor): Current time step. shape: [n_atoms]

        Returns:
            Tensor: Score function at time t. shape: [n_atoms, 3]
        """
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1 - t, -1
        mean = y_t
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * v_t - mean) / var
        return score

    def diffusion_coefficient(
        self,
        t: Tensor,
        epsilon: float = 1e-3,
        w_cutoff: float = 0.9,
    ) -> Tensor:
        """Compute the diffusion coefficient at time t.

        Args:
            t (Tensor): Current time step. shape: [n_atoms]
            epsilon (float): Small constant to avoid division by zero.
            w_cutoff (float): Cutoff value for the diffusion coefficient.

        Returns:
            Tensor: Diffusion coefficient at time t. shape: [n_atoms]
        """
        w = (1.0 - t) / (t + epsilon)
        if t >= w_cutoff:
            w = torch.zeros_like(t)
        return w
