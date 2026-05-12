"""Taken and modified from the nanoGPT repository:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm

from ..dataset.augmentation import RandomRotationAugmentation
from .attention import BidirectionalAttentionBlock, CrossAttentionBlock
from .positional_encoding import (AxialRotaryPositionEncoding,
                                  FourierPositionEncoding)
from .simple_mlp import SimpleMLPAdaLN


class NEAT(LightningModule):
    """NEAT model for molecular generation using continuous flow matching."""

    def __init__(self, **params) -> None:
        super(NEAT, self).__init__()
        self.hparams.setdefault("noise_std", 1.0)
        self.save_hyperparameters()

        # Atom type embedding layer
        self.atom_type_embedding_layer = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, 
            embedding_dim=self.hparams.n_embd, 
        )
        
        # Pocket residue type embedding layer
        self.hparams.setdefault("pocket_residue_vocab_size", 21)
        self.pocket_residue_type_embedding_layer = nn.Embedding(
            num_embeddings=self.hparams.pocket_residue_vocab_size,
            embedding_dim=self.hparams.n_embd,
        )
        self.pocket_atom_to_residue_attention_layer = nn.Sequential(
            nn.Linear(self.hparams.n_embd, self.hparams.n_embd,),
            nn.GELU(),
            nn.Linear(self.hparams.n_embd, 1),
        )

        # Fourier features for embedding of Cartesian coordinates
        self.fourier_embedding_layer = FourierPositionEncoding(
            out_dim=self.hparams.n_embd
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.hparams.dropout)

        # Transformer blocks for the ligand stream
        self.hparams.setdefault("ligand_n_layer", 12)
        self.transformer_blocks = nn.ModuleList(
            [
                BidirectionalAttentionBlock(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.dropout,
                    self.hparams.bias,
                    (
                        AxialRotaryPositionEncoding(
                            embed_dim=self.hparams.n_embd,
                            num_heads=self.hparams.n_head,
                        )
                        if self.hparams.rope
                        else None
                    ),
                )
                for _ in range(self.hparams.ligand_n_layer)
            ]
        )

        # Transformer blocks for the pocket stream
        self.hparams.setdefault("pocket_n_layer", self.hparams.n_layer)
        self.pocket_transformer_blocks = nn.ModuleList(
            [
                BidirectionalAttentionBlock(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.dropout,
                    self.hparams.bias,
                    (
                        AxialRotaryPositionEncoding(
                            embed_dim=self.hparams.n_embd,
                            num_heads=self.hparams.n_head,
                        )
                        if self.hparams.rope
                        else None
                    ),
                )
                for _ in range(self.hparams.pocket_n_layer)
            ]
        )

        # Classifier-free guidance: during training, randomly drop pocket
        # conditioning so the model sees an explicit unconditional signal.
        self.cfg_null_pocket_token = nn.Parameter(
            torch.zeros(1, 1, self.hparams.n_embd)
        )
        nn.init.normal_(self.cfg_null_pocket_token, mean=0.0, std=0.02)

        # Cross-attention from ligand source tokens (queries) to pocket
        # residue tokens (keys/values). A value of 0 disables cross-attention
        # so the model falls back to the unconditioned baseline behavior.
        self.hparams.setdefault("cross_attn_n_layer", self.hparams.n_layer)
        self.cross_attention_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.dropout,
                    self.hparams.bias,
                )
                for _ in range(self.hparams.cross_attn_n_layer)
            ]
        )

        # Layer normalization after the transformer blocks
        self.layer_norm_after_transformer = nn.LayerNorm(
            self.hparams.n_embd, bias=False
        )
        self.layer_norm_after_pocket_transformer = nn.LayerNorm(
            self.hparams.n_embd, bias=False
        )
        self.layer_norm_after_cross_attention = nn.LayerNorm(
            self.hparams.n_embd, bias=False
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
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.hparams.n_layer)
                )

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

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the NEAT model.

        Args:
            data: Batch of data.

        Returns:
            Tuple of total loss, atom type prediction loss, and flow matching loss.
        """
        device = data.x.device

        # We split the molecular data into source and target atom sets.
        # The indexing tensors point to the same molecules as in the original batch.
        # The source set contains at least one atom, and at most all atoms.
        # If it contains all atoms, then the target set will be empty.
        # The stop tokens mask indicates which molecules have empty target sets.

        # (1) Compute the representation of the source atom sets with the transformer
        source_set_representation = self.compute_source_set_representation(
            data.x[data.source_ptr],
            data.pos[data.source_ptr],
            data.batch[data.source_ptr],
            device,
            pocket_x=getattr(data, "pocket_x", None),
            pocket_pos=getattr(data, "pocket_pos", None),
            pocket_residue_index=getattr(data, "pocket_residue_index", None),
            pocket_residue_type=getattr(data, "pocket_residue_type", None),
            pocket_batch=getattr(data, "pocket_pos_batch", None),
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
        loss_fm = self.compute_flow_matching_loss(
            data.x[data.target_ptr],
            data.pos[data.target_ptr],
            data.pos_random,
            data.batch[data.target_ptr],
            source_set_representation,
            device,
        )

        # (5) Add the two losses together
        # Note that these two objectives are disentangled and independent of each other.
        loss = loss_ce + loss_fm

        return loss, loss_ce, loss_fm

    def compute_source_set_representation(
        self,
        x_source: Tensor,
        pos_source: Tensor,
        batch_source: Tensor,
        device: torch.device,
        pocket_x: Tensor | None = None,
        pocket_pos: Tensor | None = None,
        pocket_residue_index: Tensor | None = None,
        pocket_residue_type: Tensor | None = None,
        pocket_batch: Tensor | None = None,
    ) -> Tensor:
        """Compute the representation of the source atom sets.

        Args:
            x_source (Tensor): The atom types of the source atoms. shape: [n_source_atoms]
            pos_source (Tensor): The positions of the source atoms. shape: [n_source_atoms, 3]
            batch_source (Tensor): The batch indices of the source atoms. shape: [n_source_atoms]
            device (torch.device): The device to use for computations.

        Returns:
            Tensor: The representation of the source atom sets. shape: [batch_size, n_embd]
        """
        x_source = x_source.to(device)
        pos_source = pos_source.to(device)
        batch_source = batch_source.to(device)
        batch_size = int(batch_source.max().item()) + 1 if batch_source.numel() > 0 else 0

        pocket_tokens, pocket_mask = self.encode_pocket(
            batch_size=batch_size,
            device=device,
            pocket_x=pocket_x,
            pocket_pos=pocket_pos,
            pocket_residue_index=pocket_residue_index,
            pocket_residue_type=pocket_residue_type,
            pocket_batch=pocket_batch,
        )

        pocket_tokens, pocket_mask = self.maybe_drop_pocket(
            pocket_tokens, pocket_mask
        )

        # (1) Compute atom counts of the source sets
        atom_count_source = torch.bincount(batch_source)

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
        # The attention mask is used in the transformer blocks and is the outer product of the atom mask.
        attn_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(
            2
        )  # [batch_size, max_atom_count, max_atom_count]
        attn_mask = attn_mask.unsqueeze(1).expand(
            -1, self.hparams.n_head, -1, -1
        )  # [batch_size, n_head, max_atom_count, max_atom_count]

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

        if self.hparams.rope is True:
            dim_pos = [len(atom_count_source), atom_count_source.max(), 3]
            positions = torch.zeros(
                dim_pos, device=device
            )  # [batch_size, max_atom_count, n_embd]
            positions[atom_mask] = pos_source  # [batch_size, max_atom_count, n_embd]
        else:
            positions = None

        # (7) Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(
                x, attn_mask=attn_mask, pos=positions
            )  # [batch_size, max_atom_count, n_embd]

        # (8) Apply the output layer normalization
        x = self.layer_norm_after_transformer(x)  # [batch_size, max_atom_count, n_embd]

        # (9) Inject pocket residue tokens into the ligand stream via
        # cross-attention. Cross-attention is skipped when no pocket tokens are
        # available (e.g. unconditional generation or missing pocket data) so
        # the model behaves like the unconditioned baseline.
        if (
            len(self.cross_attention_blocks) > 0
            and pocket_tokens.size(1) > 0
        ):
            cross_attn_mask = atom_mask.unsqueeze(2) & pocket_mask.unsqueeze(
                1
            )  # [batch_size, max_atom_count, max_residues]
            cross_attn_mask = cross_attn_mask.unsqueeze(1).expand(
                -1, self.hparams.n_head, -1, -1
            )  # [batch_size, n_head, max_atom_count, max_residues]
            for block in self.cross_attention_blocks:
                x = block(
                    x, pocket_tokens, attn_mask=cross_attn_mask
                )  # [batch_size, max_atom_count, n_embd]
            x = self.layer_norm_after_cross_attention(x)

        # (10) Apply the atom mask to the input embedding (not really needed, could be removed)
        x = x * atom_mask.unsqueeze(-1)  # [batch_size, max_atom_count, n_embd]

        # (11) Pool the atom embeddings into a molecule embedding
        source_set_representation = x.sum(dim=1)  # [batch_size, n_embd]

        return source_set_representation

    def encode_pocket(
        self,
        batch_size: int,
        device: torch.device,
        pocket_x: Tensor | None = None,
        pocket_pos: Tensor | None = None,
        pocket_residue_index: Tensor | None = None,
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
            or pocket_residue_index is None
            or pocket_residue_type is None
            or pocket_batch is None
            or pocket_x.numel() == 0
        ):
            return (
                torch.zeros(batch_size, 0, self.hparams.n_embd, device=device),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            )

        pocket_x = pocket_x.to(device)
        pocket_pos = pocket_pos.to(device)
        pocket_residue_index = pocket_residue_index.to(device).long()
        pocket_residue_type = pocket_residue_type.to(device).long()
        pocket_batch = pocket_batch.to(device).long()

        # 1. Atom-level embeddings
        atom_embedding = self.atom_type_embedding_layer(pocket_x)
        pos_embedding = self.fourier_embedding_layer(pocket_pos)
        residue_type_embedding = self.pocket_residue_type_embedding_layer(pocket_residue_type)
        atom_tokens = self.dropout_layer(atom_embedding + pos_embedding + residue_type_embedding)  # shape: [n_atoms, n_embd]
        # TODO: Not so sure about adding the residue_type_embedding. 
        # This is not what we originally wanted, but maybe it helps. 
        # Decide later.

        # 2. Residue-level embeddings
        # 2.1 Track which atom belongs to which residue.
        max_residue_index = int(pocket_residue_index.max().item()) + 1
        global_residue_index = pocket_batch * max_residue_index + pocket_residue_index
        unique_global_residue_index, atom_to_residue = torch.unique(
            global_residue_index, return_inverse=True, sorted=True
        )  # sorted list of unique residue indices in the batch, and a mapping from atom indices to residue indices
        residue_count = unique_global_residue_index.numel()  # Number of distinct residues in the batch
        if residue_count == 0:
            return (
                torch.zeros(batch_size, 0, self.hparams.n_embd, device=device),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            )

        # 2.2 Pool atom-level tokens into residue-level tokens via attention pooling.
        pooled_tokens = self.pool_atoms_to_residues_with_attention(
            atom_tokens=atom_tokens,
            atom_to_residue=atom_to_residue,
            residue_count=residue_count,
        )  # shape: [R_max, n_embd]

        # 2.3 Compute the number of residues per graph and the maximum number of residues per graph.
        residue_batch = torch.div(
            unique_global_residue_index, max_residue_index, rounding_mode="floor"
        )  # shape: [n_atoms] Recovers graph id for each residue
        residue_count_per_graph = torch.bincount(residue_batch, minlength=batch_size)
        max_residues = int(residue_count_per_graph.max().item())

        # 2.4 Pad the residue-level tokens to the maximum number of residues per graph.
        # and compute a boolean mask s.t. residues only attend within the same graph and not to padding residues.
        pocket_mask = (
            torch.arange(max_residues, device=device).unsqueeze(0)
            < residue_count_per_graph.unsqueeze(1)
        )
        pocket_tokens = torch.zeros(
            batch_size, max_residues, self.hparams.n_embd, device=device
        )
        pocket_tokens[pocket_mask] = pooled_tokens

        # 3. Pass through transformer blocks
        # 3.1 Compute a boolean attention mask 
        # s.t. residues only attend within the same graph and not to padding residues.
        pocket_attn_mask = pocket_mask.unsqueeze(1) * pocket_mask.unsqueeze(2)  # shape: [batch_size, R_max, R_max]
        pocket_attn_mask = pocket_attn_mask.unsqueeze(1).expand(
            -1, self.hparams.n_head, -1, -1
        )  # shape: [batch_size, n_head, R_max, R_max]
        # 3.2 Pass through transformer blocks.
        for block in self.pocket_transformer_blocks:
            pocket_tokens = block(pocket_tokens, attn_mask=pocket_attn_mask, pos=None)
        # 3.3 Apply the output layer normalization
        pocket_tokens = self.layer_norm_after_pocket_transformer(pocket_tokens)
        # 3.4 Apply the atom mask to the input embedding (not really needed, could be removed)
        pocket_tokens = pocket_tokens * pocket_mask.unsqueeze(-1)

        return pocket_tokens, pocket_mask

    def maybe_drop_pocket(
        self,
        pocket_tokens: Tensor,
        pocket_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Training-time CFG: randomly replace pocket tokens with a learnable null.

        Uses per-graph Bernoulli sampling so a fraction of the batch trains as
        unconditional while the rest stays pocket-conditioned.
        """
        p = float(self.hparams.cfg_train_drop_prob)

        # 0. Early exit if the module is in eval mode or if the drop probability is 0.
        if (
            not self.training
            or p <= 0.0
        ):
            return pocket_tokens, pocket_mask

        # 1. Initialize drop mask with per-graph Bernoulli sampling.
        # Any 'True' means: for this graph, pretend we do not have pocket data.
        batch_size = pocket_tokens.size(0)
        device = pocket_tokens.device
        drop = (torch.rand(batch_size, device=device) < p)

        # 2. Replace pocket tokens with a learnable null for the dropped graphs.
        pocket_tokens = pocket_tokens.clone()
        pocket_mask = pocket_mask.clone()
        # 2.1 Create a learnable null residue token.
        null = self.cfg_null_pocket_token.to(
            device=device, dtype=pocket_tokens.dtype
        ).view(1, -1)
        # 2.2 Clear the tensor's entire entries for the dropped graphs.
        pocket_tokens[drop] = 0
        pocket_mask[drop] = False
        # 2.3 Install exactly one learnable null residue at slot 0 and mark it as valid according in the mask.
        pocket_tokens[drop, 0, :] = null
        pocket_mask[drop, 0] = True

        return pocket_tokens, pocket_mask

    def pool_atoms_to_residues_with_attention(
        self,
        atom_tokens: Tensor,
        atom_to_residue: Tensor,
        residue_count: int,
    ) -> Tensor:
        """Pool atom-level tokens into residue-level tokens using attention."""
        pooled_tokens = torch.zeros(
            residue_count,
            self.hparams.n_embd,
            device=atom_tokens.device,
            dtype=atom_tokens.dtype,
        )
        attention_logits = self.pocket_atom_to_residue_attention_layer(atom_tokens).squeeze(-1)

        for residue_idx in range(residue_count):
            residue_atom_mask = atom_to_residue == residue_idx
            residue_atom_tokens = atom_tokens[residue_atom_mask]
            residue_attention = F.softmax(
                attention_logits[residue_atom_mask], dim=0
            ).unsqueeze(-1)
            pooled_tokens[residue_idx] = torch.sum(
                residue_attention * residue_atom_tokens, dim=0
            )

        return pooled_tokens

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
        device: torch.device,
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

        # (9) Return the mean loss over all paths and time steps.
        return loss_fm.mean()  # [1]

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
        loss, loss_ce, loss_fm = self(batch)

        return loss, loss_ce, loss_fm

    def on_train_start(self) -> None:
        """Initialization of the logger"""
        self.logger.log_hyperparams(
            self.hparams,
            {"train/train_loss": torch.inf, "val/val_loss": torch.inf},
        )

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
        loss, loss_ce, loss_fm = self.shared_step(batch, batch_idx)

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

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Validation step and logging"""
        loss, loss_ce, loss_fm = self.shared_step(batch, batch_idx)

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

        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_atoms: int = 200,
        num_time_steps: int = 60,
        device: torch.device = torch.device("cuda"),
        prefix_x: Tensor = None,
        prefix_pos: Tensor = None,
        time_step_spacing: str = "linear",
        integration_method: str = "euler_maruyama",
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

        Returns:
            tuple[Tensor, Tensor, Tensor]: The atom types, their positions, and the batch indices of the generated molecules.
        """
        if prefix_x is not None and prefix_pos is not None:
            # (1) Initialize starting atom types with the provided prefix
            x = torch.cat([prefix_x for _ in range(batch_size)]).to(device)
            # (2) Initialize starting positions with the provided prefix
            pos = torch.cat([prefix_pos for _ in range(batch_size)], dim=0).to(device)
            # (3) Initialize the batch source tensor with the provided prefix
            batch_source = torch.cat(
                [torch.ones_like(prefix_x) * i for i in range(batch_size)]
            ).to(device)

            rotation_augmentation = RandomRotationAugmentation()
            pos = rotation_augmentation.rotate_molecule_randomly(pos, batch_source)
            # trans = torch.randn(batch_size, 3, device=device)
            # pos += trans[batch_source]
        else:
            # (1) Sample initial atoms from the prior distribution of atom types in QM9
            if self.hparams.data_set == "QM9":
                dist = torch.tensor(
                    [0.0000, 0.5109, 0.3517, 0.0580, 0.0780, 0.0014], device=device
                )
                x = torch.multinomial(
                    dist, batch_size, replacement=True
                )  # [batch_size]
            elif self.hparams.data_set == "GEOM":
                dist = torch.tensor(
                    [
                        0,
                        4.4115e-01,
                        1.0262e-06,
                        4.0569e-01,
                        6.4707e-02,
                        6.6119e-02,
                        4.8757e-03,
                        0,
                        7.2215e-07,
                        9.3404e-05,
                        1.2265e-02,
                        4.0290e-03,
                        0,
                        1.0497e-03,
                        1.9821e-05,
                        0,
                        7.6015e-08,
                    ],
                    device=device,
                )
                x = torch.multinomial(
                    dist, batch_size, replacement=True
                )  # [batch_size]
            else:
                raise ValueError(f"Unknown data set: {self.hparams.data_set}")
            # (2) Initialize starting positions with random ones
            pos = torch.zeros(batch_size, 3, device=device)
            # pos = self.hparams.noise_std * torch.randn(batch_size, 3, device=device)
            # (3) Initialize the batch source tensor
            batch_source = torch.arange(batch_size, device=device)
        # (4) Create a mask for the stop tokens that will be used to track which molecules have a stop token
        stop_token_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        # (5) Create a tensor of molecule indices that do not have a stop token
        active_mol_idx = torch.arange(batch_size, device=device)[~stop_token_mask]

        # (6) Iterate over the maximum number of atoms to generate
        with tqdm(range(max_atoms)) as pbar:
            for i in pbar:
                # (6.1) Compute source set representation
                expanded_mask = torch.isin(batch_source, active_mol_idx)
                masked_x = x[expanded_mask]
                masked_pos = pos[expanded_mask]
                masked_batch_source = batch_source[expanded_mask]
                _, batch_source_remapped = torch.unique(
                    masked_batch_source.clone(), return_inverse=True
                )
                source_set_representation = self.compute_source_set_representation(
                    masked_x, masked_pos, batch_source_remapped, device
                )  # [active_mol_count, n_embd]

                # (6.2) Compute logits
                logits = self.atom_type_prediction_head(
                    source_set_representation
                )  # [active_mol_count, vocab_size]

                # (6.3) Compute probabilities
                probabilities = F.softmax(
                    logits, dim=-1
                )  # [active_mol_count, vocab_size]

                # (6.4) Sample next atom types from the resulting distribution
                x_next = torch.argmax(probabilities, dim=1)
                x_next_0_mask = x_next == 0
                x_next_1_mask = x_next == 1
                x_next = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                x_next[x_next_0_mask] = 0
                x_next[x_next_1_mask] = 1

                # (6.5) Create a mask on the active molecules given the newly predicted atom types
                x_next_mask = x_next == 0  # [active_mol_count]

                pbar.set_postfix_str(
                    f"Generating atom {i + 2} for {(~x_next_mask).sum()} molecules."
                )
                pbar.refresh()

                # (6.6) Update the stop token mask with the newly predicted stop tokens
                stop_token_mask[active_mol_idx] += x_next_mask  # [batch_size]

                # (6.7) Count the number of stop tokens and break if all molecules
                # have a stop token also update the active molecule indices and count
                n_stop_tokens = stop_token_mask.sum()
                active_mol_idx = torch.arange(batch_size, device=device)[
                    ~stop_token_mask
                ]  # [active_mol_count] carefull, this might be shorter than before, if stop tokens were predicted!
                if n_stop_tokens == batch_size:
                    break

                # (6.8) Select only the source set representations for the active molecules
                x_next = x_next[~x_next_mask]
                source_set_representation = source_set_representation[~x_next_mask]
                # (6.9) Calculate the positions of the newly predicted atoms with flow matching
                pos_next = self.calculate_positions(
                    x_next,
                    source_set_representation,
                    num_time_steps,
                    device,
                    time_step_spacing,
                    integration_method,
                )

                # (6.10) Update the x, pos, and batch source tensors
                updated_x = []
                updated_pos = []
                updated_batch = []
                for idx in range(batch_size):
                    if idx in active_mol_idx:
                        active_idx = torch.where(active_mol_idx == idx)[0]
                        updated_x.append(
                            torch.cat(
                                (x[batch_source == idx], x_next[active_idx].view(1)),
                                dim=0,
                            )
                        )  # [num_atoms+1]
                        updated_pos.append(
                            torch.cat(
                                (
                                    pos[batch_source == idx],
                                    pos_next[active_idx].view(1, 3),
                                ),
                                dim=0,
                            )
                        )  # [num_atoms+1, 3]
                        updated_batch.append(
                            torch.cat(
                                (
                                    batch_source[batch_source == idx],
                                    torch.tensor(idx, device=device).view(1),
                                ),
                                dim=0,
                            )
                        )  # [num_atoms+1]
                    else:
                        updated_x.append(x[batch_source == idx])
                        updated_pos.append(pos[batch_source == idx])
                        updated_batch.append(batch_source[batch_source == idx])

                x = torch.cat(updated_x, dim=0)  # [batch_size]
                pos = torch.cat(updated_pos, dim=0)  # [batch_size, 3]
                batch_source = torch.cat(updated_batch, dim=0)  # [batch_size]
                mean_pos = global_mean_pool(pos, batch_source)
                pos = pos - mean_pos[batch_source]

        return Batch(x=x, pos=pos, batch=batch_source)

    def calculate_positions(
        self,
        x_next: Tensor,
        source_set_representation: Tensor,
        num_time_steps: int,
        device: torch.device,
        time_step_spacing: str = "linear",
        integration_method: str = "euler_maruyama",
    ) -> Tensor:
        """Calculate the positions of the newly predicted atoms with flow matching.

        Args:
            x_next (Tensor): Atom types of the newly predicted atoms. shape: [n_atoms, 1]
            source_set_representation (Tensor):Representation of the source sets. shape: [n_atoms, n_embd]
            num_time_steps (int): Number of time steps to use for the flow matching.
            device (torch.device): Device to use for computations.
            time_step_spacing (str): Spacing of the time steps. Options: 'linear', 'logarithmic', 'quadratic'.
            integration_method (str): Integration method to use for the flow matching. Options: 'euler', 'euler_maruyama'.

        Returns:
            Tensor: The positions of the newly predicted atoms. shape: [n_atoms, 3]
        """
        # (1) Initialize next atoms' position with a random position
        pos_next = self.hparams.noise_std * torch.randn(
            x_next.shape[0], 3, device=device
        )

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

        # (2) Find position of the atoms through integration of the time trajectory
        if integration_method == "euler":
            for dt, time_step in zip(dts, time_steps[:-1]):
                # for time_step in torch.linspace(0, 1, num_time_steps, device=device)[:-1]:
                time_step = time_step.expand(x_next.shape[0])
                velocity = self.compute_vector_field(
                    x_next,
                    pos_next,
                    time_step,
                    source_set_representation,
                    device=device,
                )
                delta_pos = dt * velocity
                pos_next += delta_pos
        elif integration_method == "euler_maruyama":
            # Following: https://github.com/apple/ml-simplefold/blob/0f44c59b1664e58acf2c72145b3f88c9c16dd6c4/src/simplefold/model/torch/sampler.py
            for dt, time_step in zip(dts, time_steps[:-1]):
                velocity = self.compute_vector_field(
                    x_next,
                    pos_next,
                    time_step.expand(x_next.shape[0]),
                    source_set_representation,
                    device=device,
                )
                delta_pos = self.compute_euler_maruyama_step(
                    pos_next, velocity, time_step, dt
                )
                pos_next += delta_pos
        else:
            raise ValueError(
                f"Invalid integration_method: {integration_method}. Must be 'euler' or 'euler_maruyama'."
            )

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
