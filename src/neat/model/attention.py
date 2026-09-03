"""Attention modules for NEAT, adapted from nanoGPT.

Tensor layout
-------------
Unlike padded transformers that use [batch_size, seq_len, n_embd], these
modules operate on a flattened token list (PyG style):

- Token features x have shape [num_tokens, n_embd].
- batch is a one-dimensional index vector: batch[i] is the
  attention-sequence id for token i. Tokens with the same id attend only
  to each other (no padding tensor needed).
- The grouping encoded by batch is caller-defined. For example, it may
  label molecules (ligand self-attention), residues (pocket atom attention),
  or pockets (pocket residue attention).

Variable-length Flash Attention (varlen_attn) receives projected
q, k, v of shape [num_tokens, n_head, head_dim] plus
cumulative sequence boundaries derived from batch.
"""

from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.attention.varlen import varlen_attn


def batch_to_ptr(batch: Tensor) -> Tensor:
    """Convert per-token sequence ids to cumulative sequence boundaries.

    Each unique value in batch defines one variable-length attention
    sequence.

    Args:
        batch: Per-token sequence ids, shape [num_tokens].

    Returns:
        Cumulative token offsets, shape [num_seqs + 1], dtype int32.
        cu_seqlens[s] and cu_seqlens[s + 1] are the start (inclusive)
        and end (exclusive) token indices for sequence s.

    Example:
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> batch_to_ptr(batch)
        tensor([0, 3, 5, 9], dtype=torch.int32)
    """
    counts = torch.bincount(batch)
    cu_seqlens = torch.zeros(
    counts.size(0) + 1, dtype=torch.int32, device=batch.device
    )
    cu_seqlens[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    
    return cu_seqlens


# The maximum number of sequences FlashAttention varlen can handle is 65535.
MAX_VARLEN_SEQS = 65535


def safe_varlen_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batch_q: Tensor,
    batch_k: Optional[Tensor] = None,
) -> Tensor:
    """Run variable-length attention, chunking if sequence count exceeds the Flash limit.

    Args:
        q: Query tensor, shape [num_tokens_q, n_head, head_dim].
        k: Key tensor, shape [num_tokens_k, n_head, head_dim].
        v: Value tensor, shape [num_tokens_k, n_head, head_dim].
        batch_q: Per-query-token sequence ids, shape [num_tokens_q].
        batch_k: Per-key/value-token sequence ids, shape [num_tokens_k].
            Defaults to batch_q (self-attention). For cross-attention,
            sequence s in queries attends to sequence s in keys/values.

    Returns:
        Attention output, shape [num_tokens_q, n_head, head_dim].
    """
    if batch_k is None:
        batch_k = batch_q

    cu_q = batch_to_ptr(batch_q)  # Cumulative sequence lengths for the query.
    cu_k = batch_to_ptr(batch_k) if batch_k is not batch_q else cu_q  # Cumulative sequence lengths for the key.
    num_seqs_q = cu_q.numel() - 1  # Number of sequences in the query.
    num_seqs_k = cu_k.numel() - 1  # Number of sequences in the key.
    max_seqlen_q = int((cu_q[1:] - cu_q[:-1]).max().item())  # Maximum sequence length in the query.
    max_seqlen_k = int((cu_k[1:] - cu_k[:-1]).max().item())  # Maximum sequence length in the key.

    if num_seqs_q <= MAX_VARLEN_SEQS and num_seqs_k <= MAX_VARLEN_SEQS:
        return varlen_attn(
            q,
            k,
            v,
            cu_seq_q=cu_q,
            cu_seq_k=cu_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
        )

    if batch_k is batch_q:
        # Self-attention: slice tokens and re-index sequence ids per chunk.
        outputs = []
        for seq_start in range(0, num_seqs_q, MAX_VARLEN_SEQS):
            seq_end = min(seq_start + MAX_VARLEN_SEQS, num_seqs_q)
            tok_start = int(cu_q[seq_start].item())
            tok_end = int(cu_q[seq_end].item())
            batch_chunk = batch_q[tok_start:tok_end] - seq_start
            cu_chunk = batch_to_ptr(batch_chunk)
            max_chunk = int((cu_chunk[1:] - cu_chunk[:-1]).max().item())
            outputs.append(
                varlen_attn(
                    q[tok_start:tok_end],
                    k[tok_start:tok_end],
                    v[tok_start:tok_end],
                    cu_seq_q=cu_chunk,
                    cu_seq_k=cu_chunk,
                    max_q=max_chunk,
                    max_k=max_chunk,
                )
            )
        return torch.cat(outputs, dim=0)

    # Cross-attention: slice query and key/value token ranges in parallel.
    outputs = []
    for seq_start in range(0, num_seqs_q, MAX_VARLEN_SEQS):
        seq_end = min(seq_start + MAX_VARLEN_SEQS, num_seqs_q)
        tok_q_start = int(cu_q[seq_start].item())
        tok_q_end = int(cu_q[seq_end].item())
        tok_k_start = int(cu_k[seq_start].item())
        tok_k_end = int(cu_k[seq_end].item())
        batch_q_chunk = batch_q[tok_q_start:tok_q_end] - seq_start
        batch_k_chunk = batch_k[tok_k_start:tok_k_end] - seq_start
        cu_q_chunk = batch_to_ptr(batch_q_chunk)
        cu_k_chunk = batch_to_ptr(batch_k_chunk)
        max_q_chunk = int((cu_q_chunk[1:] - cu_q_chunk[:-1]).max().item())
        max_k_chunk = int((cu_k_chunk[1:] - cu_k_chunk[:-1]).max().item())
        outputs.append(
            varlen_attn(
                q[tok_q_start:tok_q_end],
                k[tok_k_start:tok_k_end],
                v[tok_k_start:tok_k_end],
                cu_seq_q=cu_q_chunk,
                cu_seq_k=cu_k_chunk,
                max_q=max_q_chunk,
                max_k=max_k_chunk,
            )
        )
    return torch.cat(outputs, dim=0)


class MLP(nn.Module):
    """A simple feed-forward neural network (MLP) used in transformer blocks.

    Args:
        n_embd (int): Number of embedding dimensions.
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in the layers.

    Returns:
        nn.Module: An MLP module.
    """

    def __init__(
        self,
        n_embd: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MaskedBidirectionalAttention(nn.Module):
    """Bidirectional self-attention over flattened, variable-length token groups.

    Tokens that share the same batch id form one sequence and attend only
    within that group. No explicit attention-mask tensor is used; boundaries
    come from batch via varlen_attn.

    Args:
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        dropout: Dropout rate on the output projection.
        bias: Whether linear layers include bias.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply bidirectional self-attention.

        Args:
            x: Token embeddings, shape [num_tokens, n_embd].
            batch: Per-token sequence ids, shape [num_tokens].

        Returns:
            Updated token embeddings, shape [num_tokens, n_embd].
        """
        num_tokens, n_embd = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=1)
        k = k.view(num_tokens, self.n_head, self.head_dim)
        q = q.view(num_tokens, self.n_head, self.head_dim)
        v = v.view(num_tokens, self.n_head, self.head_dim)

        y = safe_varlen_attn(q, k, v, batch_q=batch)
        y = y.contiguous().view(num_tokens, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MaskedCrossAttention(nn.Module):
    """Cross-attention between a query stream and a separate key/value stream.

    Queries are projected from q_input (e.g. ligand atoms) and keys/values
    from kv_input (e.g. pocket residue embeddings). Sequence boundaries
    are set independently via batch and cross_attn_batch; aligned
    sequence indices are paired (query sequence s attends to key/value sequence s).

    Args:
        n_embd: Embedding dimension shared by both streams.
        n_head: Number of attention heads.
        dropout: Dropout rate on the output projection.
        bias: Whether linear layers include bias.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.kv_proj = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        cross_attn_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention from queries to keys/values.

        Args:
            q_input: Query token embeddings, shape [num_tokens_q, n_embd].
            kv_input: Key/value token embeddings, shape [num_tokens_k, n_embd].
            batch: Per-query-token sequence ids, shape [num_tokens_q].
            cross_attn_batch: Per-key/value-token sequence ids,
                shape [num_tokens_k].

        Returns:
            Updated query embeddings, shape [num_tokens_q, n_embd].
        """
        num_tokens_q, n_embd = q_input.size()
        num_tokens_k, _ = kv_input.size()

        q = self.q_proj(q_input)
        k, v = self.kv_proj(kv_input).split(self.n_embd, dim=1)

        q = q.view(num_tokens_q, self.n_head, self.head_dim)
        k = k.view(num_tokens_k, self.n_head, self.head_dim)
        v = v.view(num_tokens_k, self.n_head, self.head_dim)

        y = safe_varlen_attn(
            q, k, v, batch_q=batch, batch_k=cross_attn_batch
        )
        y = y.contiguous().view(num_tokens_q, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class BidirectionalAttentionBlock(nn.Module):
    """Transformer block: self-attention, optional cross-attention, then MLP.

    All sub-layers use the same flattened [num_tokens, n_embd] layout.
    The batch argument passed to self-attention defines token groupings;
    when cross-attention is enabled, cross_attn_batch defines groupings
    for the context stream independently.

    Args:
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        dropout: Dropout rate.
        bias: Whether linear layers include bias.
        enable_cross_attention: If True, add a cross-attention sub-layer and
            AdaLN conditioning from ada_ln_condition.
        scale_shift_weights: If True, learn AdaLN scale/shift from the
            condition vector (standard AdaLN). If False, use a learned bias
            added to the condition (AdaLN-single, the default in NEAT).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        enable_cross_attention: bool = False,
        scale_shift_weights: bool = False,
    ) -> None:
        super().__init__()
        self.scale_shift_weights = scale_shift_weights
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = MaskedBidirectionalAttention(
            n_embd, n_head, dropout, bias
        )
        if enable_cross_attention:
            self.ln_2 = nn.LayerNorm(n_embd, bias=False)
            self.attn_cross = MaskedCrossAttention(n_embd, n_head, dropout, bias)
            if scale_shift_weights:
                self.scale_shift = nn.Sequential(
                    nn.Linear(n_embd, 256, bias=False),
                    nn.SiLU(),
                    nn.Linear(256, 8 * n_embd, bias=False),
                )
            else:
                self.scale_shift = nn.Parameter(torch.zeros(8 * n_embd))
        self.ln_3 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(n_embd, dropout, bias=False)
        self.cross_attention = enable_cross_attention

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        cross_attn_input: Optional[torch.Tensor] = None,
        cross_attn_batch: Optional[torch.Tensor] = None,
        ada_ln_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the block on flattened tokens.

        Args:
            x: Token embeddings, shape [num_tokens, n_embd].
            batch: Per-token sequence ids for self-attention, shape
                [num_tokens].
            cross_attn_input: Context embeddings for cross-attention,
                shape [num_tokens_ctx, n_embd]. Required when cross-attention
                is enabled.
            cross_attn_batch: Per-context-token sequence ids, shape
                [num_tokens_ctx].
            ada_ln_condition: Global conditioning vector per graph,
                shape [num_graphs, n_embd] (or [num_graphs, 8 * n_embd]
                when scale_shift_weights is True). Used to modulate
                layer norms via AdaLN; indexed by batch.

        Returns:
            Updated token embeddings, shape [num_tokens, n_embd].
        """
        # AdaLN scale/shift parameters, one row per graph in the batch.
        if self.cross_attention:
            if self.scale_shift_weights:
                scale_shift = self.scale_shift(ada_ln_condition)
            else:
                scale_shift = (self.scale_shift + ada_ln_condition)
            alpha_1, beta_1, gamma_1, alpha_2, beta_2, alpha_3, beta_3, gamma_3 = (
                scale_shift.chunk(8, dim=-1)
            )

        # (1) Self-attention within each sequence group.
        norm_x = self.ln_1(x)
        norm_x = norm_x * (1 + alpha_1[batch]) + beta_1[batch] if self.cross_attention else norm_x
        attention_residuals = self.attn(norm_x, batch=batch)
        attention_residuals = (
            attention_residuals * (1 + gamma_1[batch])
            if self.cross_attention
            else attention_residuals
        )
        x = x + attention_residuals

        # (2) Optional cross-attention to a separate context stream.
        if (
            self.cross_attention
            and cross_attn_input is not None
            and cross_attn_batch is not None
        ):
            norm_x = self.ln_2(x)
            norm_x = norm_x * (1 + alpha_2[batch]) + beta_2[batch]
            cross_attn_residuals = self.attn_cross(
                norm_x, cross_attn_input, batch=batch, cross_attn_batch=cross_attn_batch
            )
            x = x + cross_attn_residuals

        # (3) Position-wise feed-forward network.
        norm_x = self.ln_3(x)
        norm_x = norm_x * (1 + alpha_3[batch]) + beta_3[batch] if self.cross_attention else norm_x
        ffn_projection = self.mlp(norm_x)
        ffn_projection = (
            ffn_projection * (1 + gamma_3[batch]) if self.cross_attention else ffn_projection
        )
        x = x + ffn_projection
        return x
