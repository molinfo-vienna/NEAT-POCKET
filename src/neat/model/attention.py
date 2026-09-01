"""Taken and modified from the nanoGPT repository:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.attention.varlen import varlen_attn


def batch_to_ptr(batch: Tensor) -> Tensor:
    counts = torch.bincount(batch)
    cu_seqlens = torch.zeros(
    counts.size(0) + 1, dtype=torch.int32, device=batch.device
    )
    cu_seqlens[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    
    return cu_seqlens


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
    """Masked Bidirectional Self-Attention module using FlashAttention variable-length API.

    Args:
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        dropout (float): Dropout rate.
        bias (bool): Whether to include bias terms in linear layers.

    Returns:
        nn.Module: A Masked Bidirectional Attention module.
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

        # key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        x: Tensor of shape (B, T, C)
        attn_mask: Optional Boolean/Byte tensor of shape (B, T) where True indicates
          valid tokens and False indicates padding tokens.
        """
        BT, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=1)
        k = k.view(BT, self.n_head, self.head_dim)
        q = q.view(BT, self.n_head, self.head_dim)
        v = v.view(BT, self.n_head, self.head_dim)

        cu_seqlens = batch_to_ptr(batch)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    
        y = varlen_attn(
            q,
            k,
            v,
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
        )
        y = y.contiguous().view(BT, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MaskedCrossAttention(nn.Module):
    """Masked Cross-Attention module.

    Queries are projected from a primary stream ``x`` (ligand source
    tokens) and keys/values are projected from a context stream (pocket
    residue tokens). An optional attention mask supports key padding so that
    queries do not attend to padded context tokens.

    Args:
        n_embd (int): Embedding dimension shared between query and context.
        n_head (int): Number of attention heads.
        dropout (float): Dropout rate.
        bias (bool): Whether to include bias terms in linear layers.

    Returns:
        nn.Module: A Masked Cross-Attention module.
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
        query_input: torch.Tensor,
        kv_input: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        cross_attn_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        BT_q, C = query_input.size()
        BT_k, _ = kv_input.size()

        q = self.q_proj(query_input)
        k, v = self.kv_proj(kv_input).split(self.n_embd, dim=1)
        
        q = q.view(BT_q, self.n_head, self.head_dim)
        k = k.view(BT_k, self.n_head, self.head_dim)
        v = v.view(BT_k, self.n_head, self.head_dim)

        cu_seqlens_q = batch_to_ptr(batch)
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        cu_seqlens_k = batch_to_ptr(cross_attn_batch)
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

        y = varlen_attn(
            q,
            k,
            v,
            cu_seq_q=cu_seqlens_q,
            cu_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
        )
        y = y.contiguous().view(BT_q, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class BidirectionalAttentionBlock(nn.Module):
    """A transformer block with masked bidirectional attention.

    Args:
        n_embd (int): Number of embedding dimensions.
        n_head (int): Number of attention heads.
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in the layers.
        enable_cross_attention (bool): Whether to enable cross-attention. Default is False.
        scale_shift_weights (bool): Cross-attention uses scale and shift tensors for AdaLN. 
            If true, scale and shift weights are learned at each attention block. This 
            corresponds to the standard AdaLN implementation. If false, the AdaLN condition is 
            only modulated by a learned bias at attention block level (the non-linear 
            mapping should already happen before on a global level). This corresponds to
            the AdaLN-single implementation, that we use by default. Default is False.

    Returns:
        nn.Module: A transformer block module.
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
        # Preliminary: If using cross-attention, we use AdaLN
        if self.cross_attention:
            if self.scale_shift_weights:
                scale_shift = self.scale_shift(ada_ln_condition)
            else:
                scale_shift = (self.scale_shift + ada_ln_condition)
            alpha_1, beta_1, gamma_1, alpha_2, beta_2, alpha_3, beta_3, gamma_3 = (
                scale_shift.chunk(8, dim=-1)
            )

        # (1) Self-attention
        norm_x = self.ln_1(x)
        norm_x = norm_x * (1 + alpha_1[batch]) + beta_1[batch] if self.cross_attention else norm_x
        attention_residuals = self.attn(norm_x, batch=batch)
        attention_residuals = (
            attention_residuals * (1 + gamma_1[batch])
            if self.cross_attention
            else attention_residuals
        )
        x = x + attention_residuals

        # (2) (Optional) Cross-attention
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

        # (3) Feed-forward network
        norm_x = self.ln_3(x)
        norm_x = norm_x * (1 + alpha_3[batch]) + beta_3[batch] if self.cross_attention else norm_x
        ffn_projection = self.mlp(norm_x)
        ffn_projection = (
            ffn_projection * (1 + gamma_3[batch]) if self.cross_attention else ffn_projection
        )
        x = x + ffn_projection
        return x
