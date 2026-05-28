"""Taken and modified from the nanoGPT repository:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from typing import Optional

import torch
import torch.nn as nn


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
    """Masked Bidirectional Self-Attention module.

    Args:
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        dropout (float): Dropout rate.
        bias (bool): Whether to include bias terms in linear layers.
        pos_embedder (nn.Module, optional): Positional embedding module. Default is None.

    Returns:
        nn.Module: A Masked Bidirectional Attention module.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        pos_embedder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.pos_embedder = pos_embedder

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        # Apply scaled dot-product attention with the provided attention mask
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # Pass the attention mask here
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,  # Bidirectional attention, not causal
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class BidirectionalAttentionBlock(nn.Module):
    """A transformer block with masked bidirectional attention.

    Args:
        n_embd (int): Number of embedding dimensions.
        n_head (int): Number of attention heads.
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in the layers.
        pos_embedder (Optional[nn.Module]): Positional embedder to use. Relates to rope embeddings.

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
        pos_embedder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = MaskedBidirectionalAttention(
            n_embd, n_head, dropout, bias, pos_embedder
        )
        if enable_cross_attention:
            self.ln_2 = nn.LayerNorm(n_embd, bias=False)
            self.attn_cross = MaskedCrossAttention(n_embd, n_head, dropout, bias)
            self.scale_shift = nn.Linear(n_embd, 8 * n_embd, bias=False)
        self.ln_3 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(n_embd, dropout, bias=False)
        self.cross_attention = enable_cross_attention

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cross_attn_input: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        ada_ln_condition: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Preliminary: If using cross-attention, we use AdaLN
        if self.cross_attention:
            scale_shift = self.scale_shift(ada_ln_condition).unsqueeze(1)
            alpha_1, beta_1, gamma_1, alpha_2, beta_2, alpha_3, beta_3, gamma_3 = (
                scale_shift.chunk(8, dim=-1)
            )

        # (1) Self-attention
        norm_x = self.ln_1(x)
        norm_x = norm_x * (1 + alpha_1) + beta_1 if self.cross_attention else norm_x
        attention_residuals = self.attn(norm_x, attn_mask=attn_mask, pos=pos)
        attention_residuals = (
            attention_residuals * (1 + gamma_1)
            if self.cross_attention
            else attention_residuals
        )
        x = x + attention_residuals

        # (2) (Optional) Cross-attention
        if (
            self.cross_attention
            and cross_attn_input is not None
            and cross_attn_mask is not None
        ):
            norm_x = self.ln_2(x)
            norm_x = norm_x * (1 + alpha_2) + beta_2
            cross_attn_residuals = self.attn_cross(
                norm_x, cross_attn_input, attn_mask=cross_attn_mask
            )
            x = x + cross_attn_residuals

        # (3) Feed-forward network
        norm_x = self.ln_3(x)
        norm_x = norm_x * (1 + alpha_3) + beta_3 if self.cross_attention else norm_x
        ffn_projection = self.mlp(norm_x)
        ffn_projection = (
            ffn_projection * (1 + gamma_3) if self.cross_attention else ffn_projection
        )
        x = x + ffn_projection
        return x


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
        self.dropout = dropout

    def forward(
        self,
        query_input: torch.Tensor,
        kv_input: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_q, C = query_input.size()
        _, T_k, _ = kv_input.size()

        q = self.q_proj(query_input)
        k, v = self.kv_proj(kv_input).split(self.n_embd, dim=2)

        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_q, hs)
        k = k.view(B, T_k, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_k, hs)
        v = v.view(B, T_k, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_k, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
