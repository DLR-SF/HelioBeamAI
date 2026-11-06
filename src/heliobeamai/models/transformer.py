"""Encoder-decoder transformer that maps calibration measurements to latents.

The encoder reads a set of context measurements, each a VAE latent of the
measured focal spot concatenated with its conditions (sun vector, aim
direction, normalized distance). The decoder queries the encoded context with
target conditions and returns one latent per query; the VAE decoder turns it
into a focal spot image.

Parameter names follow the original implementation, so checkpoints trained
with it load directly (see :mod:`heliobeamai.checkpoints`).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer norm with ``gamma``/``beta`` parameters and eps 1e-12."""

    def __init__(self, d_model: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class FeedForward(nn.Module):
    """Two linear layers with ReLU (and optional dropout) in between."""

    def __init__(self, d_in: int, hidden: int, d_out: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, hidden)
        self.linear2 = nn.Linear(hidden, d_out or d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} not divisible by n_head={n_head}")
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        return x.view(b, length, self.n_head, d // self.n_head).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self._split(self.w_q(q)), self._split(self.w_k(k)), self._split(self.w_v(v))
        score = q @ k.transpose(2, 3) / math.sqrt(k.size(-1))
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        out = torch.softmax(score, dim=-1) @ v
        b, h, length, d = out.shape
        return self.w_concat(out.transpose(1, 2).contiguous().view(b, length, h * d))


class EncoderLayer(nn.Module):
    """Post-norm self-attention block."""

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int,
                 dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, ffn_hidden, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.attention(x, x, x, mask)))
        return self.norm2(x + self.dropout2(self.ffn(x)))


class DecoderLayer(nn.Module):
    """Cross-attention block without self-attention.

    Target queries are independent of each other: every query only attends to
    the encoded context, so predictions do not depend on which other targets
    are in the batch.
    """

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int,
                 dropout: float) -> None:
        super().__init__()
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, ffn_hidden, dropout=dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm2(x + self.dropout2(self.enc_dec_attention(x, memory, memory, mask)))
        return self.norm3(x + self.dropout3(self.ffn(x)))


class Encoder(nn.Module):

    def __init__(self, d_in: int, d_model: int, ffn_hidden: int, n_head: int,
                 n_layers: int, dropout: float) -> None:
        super().__init__()
        self.emb = FeedForward(d_in, d_model, d_model)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layers))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):

    def __init__(self, d_in: int, d_model: int, ffn_hidden: int, n_head: int,
                 n_layers: int, dropout: float) -> None:
        super().__init__()
        self.emb = FeedForward(d_in, d_model // 2, d_model)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layers))

    def forward(self, trg: torch.Tensor, memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.emb(trg)
        for layer in self.layers:
            x = layer(x, memory, mask)
        return x


class FluxTransformer(nn.Module):
    """Context encoder + target decoder operating in VAE latent space.

    Args:
        latent_dim: size of one flattened VAE latent (64 for ``z64CH4``).
        cond_dim: number of condition values per measurement
            (sun vector 3, aim direction 3, normalized distance 1).
        d_model, ffn_hidden, n_head: attention dimensions.
        n_layers_enc, n_layers_dec: number of blocks.
        dropout: dropout in the encoder blocks.
        decoder_dropout: dropout in the decoder blocks (the original code fixed
            it to 0.1 regardless of ``dropout``).
    """

    def __init__(self, latent_dim: int = 64, cond_dim: int = 7, d_model: int = 64,
                 ffn_hidden: int = 64, n_head: int = 4, n_layers_enc: int = 2,
                 n_layers_dec: int = 4, dropout: float = 0.1,
                 decoder_dropout: float = 0.1) -> None:
        super().__init__()
        if d_model != latent_dim:
            raise ValueError("The decoder output is used as VAE latent directly, "
                             f"so d_model ({d_model}) must equal latent_dim ({latent_dim}).")
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.encoder =Encoder(latent_dim + cond_dim, d_model, ffn_hidden, n_head,
                               n_layers_enc, dropout)
        self.decoder = Decoder(cond_dim, d_model, ffn_hidden, n_head, n_layers_dec,
                               decoder_dropout)

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``src`` [B, S, latent+cond], ``trg`` [B, T, cond] -> latents [B, T, latent]."""
        return self.decoder(trg, self.encoder(src, src_mask), src_mask)
