"""
Нейросетевые архитектуры последовательных рекомендаций: GRU4Rec и SASRec.

Обе модели кодируют последовательность взаимодействий сессии в вектор
представления, а скоринг отеля вычисляется скалярным произведением этого
вектора и эмбеддинга отеля (веса эмбеддингов разделяются между входом и
выходом — tied weights). Это позволяет на инференсе дёшево скорить лишь
показанные варианты, не вычисляя softmax по всему каталогу.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from recsys.models.sequential.seqdata import PAD


class _SeqEncoderBase(nn.Module):
    def __init__(self, n_items: int, emb_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD)
        self.emb_dim = emb_dim
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()

    def encode(self, seq):
        raise NotImplementedError

    def score_items(self, repr_vec, item_ids):
        """Скор отелей по индексам: <repr, item_emb>."""
        emb = self.item_emb(item_ids)                       # (..., emb_dim)
        return (repr_vec.unsqueeze(1) * emb).sum(-1) if emb.dim() == 3 else repr_vec @ emb.t()

    def full_scores(self, repr_vec):
        return repr_vec @ self.item_emb.weight.t()


class GRU4Rec(_SeqEncoderBase):
    def __init__(self, n_items: int, emb_dim: int = 64, hidden: int = 128,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__(n_items, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, emb_dim)
        self.drop = nn.Dropout(dropout)

    def encode(self, seq):
        x = self.drop(self.item_emb(seq))                   # (B, L, E)
        out, h = self.gru(x)
        return self.proj(out[:, -1, :])                     # представление последней позиции


class SASRec(_SeqEncoderBase):
    def __init__(self, n_items: int, emb_dim: int = 64, maxlen: int = 20,
                 num_heads: int = 2, num_blocks: int = 2, dropout: float = 0.2):
        super().__init__(n_items, emb_dim)
        self.pos_emb = nn.Embedding(maxlen, emb_dim)
        self.maxlen = maxlen
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                                       dim_feedforward=emb_dim * 4, dropout=dropout,
                                       batch_first=True, activation="gelu")
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def encode(self, seq):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        pad_mask = seq == PAD                                # (B, L) True там, где паддинг
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=causal, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return x[:, -1, :]                                  # представление последней позиции


def build_net(kind: str, n_items: int, emb_dim: int, maxlen: int):
    if kind == "gru":
        return GRU4Rec(n_items, emb_dim=emb_dim)
    if kind == "sasrec":
        return SASRec(n_items, emb_dim=emb_dim, maxlen=maxlen)
    raise ValueError(f"неизвестная архитектура: {kind}")
