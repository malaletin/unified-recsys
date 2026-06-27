"""
Обёртка последовательной модели под единый интерфейс ре-ранжирования.

fit() обучает сеть (GRU4Rec или SASRec) на последовательностях сессий с
BPR-функцией потерь и negative sampling (масштабируется на большой каталог).
rank() кодирует предысторию сессии и скорит только показанные отели
(impressions), что делает инференс дешёвым независимо от размера каталога.

Требует пакет torch (GPU включается автоматически при наличии CUDA).
"""
from __future__ import annotations

import numpy as np

from recsys.models.base import stable_order
from recsys.models.sequential.seqdata import PAD, Vocab, build_sequences

try:
    import torch
    import torch.nn.functional as F
    from recsys.models.sequential.nets import build_net
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


class SequentialRanker:
    def __init__(self, kind: str = "sasrec", emb_dim: int = 64, maxlen: int = 20,
                 epochs: int = 10, batch_size: int = 512, lr: float = 1e-3,
                 n_neg: int = 4, weight_decay: float = 1e-6, device: str | None = None,
                 seed: int = 42):
        self.kind = kind
        self.name = {"sasrec": "SASRec", "gru": "GRU4Rec"}.get(kind, kind)
        self.emb_dim, self.maxlen = emb_dim, maxlen
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.n_neg, self.weight_decay, self.seed = n_neg, weight_decay, seed
        self.device = device
        self.vocab: Vocab | None = None
        self.net = None

    def fit(self, clickouts, metadata=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Не установлен torch: см. requirements-gpu.txt")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = Vocab().fit(clickouts)
        seqs, targets = build_sequences(clickouts, self.vocab, self.maxlen)
        if len(seqs) == 0:
            raise RuntimeError("нет обучающих последовательностей (пустые предыстории)")

        n_items = len(self.vocab)
        self.net = build_net(self.kind, n_items, self.emb_dim, self.maxlen).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        seqs_t = torch.as_tensor(seqs, device=self.device)
        tgt_t = torch.as_tensor(targets, device=self.device)
        n = len(seqs_t)
        self.net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                seq, pos = seqs_t[idx], tgt_t[idx]
                neg = torch.randint(1, n_items, (len(idx), self.n_neg), device=self.device)
                repr_vec = self.net.encode(seq)                       # (B, E)
                pos_emb = self.net.item_emb(pos)                      # (B, E)
                neg_emb = self.net.item_emb(neg)                      # (B, n_neg, E)
                pos_score = (repr_vec * pos_emb).sum(-1, keepdim=True)        # (B, 1)
                neg_score = (repr_vec.unsqueeze(1) * neg_emb).sum(-1)         # (B, n_neg)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss) * len(idx)
            print(f"  [{self.name}] epoch {epoch + 1}/{self.epochs}  loss={total / n:.4f}")
        self.net.eval()
        return self

    def rank(self, inst):
        if not inst.prior_items:
            return list(inst.impressions)                            # холодный старт
        with torch.no_grad():
            seq = torch.as_tensor([self.vocab.encode(inst.prior_items, self.maxlen)],
                                  device=self.device)
            repr_vec = self.net.encode(seq)                          # (1, E)
            ids, known = [], []
            for o in inst.impressions:
                j = self.vocab.item2id.get(o, PAD)
                ids.append(j); known.append(j != PAD)
            ids_t = torch.as_tensor([ids], device=self.device)
            scores = self.net.score_items(repr_vec, ids_t).squeeze(0).cpu().numpy()
        scores = np.where(known, scores, -1e9)                       # неизвестные отели вниз
        return stable_order(inst.impressions, scores)
