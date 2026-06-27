"""
Подготовка данных для последовательных моделей.

Из clickout-инстансов строится словарь отелей и обучающие пары
(последовательность взаимодействий в сессии -> кликнутый отель). Холодные
инстансы без предыстории в обучении последовательной модели не участвуют
(на инференсе для них применяется откат к порядку показа).
"""
from __future__ import annotations

import numpy as np

PAD = 0


class Vocab:
    """Словарь отелей: item_id -> целочисленный индекс (0 зарезервирован под PAD)."""

    def __init__(self):
        self.item2id: dict[str, int] = {}
        self.id2item: list[str] = ["<pad>"]

    def fit(self, clickouts) -> "Vocab":
        for c in clickouts:
            for it in list(c.prior_items) + ([c.target] if c.target else []):
                if it not in self.item2id:
                    self.item2id[it] = len(self.id2item)
                    self.id2item.append(it)
        return self

    def __len__(self):
        return len(self.id2item)

    def encode(self, items, maxlen):
        ids = [self.item2id[i] for i in items if i in self.item2id]
        ids = ids[-maxlen:]
        if len(ids) < maxlen:
            ids = [PAD] * (maxlen - len(ids)) + ids   # левый паддинг
        return ids


def build_sequences(clickouts, vocab: Vocab, maxlen: int = 20):
    """Возвращает (seqs [N x maxlen], targets [N]) для инстансов с предысторией."""
    seqs, targets = [], []
    for c in clickouts:
        if not c.target or c.target not in vocab.item2id:
            continue
        if not c.prior_items:
            continue
        seqs.append(vocab.encode(c.prior_items, maxlen))
        targets.append(vocab.item2id[c.target])
    if not seqs:
        return np.zeros((0, maxlen), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(seqs, dtype=np.int64), np.asarray(targets, dtype=np.int64)
