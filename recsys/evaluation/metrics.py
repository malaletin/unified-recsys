"""
Ранжирующие метрики офлайн-оценки.

Основная метрика челленджа — MRR. Дополнительно считаются HitRate@K,
NDCG@K, MAP@K, Recall@K (релевантным считается единственный кликнутый
отель), а также системные метрики Coverage, Diversity, Novelty.

Все функции принимают `ranked` — список item_id в порядке убывания
предпочтения, и `target` — истинно кликнутый item_id.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def reciprocal_rank(ranked: list[str], target: str) -> float:
    for i, item in enumerate(ranked, 1):
        if item == target:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    for i, item in enumerate(ranked[:k], 1):
        if item == target:
            return 1.0 / math.log2(i + 1)   # IDCG = 1 (единственный релевантный)
    return 0.0


def average_precision_at_k(ranked: list[str], target: str, k: int) -> float:
    for i, item in enumerate(ranked[:k], 1):
        if item == target:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


class MetricAccumulator:
    """Агрегирует метрики по множеству инстансов ранжирования."""

    def __init__(self, ks=(1, 5, 10, 25)):
        self.ks = ks
        self.n = 0
        self.mrr = 0.0
        self.hit = {k: 0.0 for k in ks}
        self.ndcg = {k: 0.0 for k in ks}
        self.ap = {k: 0.0 for k in ks}
        self.recommended_at1: Counter = Counter()   # для Coverage
        self.top1_props: list[set] = []              # для Diversity (опц.)
        self._catalog: set = set()

    def add(self, ranked: list[str], target: str, catalog: set | None = None):
        self.n += 1
        self.mrr += reciprocal_rank(ranked, target)
        for k in self.ks:
            self.hit[k] += hit_at_k(ranked, target, k)
            self.ndcg[k] += ndcg_at_k(ranked, target, k)
            self.ap[k] += average_precision_at_k(ranked, target, k)
        if ranked:
            self.recommended_at1[ranked[0]] += 1
        if catalog:
            self._catalog |= catalog

    def result(self) -> dict:
        if self.n == 0:
            return {}
        out = {"n": self.n, "MRR": self.mrr / self.n}
        for k in self.ks:
            out[f"HitRate@{k}"] = self.hit[k] / self.n
            out[f"NDCG@{k}"] = self.ndcg[k] / self.n
            out[f"MAP@{k}"] = self.ap[k] / self.n
        # Coverage@1 — доля каталога, попавшая в топ-1 хотя бы раз
        if self._catalog:
            out["Coverage@1"] = len(self.recommended_at1) / len(self._catalog)
        return out


def novelty_at_k(rankings: list[list[str]], popularity: dict[str, int],
                 total: int, k: int = 5) -> float:
    """Средняя новизна (self-information) топ-K: -log2(p(item))."""
    vals = []
    for ranked in rankings:
        for item in ranked[:k]:
            p = popularity.get(item, 1) / max(1, total)
            vals.append(-math.log2(max(p, 1e-12)))
    return float(np.mean(vals)) if vals else 0.0


def intra_list_diversity(rankings: list[list[str]], item_vectors, k: int = 5) -> float:
    """Среднее внутрисписочное разнообразие топ-K (1 - средн. косинус. сходство)."""
    from recsys.mlcore import cosine_similarity
    vals = []
    for ranked in rankings:
        top = [item_vectors(i) for i in ranked[:k] if item_vectors(i) is not None]
        if len(top) < 2:
            continue
        M = np.vstack(top)
        sim = cosine_similarity(M, M)
        n = len(top)
        off = (sim.sum() - np.trace(sim)) / (n * (n - 1))
        vals.append(1.0 - off)
    return float(np.mean(vals)) if vals else 0.0
