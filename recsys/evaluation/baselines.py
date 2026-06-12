"""
Линейка моделей ранжирования для задачи Trivago (переранжирование impressions).

Каждая модель реализует интерфейс:
    fit(train_clickouts, metadata) -> self
    rank(instance) -> list[item_id]   # порядок убывания предпочтения

Базовые линии:
  * ImpressionOrder  — порядок, в котором площадка показала отели (сильный baseline);
  * Popularity       — по глобальной популярности кликов в train;
  * PriceAsc         — по возрастанию цены;
  * ItemKNN          — session-based item-item по со-встречаемости в сессиях;
  * ContentSim       — по сходству свойств отеля с профилем сессии (item_metadata).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from recsys.mlcore import cosine_similarity


class ImpressionOrder:
    """Контрольная линия: оставить порядок показа без изменений."""
    name = "ImpressionOrder"

    def fit(self, clickouts, metadata):
        return self

    def rank(self, inst):
        return list(inst.impressions)


class Popularity:
    """Сортировка показанных отелей по глобальной популярности кликов."""
    name = "Popularity"

    def __init__(self):
        self.pop: Counter = Counter()

    def fit(self, clickouts, metadata):
        for c in clickouts:
            if c.target:
                self.pop[c.target] += 1
        return self

    def rank(self, inst):
        return sorted(inst.impressions, key=lambda i: (-self.pop.get(i, 0), inst.impressions.index(i)))


class PriceAsc:
    """Сортировка по возрастанию цены (учёт ценовой чувствительности)."""
    name = "PriceAsc"

    def fit(self, clickouts, metadata):
        return self

    def rank(self, inst):
        order = sorted(range(len(inst.impressions)),
                       key=lambda j: (inst.prices[j] if j < len(inst.prices) else 1e12, j))
        return [inst.impressions[j] for j in order]


class ItemKNN:
    """Session-based item-item: со-встречаемость отелей в рамках сессий."""
    name = "ItemKNN"

    def __init__(self, normalize: bool = True):
        self.cooc: dict[str, Counter] = defaultdict(Counter)
        self.freq: Counter = Counter()
        self.normalize = normalize

    def fit(self, clickouts, metadata):
        for c in clickouts:
            items = list(dict.fromkeys(c.prior_items + ([c.target] if c.target else [])))
            for a in items:
                self.freq[a] += 1
                for b in items:
                    if a != b:
                        self.cooc[a][b] += 1
        return self

    def _score(self, candidate, context):
        s = 0.0
        for ctx in context:
            c = self.cooc.get(ctx)
            if c and candidate in c:
                s += c[candidate] / (self.freq[candidate] if self.normalize and self.freq[candidate] else 1)
        return s

    def rank(self, inst):
        ctx = inst.prior_items
        if not ctx:
            # без контекста откатываемся к порядку показа
            return list(inst.impressions)
        scored = [(i, self._score(i, ctx)) for i in inst.impressions]
        return [i for i, _ in sorted(scored, key=lambda t: (-t[1], inst.impressions.index(t[0])))]


class ContentSim:
    """Контентное ранжирование: сходство свойств отеля с профилем сессии."""
    name = "ContentSim"

    def __init__(self):
        self.metadata = None

    def fit(self, clickouts, metadata):
        self.metadata = metadata
        return self

    def _profile(self, prior_items):
        vecs = [self.metadata.vector(i) for i in prior_items if self.metadata.has(i)]
        if not vecs:
            return None
        return np.mean(np.vstack(vecs), axis=0)

    def rank(self, inst):
        prof = self._profile(inst.prior_items)
        if prof is None or prof.sum() == 0:
            return list(inst.impressions)
        mat = np.vstack([self.metadata.vector(i) for i in inst.impressions])
        sims = cosine_similarity(prof.reshape(1, -1), mat).ravel()
        order = sorted(range(len(inst.impressions)), key=lambda j: (-sims[j], j))
        return [inst.impressions[j] for j in order]


def all_baselines():
    return [ImpressionOrder(), Popularity(), PriceAsc(), ItemKNN(), ContentSim()]
