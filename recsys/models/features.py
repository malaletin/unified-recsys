"""
Инженерия признаков для Learning-to-Rank (LightGBM) на задаче Trivago.

Для каждого показанного отеля (impression) в момент clickout строится вектор
признаков, отражающих позицию, цену, популярность, сигналы текущей сессии и
контентное соответствие. Эти признаки — основа сильнейших решений челленджа.

FeatureBuilder обучается на train (популярность, пространство свойств) и
преобразует инстансы в матрицы признаков как для обучения LTR, так и для
инференса (ре-ранжирования).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from recsys.mlcore import cosine_similarity

FEATURE_NAMES = [
    "position",          # позиция показа (0 = первый)
    "price",             # цена за ночь
    "price_rank",        # ранг по цене внутри выдачи (0 = самый дешёвый)
    "price_rel_mean",    # цена / средняя цена в выдаче
    "price_is_min",      # 1, если самая дешёвая в выдаче
    "pop_log",           # log(1 + глобальная популярность кликов)
    "pop_impr_log",      # log(1 + сколько раз показывался)
    "sess_interactions", # число взаимодействий с отелем в текущей сессии
    "sess_seen",         # 1, если уже взаимодействовали в сессии
    "content_sim",       # косинус свойств отеля и профиля сессии
    "stars",             # звёздность (из свойств)
    "n_props",           # число свойств отеля
    "rating_prop",       # рейтинговое свойство (Satisfactory/Good/... -> числ.)
]


def _stars_from_props(props: set[str]) -> float:
    for s, v in (("5 Star", 5), ("4 Star", 4), ("3 Star", 3), ("2 Star", 2), ("1 Star", 1)):
        if s in props:
            return float(v)
    return 0.0


def _rating_from_props(props: set[str]) -> float:
    order = ["Satisfactory Rating", "Good Rating", "Very Good Rating", "Excellent Rating"]
    val = 0.0
    for i, r in enumerate(order, 1):
        if r in props:
            val = float(i)
    return val


class FeatureBuilder:
    def __init__(self, metadata):
        self.metadata = metadata
        self.pop_click: Counter = Counter()
        self.pop_impr: Counter = Counter()
        self._props_cache: dict[str, set] = {}

    def fit(self, clickouts) -> "FeatureBuilder":
        for c in clickouts:
            if c.target:
                self.pop_click[c.target] += 1
            for it in c.impressions:
                self.pop_impr[it] += 1
        return self

    # ---- свойства отеля (с кэшем) ---- #
    def _props(self, item_id: str) -> set[str]:
        p = self._props_cache.get(item_id)
        if p is None:
            idx = self.metadata._item_index.get(item_id)
            if idx is None:
                p = set()
            else:
                row = self.metadata.matrix[idx]
                p = {self.metadata.properties[j] for j in np.nonzero(row)[0]}
            self._props_cache[item_id] = p
        return p

    def _session_profile(self, prior_items):
        vecs = [self.metadata.vector(i) for i in prior_items if self.metadata.has(i)]
        if not vecs:
            return None
        return np.mean(np.vstack(vecs).astype(np.float32), axis=0)

    # ---- матрица признаков одного инстанса ---- #
    def transform_instance(self, inst) -> np.ndarray:
        imps = inst.impressions
        prices = np.array(inst.prices + [0.0] * (len(imps) - len(inst.prices)), dtype=np.float32)[:len(imps)]
        mean_price = prices[prices > 0].mean() if np.any(prices > 0) else 1.0
        price_order = np.argsort(np.argsort(prices))   # ранги
        min_price = prices[prices > 0].min() if np.any(prices > 0) else 0.0
        sess_cnt = Counter(inst.prior_items)
        profile = self._session_profile(inst.prior_items)

        rows = np.zeros((len(imps), len(FEATURE_NAMES)), dtype=np.float32)
        if profile is not None and profile.sum() > 0:
            imp_mat = np.vstack([self.metadata.vector(o).astype(np.float32) for o in imps])
            sims = cosine_similarity(profile.reshape(1, -1), imp_mat).ravel()
        else:
            sims = np.zeros(len(imps), dtype=np.float32)

        for j, it in enumerate(imps):
            props = self._props(it)
            rows[j] = [
                j,
                prices[j],
                price_order[j],
                prices[j] / mean_price if mean_price else 0.0,
                1.0 if (prices[j] > 0 and prices[j] == min_price) else 0.0,
                np.log1p(self.pop_click.get(it, 0)),
                np.log1p(self.pop_impr.get(it, 0)),
                sess_cnt.get(it, 0),
                1.0 if it in sess_cnt else 0.0,
                sims[j],
                _stars_from_props(props),
                float(len(props)),
                _rating_from_props(props),
            ]
        return rows

    # ---- сборка обучающей матрицы (X, y, groups) ---- #
    def build_training(self, clickouts):
        X_parts, y, groups = [], [], []
        for c in clickouts:
            if not c.has_target:
                continue
            feats = self.transform_instance(c)
            labels = [1 if o == c.target else 0 for o in c.impressions]
            X_parts.append(feats)
            y.extend(labels)
            groups.append(len(c.impressions))
        X = np.vstack(X_parts) if X_parts else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        return X, np.array(y, dtype=np.int32), np.array(groups, dtype=np.int32)
