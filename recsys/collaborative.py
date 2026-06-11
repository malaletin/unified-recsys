"""
Коллаборативный контур (Collaborative Filtering) на основе TruncatedSVD.

Строит разреженную матрицу взаимодействий «пользователь–оффер» со
взвешиванием сигналов (impression/click/book) и затуханием по времени
(Time Decay), раскладывает её усечённым SVD и формирует скоринг как
косинусное сходство латентных факторов пользователя и офферов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .mlcore import TruncatedSVD, cosine_similarity
from .time_decay import decay_weight


class CollaborativeModel:
    def __init__(self, offer_ids: list[str], n_components: int = C.MODEL.svd_components):
        self.offer_ids = offer_ids
        self.offer_index = {o: i for i, o in enumerate(offer_ids)}
        self.user_index: dict[str, int] = {}
        self.svd = TruncatedSVD(n_components=n_components, random_state=C.RANDOM_STATE)
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeModel":
        users = sorted(interactions["user_id"].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        n_u, n_i = len(users), len(self.offer_ids)
        R = np.zeros((n_u, n_i), dtype=float)

        ev_w = interactions["event"].map(C.EVENT_WEIGHTS).to_numpy(dtype=float)
        t_w = decay_weight(interactions["timestamp"])
        weights = ev_w * t_w
        for uid, oid, w in zip(interactions["user_id"], interactions["offer_id"], weights):
            ui = self.user_index.get(uid); oi = self.offer_index.get(oid)
            if ui is not None and oi is not None:
                R[ui, oi] += w

        self.R = R
        self.svd.fit(R)
        self.user_factors = self.svd.user_factors(R)     # (n_u x k)
        self.item_factors = self.svd.item_factors()      # (n_i x k)
        return self

    def has_user(self, user_id: str) -> bool:
        return user_id in self.user_index and self.R[self.user_index[user_id]].sum() > 0

    def scores(self, user_id: str, offer_ids: list[str]) -> np.ndarray:
        """Нормализованный коллаборативный скоринг по списку офферов."""
        if not self.has_user(user_id):
            return np.zeros(len(offer_ids))
        uf = self.user_factors[self.user_index[user_id]].reshape(1, -1)
        idx = [self.offer_index[o] for o in offer_ids if o in self.offer_index]
        if not idx:
            return np.zeros(len(offer_ids))
        sims = cosine_similarity(uf, self.item_factors[idx]).ravel()
        out = np.zeros(len(offer_ids))
        j = 0
        for k, o in enumerate(offer_ids):
            if o in self.offer_index:
                out[k] = sims[j]; j += 1
        lo, hi = out.min(), out.max()
        return (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
