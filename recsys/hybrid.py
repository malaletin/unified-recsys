"""
Расширенное гибридное ядро ранжирования.

Score = w_content·CB + w_collab·CF(SVD) + w_cluster·ClusterBoost + w_context·Context

  * CB           — контентное косинусное сходство (атрибутный матчинг);
  * CF(SVD)      — коллаборативный контур на латентных факторах с Time Decay;
  * ClusterBoost — историческая популярность оффера в кластере пользователя;
  * Context      — соответствие сезона/контекста датам запроса.

Реализованы: построчная нормализация, детерминированная обработка коллизий,
динамическое переключение режимов (Cold Start) — при недостатке истории
коллаборативный и кластерный контуры обнуляются, доминируют контент и контекст.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .clustering import UserSegmentation
from .collaborative import CollaborativeModel
from .content import ContentModel
from .context import context_scores
from .features import OfferFeatureSpace, build_behavioral_features


class HybridRecommender:
    def __init__(self, offers: pd.DataFrame, weights: dict | None = None,
                 cold_start_threshold: int = C.MODEL.cold_start_threshold):
        self.offers = offers
        self.offers_meta = offers.set_index("offer_id")
        self.space = OfferFeatureSpace().fit(offers)
        self.content = ContentModel(self.space)
        self.collab: CollaborativeModel | None = None
        self.segmentation: UserSegmentation | None = None
        self.cluster_aff: dict = {}
        self.threshold = cold_start_threshold
        self.w = weights or dict(content=C.MODEL.w_content, collab=C.MODEL.w_collab,
                                 cluster=C.MODEL.w_cluster, context=C.MODEL.w_context)
        self._hist_counts: dict[str, int] = {}
        self._history: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    def fit(self, history: pd.DataFrame, interactions: pd.DataFrame, users: pd.DataFrame):
        self._history = history
        self._hist_counts = history.groupby("user_id").size().to_dict()
        self.content.fit(history)
        self.collab = CollaborativeModel(self.space.offer_ids).fit(interactions)
        behavioral = build_behavioral_features(history, self.offers, users)
        self.segmentation = UserSegmentation().fit(behavioral)
        self.cluster_aff = self.segmentation.cluster_offer_affinity(history)
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def _minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)

    def _cluster_scores(self, cluster_id, offer_ids):
        if cluster_id is None or cluster_id not in self.cluster_aff:
            return np.zeros(len(offer_ids))
        aff = self.cluster_aff[cluster_id]
        return np.array([aff.get(o, 0.0) for o in offer_ids])

    def recommend(self, user_id: str | None, available_offers: list[str],
                  questionnaire: dict | None = None, check_in: str | None = None,
                  top_k: int = C.MODEL.top_k) -> list[dict]:
        if not available_offers:
            return []
        hist_len = self._hist_counts.get(user_id, 0) if user_id else 0
        cold = hist_len < self.threshold

        cb = self.content.scores(user_id, available_offers, questionnaire, cold)
        cf = (self.collab.scores(user_id, available_offers)
              if (not cold and user_id) else np.zeros(len(available_offers)))
        cid = self.segmentation.user_to_cluster.get(user_id) if (self.segmentation and not cold and user_id) else None
        clu = self._cluster_scores(cid, available_offers)
        ctx = context_scores(self.offers_meta, available_offers, check_in)

        w = self.w
        if cold:                       # Cold Start: коллаборатив и кластер отключены
            final = w["content"] * cb + w["context"] * ctx
        else:
            final = (w["content"] * cb + w["collab"] * cf +
                     w["cluster"] * clu + w["context"] * ctx)
        final = self._minmax(final)

        order = sorted(range(len(available_offers)), key=lambda i: (-final[i], available_offers[i]))
        out = []
        for rank, i in enumerate(order[:top_k], start=1):
            out.append(dict(
                offer_id=available_offers[i], rank=rank, score=round(float(final[i]), 4),
                score_content=round(float(cb[i]), 4), score_collab=round(float(cf[i]), 4),
                score_cluster=round(float(clu[i]), 4), score_context=round(float(ctx[i]), 4),
                cluster_label=self.segmentation.cluster_labels.get(cid) if (self.segmentation and cid is not None) else None,
                cold_start=cold))
        return out


class PopularityBaseline:
    def __init__(self, history: pd.DataFrame):
        self.pop = history["offer_id"].value_counts().to_dict()
        self._max = max(self.pop.values()) if self.pop else 1

    def recommend(self, user_id, available_offers, questionnaire=None, check_in=None, top_k=C.MODEL.top_k):
        order = sorted(available_offers, key=lambda o: (-self.pop.get(o, 0), o))
        return [dict(offer_id=o, rank=r + 1, score=round(self.pop.get(o, 0) / self._max, 4))
                for r, o in enumerate(order[:top_k])]
