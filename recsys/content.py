"""
Контентный контур (Content-Based Filtering).

Формирует вектор предпочтений пользователя как агрегат контентных векторов
ранее забронированных офферов (для активных пользователей) либо как маппинг
ответов анкеты (для холодного старта), и оценивает офферы косинусным сходством.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OfferFeatureSpace, user_vector_from_questionnaire
from .mlcore import cosine_similarity


class ContentModel:
    def __init__(self, space: OfferFeatureSpace):
        self.space = space
        self._user_offers: dict[str, list[str]] = {}

    def fit(self, history: pd.DataFrame) -> "ContentModel":
        self._user_offers = (history.groupby("user_id")["offer_id"]
                             .apply(list).to_dict())
        return self

    @staticmethod
    def _minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)

    def _user_vector(self, user_id, questionnaire, cold):
        if not cold and user_id in self._user_offers:
            return self.space.vector_for_offers(self._user_offers[user_id])
        if questionnaire:
            return user_vector_from_questionnaire(self.space, questionnaire)
        return self.space.matrix.mean(axis=0)

    def scores(self, user_id, offer_ids, questionnaire=None, cold=False) -> np.ndarray:
        vec = self._user_vector(user_id, questionnaire, cold).reshape(1, -1)
        idx = [self.space.index(o) for o in offer_ids]
        sims = cosine_similarity(vec, self.space.matrix[idx]).ravel()
        return self._minmax(sims)
