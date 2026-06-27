"""
Learning-to-Rank на градиентном бустинге (LightGBM, LambdaMART).

Сильнейший практический подход для задачи Trivago: для каждого clickout
кандидаты (impressions) описываются признаками (recsys.models.features), а
модель обучается оптимизировать ранжирование (objective=lambdarank) с
группировкой по инстансам. На инференсе скорит impressions и сортирует.

Требует пакет lightgbm (CPU; быстро на многоядерном CPU).
"""
from __future__ import annotations

import numpy as np

from recsys.models.base import stable_order
from recsys.models.features import FEATURE_NAMES, FeatureBuilder

try:
    from lightgbm import LGBMRanker
    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False


class LightGBMRanker:
    name = "LightGBM-LTR"

    def __init__(self, params: dict | None = None, num_boost_round: int = 300):
        self.params = params or dict(
            objective="lambdarank", metric="ndcg", n_estimators=num_boost_round,
            learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        self.model = None
        self.fb: FeatureBuilder | None = None

    def fit(self, clickouts, metadata):
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("Не установлен lightgbm: pip install lightgbm")
        self.fb = FeatureBuilder(metadata).fit(clickouts)
        X, y, groups = self.fb.build_training(clickouts)
        self.model = LGBMRanker(**self.params)
        # без feature_name: на инференсе подаём numpy-массив, имена не нужны
        # (важности признаков сопоставляются с FEATURE_NAMES вручную ниже)
        self.model.fit(X, y, group=groups)
        return self

    def rank(self, inst):
        feats = self.fb.transform_instance(inst)
        scores = self.model.predict(feats)
        return stable_order(inst.impressions, scores)

    def feature_importance(self) -> dict:
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        return dict(sorted(zip(FEATURE_NAMES, imp.tolist()), key=lambda t: -t[1]))
