"""
Классические модели матричной факторизации (BPR-MF и ALS) через `implicit`.

Строится разреженная матрица «пользователь–отель» по взаимодействиям сессий;
модель обучает латентные факторы отелей. На инференсе профиль сессии (среднее
факторов ранее просмотренных отелей) скорит показанные варианты. Для отелей
без факторов (холодные) применяется откат к порядку показа.

На задаче Trivago (короткие сессии, разреженность) MF-бейзлайны, как правило,
уступают LTR и контентным моделям — это ожидаемый методологический результат.
Требует пакеты implicit и scipy (CPU).
"""
from __future__ import annotations

import numpy as np

from recsys.models.base import stable_order

try:
    import scipy.sparse as sp
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    IMPLICIT_AVAILABLE = True
except Exception:  # pragma: no cover
    IMPLICIT_AVAILABLE = False


class _MFBase:
    def __init__(self, factors=64, iterations=20, random_state=42):
        self.factors = factors
        self.iterations = iterations
        self.random_state = random_state
        self.model = None
        self.item_index: dict[str, int] = {}
        self.item_factors = None

    def _make_model(self):
        raise NotImplementedError

    def fit(self, clickouts, metadata=None):
        if not IMPLICIT_AVAILABLE:
            raise RuntimeError("Не установлены implicit/scipy: pip install implicit scipy")
        users, items = {}, {}
        rows, cols, vals = [], [], []
        for c in clickouts:
            uid = c.user_id
            inter = list(c.prior_items) + ([c.target] if c.target else [])
            if not inter:
                continue
            u = users.setdefault(uid, len(users))
            for it in inter:
                j = items.setdefault(it, len(items))
                rows.append(u); cols.append(j); vals.append(1.0)
        self.item_index = items
        mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)), dtype=np.float32)
        self.model = self._make_model()
        self.model.fit(mat, show_progress=False)
        self.item_factors = np.asarray(self.model.item_factors)
        return self

    def _profile(self, prior_items):
        idx = [self.item_index[i] for i in prior_items if i in self.item_index]
        if not idx:
            return None
        return self.item_factors[idx].mean(axis=0)

    def rank(self, inst):
        prof = self._profile(inst.prior_items)
        if prof is None:
            return list(inst.impressions)               # cold -> порядок показа
        scores = np.full(len(inst.impressions), -1e9, dtype=np.float32)
        for j, it in enumerate(inst.impressions):
            k = self.item_index.get(it)
            if k is not None:
                scores[j] = float(prof @ self.item_factors[k])
        return stable_order(inst.impressions, scores)


class ALSModel(_MFBase):
    name = "ALS"

    def _make_model(self):
        return AlternatingLeastSquares(factors=self.factors, iterations=self.iterations,
                                       random_state=self.random_state, use_gpu=False)


class BPRModel(_MFBase):
    name = "BPR-MF"

    def _make_model(self):
        return BayesianPersonalizedRanking(factors=self.factors, iterations=self.iterations,
                                           random_state=self.random_state, use_gpu=False)
