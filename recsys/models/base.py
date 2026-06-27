"""
Единый интерфейс модели ре-ранжирования для задачи Trivago.

Любая модель (baseline, LTR, MF, последовательная) реализует два метода:
    fit(train_clickouts, metadata) -> self
    rank(instance) -> list[item_id]   # порядок убывания предпочтения

Благодаря общему интерфейсу все модели прогоняются одним протоколом
оценки (recsys.evaluation.protocol) и сравниваются в единой таблице метрик.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RankingModel(Protocol):
    name: str

    def fit(self, clickouts, metadata): ...

    def rank(self, instance) -> list[str]: ...


def stable_order(impressions: list[str], scores) -> list[str]:
    """Сортировка по убыванию скора с детерминированным тай-брейком по позиции."""
    idx = sorted(range(len(impressions)), key=lambda j: (-scores[j], j))
    return [impressions[j] for j in idx]
