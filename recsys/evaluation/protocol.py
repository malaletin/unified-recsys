"""
Протокол офлайн-оценки (temporal split).

Имитирует реальные условия эксплуатации: модель обучается на ранних
сессиях, оценивается на поздних. Утечка будущего исключена. Для каждого
тестового clickout модель переранжирует список показанных отелей
(impressions), после чего считаются ранжирующие метрики против истинно
кликнутого отеля.
"""
from __future__ import annotations

import numpy as np

from recsys.evaluation.metrics import MetricAccumulator, novelty_at_k


def temporal_split(clickouts, test_frac: float = 0.2):
    """Хронологическое разбиение инстансов на train/test по timestamp."""
    ordered = sorted(clickouts, key=lambda c: (c.timestamp, c.session_id, c.step))
    cut = int(len(ordered) * (1 - test_frac))
    return ordered[:cut], ordered[cut:]


def evaluate(model, train, test, metadata=None, ks=(1, 5, 10, 25),
             with_novelty: bool = True) -> dict:
    """Обучает модель на train и считает метрики на test."""
    model.fit(train, metadata)
    acc = MetricAccumulator(ks=ks)
    rankings = []
    catalog = set()
    for inst in test:
        if not inst.has_target:
            continue
        ranked = model.rank(inst)
        rankings.append(ranked)
        catalog |= set(inst.impressions)
        acc.add(ranked, inst.target, catalog=set(inst.impressions))
    res = acc.result()

    if with_novelty and rankings:
        from collections import Counter
        pop = Counter()
        for c in train:
            if c.target:
                pop[c.target] += 1
        res["Novelty@5"] = round(novelty_at_k(rankings, pop, sum(pop.values()) or 1, k=5), 3)
    return res


def evaluate_all(models, train, test, metadata=None, ks=(1, 5, 10, 25)) -> "list[dict]":
    """Оценивает список моделей и возвращает строки результатов."""
    rows = []
    for m in models:
        r = evaluate(m, train, test, metadata=metadata, ks=ks)
        r["model"] = getattr(m, "name", m.__class__.__name__)
        rows.append(r)
    rows.sort(key=lambda r: r.get("MRR", 0.0), reverse=True)
    return rows


def format_table(rows, cols=("model", "MRR", "HitRate@5", "NDCG@5", "MAP@5", "HitRate@25", "Novelty@5", "n")) -> str:
    """Аккуратная текстовая таблица результатов."""
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    widths = {c: max(len(c), *(len(fmt(r.get(c, "-"))) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append("  ".join(fmt(r.get(c, "-")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)
