"""
Партиционирование данных на федеративных «клиентов».

В кросс-платформенной постановке каждый клиент — это отдельная площадка
(локаль). В датасете Trivago роль идентификатора площадки играет поле
platform. Клиенты обучают модель локально на своих сессиях, а сервер
агрегирует веса — сырые данные площадку не покидают, что соответствует
сценарию защиты персональных данных.

Здесь clickout-инстансы группируются по площадке, после чего на ОБЩЕМ
словаре отелей строятся локальные обучающие последовательности каждого
клиента.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from recsys.models.sequential.seqdata import Vocab, build_sequences


def platform_of(inst) -> str:
    """Идентификатор площадки (клиента) для инстанса."""
    p = getattr(inst, "platform", "") or "unknown"
    return str(p)


def partition_by_platform(clickouts, min_sessions: int = 50, max_clients: int | None = None):
    """
    Группирует инстансы по площадке. Мелкие площадки (< min_sessions) при
    желании можно отбросить, чтобы у каждого клиента было достаточно данных.
    Возвращает dict: platform -> список инстансов.
    """
    groups: dict[str, list] = defaultdict(list)
    for c in clickouts:
        groups[platform_of(c)].append(c)
    clients = {p: items for p, items in groups.items() if len(items) >= min_sessions}
    if max_clients is not None and len(clients) > max_clients:
        # оставляем крупнейшие площадки
        top = sorted(clients.items(), key=lambda kv: -len(kv[1]))[:max_clients]
        clients = dict(top)
    return clients


def build_client_datasets(clients: dict, vocab: Vocab, maxlen: int = 20):
    """
    Для каждого клиента строит (seqs, targets) на общем словаре отелей.
    Клиенты без пригодных последовательностей отбрасываются.
    """
    datasets = {}
    for platform, items in clients.items():
        seqs, targets = build_sequences(items, vocab, maxlen)
        if len(seqs) > 0:
            datasets[platform] = (seqs, targets)
    return datasets


def partition_summary(datasets: dict) -> dict:
    sizes = {p: int(len(s)) for p, (s, _) in datasets.items()}
    total = sum(sizes.values())
    return {"n_clients": len(sizes), "total_sequences": total,
            "per_client": dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
            "min": min(sizes.values()) if sizes else 0,
            "max": max(sizes.values()) if sizes else 0}
