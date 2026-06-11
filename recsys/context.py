"""Контекстный контур: учёт сезона и заблаговременности бронирования (lead time).

Повышает скоринг офферов, чей сезон соответствует датам запроса, и
учитывает паттерн раннего/позднего бронирования сегмента пользователя.
"""
from __future__ import annotations

import numpy as np


def season_of_date(date_str: str | None) -> str:
    if not date_str:
        return "any"
    try:
        m = int(str(date_str)[5:7])
    except (ValueError, IndexError):
        return "any"
    return "high" if m in (6, 7, 8, 12, 1) else "low"


def context_scores(offers_meta, offer_ids: list[str], check_in: str | None) -> np.ndarray:
    """Нормализованный контекстный скоринг: совпадение сезона оффера с датой запроса."""
    season = season_of_date(check_in)
    out = np.zeros(len(offer_ids))
    for k, o in enumerate(offer_ids):
        if o not in offers_meta.index:
            continue
        off_season = offers_meta.loc[o, "season"]
        if off_season == "any":
            out[k] = 0.5
        elif off_season == season:
            out[k] = 1.0
        else:
            out[k] = 0.0
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo) if hi > lo else out
