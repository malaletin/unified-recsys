"""Бизнес-метрики (CTR, Time-to-Book, Conversion Rate) и бутстрэп-CI."""
from __future__ import annotations

import numpy as np

from . import config as C


def compute_metrics(sessions: list[dict]) -> dict:
    n = len(sessions)
    clicks = sum(s["clicked"] for s in sessions)
    books = sum(s["booked"] for s in sessions)
    ttb = [s["time_to_book"] for s in sessions if s["booked"]]
    return dict(n_sessions=n, ctr=clicks / n if n else 0.0,
                conversion_rate=books / n if n else 0.0,
                time_to_book=float(np.median(ttb)) if ttb else float("nan"),
                clicks=clicks, books=books)


def bootstrap_ci(sessions, metric, iters=C.SIM.bootstrap_iters,
                 confidence=C.SIM.confidence, seed=C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(sessions); vals = []
    for _ in range(iters):
        sample = [sessions[i] for i in rng.integers(0, n, size=n)]
        v = compute_metrics(sample)[metric]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    a = (1 - confidence) / 2
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)))


def compare(proto, base) -> dict:
    mp, mb = compute_metrics(proto), compute_metrics(base)
    d = lambda a, b: (a - b) / b * 100 if b else float("nan")
    return dict(prototype=mp, baseline=mb,
                ctr_uplift_pct=d(mp["ctr"], mb["ctr"]),
                ttb_reduction_pct=d(mp["time_to_book"], mb["time_to_book"]),
                cr_uplift_pct=d(mp["conversion_rate"], mb["conversion_rate"]),
                ctr_ci_prototype=bootstrap_ci(proto, "ctr"),
                ctr_ci_baseline=bootstrap_ci(base, "ctr"),
                ttb_ci_prototype=bootstrap_ci(proto, "time_to_book"),
                cr_ci_prototype=bootstrap_ci(proto, "conversion_rate"))
