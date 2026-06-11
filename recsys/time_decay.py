"""Экспоненциальное затухание весов взаимодействий по времени (Time Decay).

Свежие взаимодействия важнее давних. Вес сигнала умножается на
exp(-ln2 · Δt / halflife), где halflife — период полураспада (дней).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

REF_DATE = np.datetime64("2026-05-01")


def decay_weight(timestamps, ref=REF_DATE, halflife_days: float = C.MODEL.time_decay_halflife_days):
    ts = pd.to_datetime(pd.Series(timestamps)).to_numpy()
    delta_days = (ref - ts) / np.timedelta64(1, "D")
    delta_days = np.clip(delta_days.astype(float), 0, None)
    return np.exp(-np.log(2) * delta_days / halflife_days)
