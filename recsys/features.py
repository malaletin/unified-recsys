"""Инженерия признаков: контентное пространство офферов, векторы пользователя,
поведенческие агрегаты для кластеризации, контекстные признаки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .mlcore import StandardScaler


class OfferFeatureSpace:
    """Контентная матрица офферов (One-Hot + Multi-Label + StandardScaler)."""
    NUMERIC = ["price_per_night", "stars", "capacity", "rating", "poi_distance_km"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.offer_ids: list[str] = []
        self.columns: list[str] = []
        self.matrix: np.ndarray | None = None
        self.meta: pd.DataFrame | None = None

    def fit(self, offers: pd.DataFrame) -> "OfferFeatureSpace":
        self.offer_ids = offers["offer_id"].tolist()
        self.meta = offers.set_index("offer_id")
        feats, cols = self._build(offers)
        n = len(self.NUMERIC)
        feats[:, :n] = self.scaler.fit_transform(feats[:, :n])
        self.matrix, self.columns = feats, cols
        return self

    def _build(self, offers):
        cols = list(self.NUMERIC)
        blocks = [offers[self.NUMERIC].to_numpy(dtype=float)]
        for field, vocab in (("city", C.CITIES), ("meal_plan", C.MEAL_PLANS), ("room_type", C.ROOM_TYPES)):
            blocks.append(np.array([[1.0 if v == val else 0.0 for val in vocab] for v in offers[field]]))
            cols += [f"{field}={v}" for v in vocab]
        blocks.append(np.array([[1.0 if a in set(s.split("|")) else 0.0 for a in C.AMENITIES]
                                 for s in offers["amenities"]]))
        cols += [f"amen={a}" for a in C.AMENITIES]
        return np.hstack(blocks), cols

    def index(self, offer_id: str) -> int:
        return self.offer_ids.index(offer_id)

    def vector_for_offers(self, offer_ids, weights=None) -> np.ndarray:
        idx = [self.index(o) for o in offer_ids if o in self.offer_ids]
        if not idx:
            return np.zeros(self.matrix.shape[1])
        mat = self.matrix[idx]
        if weights is not None:
            w = np.asarray([weights[i] for i, o in enumerate(offer_ids) if o in self.offer_ids], dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(mat)) / len(mat)
            return (mat * w[:, None]).sum(axis=0)
        return mat.mean(axis=0)


BUDGET_TO_PRICE = {"low": 3500, "mid": 7000, "high": 14000}
GOAL_TO_AMENITIES = {"business": ["WiFi", "Parking", "Gym"], "family": ["Kitchen", "KidsZone", "Pool", "BBQ"],
                     "luxury": ["SPA", "Pool", "SeaView"], "solo": ["WiFi"],
                     "romantic": ["SPA", "SeaView", "Pool"], "active": ["Pool", "Gym", "BBQ", "Kitchen"]}
GOAL_TO_ROOM = {"business": "Standard", "family": "Apartment", "luxury": "Suite",
                "solo": "Standard", "romantic": "Suite", "active": "Apartment"}


def user_vector_from_questionnaire(space: OfferFeatureSpace, answers: dict) -> np.ndarray:
    price = BUDGET_TO_PRICE.get(answers.get("budget", "mid"), 7000)
    room = GOAL_TO_ROOM.get(answers.get("trip_goal", "solo"), "Standard")
    amen = set(GOAL_TO_AMENITIES.get(answers.get("trip_goal", "solo"), []))
    if answers.get("needs_kitchen"): amen.add("Kitchen")
    if answers.get("needs_parking"): amen.add("Parking")
    if answers.get("with_kids"): amen.add("KidsZone")
    stars = 5 if answers.get("budget") == "high" else 4 if answers.get("budget") == "mid" else 3
    numeric = np.array([price, stars, max(1, answers.get("party", 2)), 8.8, 2.0], dtype=float)
    numeric = space.scaler.transform(numeric.reshape(1, -1)).ravel()
    parts = [numeric, np.zeros(len(C.CITIES)),
             np.array([1.0 if m == "BB" else 0.0 for m in C.MEAL_PLANS]),
             np.array([1.0 if r == room else 0.0 for r in C.ROOM_TYPES]),
             np.array([1.0 if a in amen else 0.0 for a in C.AMENITIES])]
    return np.hstack(parts)


BEHAVIORAL_COLS = ["history_len", "avg_price", "avg_nights", "avg_amount", "avg_guests",
                   "business_share", "avg_stars", "avg_rating", "avg_poi",
                   "kitchen_share", "spa_share", "pool_share", "seaview_share", "parking_share"]


def build_behavioral_features(history: pd.DataFrame, offers: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    off = offers.set_index("offer_id")
    rows = []
    for _, u in users.iterrows():
        uid = u["user_id"]; h = history[history["user_id"] == uid]
        if len(h) == 0:
            rows.append(dict(user_id=uid, **{c: 0 for c in BEHAVIORAL_COLS})); continue
        meta = off.loc[h["offer_id"].tolist()]
        amen = [set(s.split("|")) for s in meta["amenities"]]
        share = lambda a: float(np.mean([a in s for s in amen]))
        rows.append(dict(user_id=uid, history_len=len(h),
                         avg_price=float(meta["price_per_night"].mean()),
                         avg_nights=float(h["nights"].mean()), avg_amount=float(h["amount"].mean()),
                         avg_guests=float(h["guests"].mean()),
                         business_share=float((h["nights"] <= 2.5).mean()),
                         avg_stars=float(meta["stars"].mean()), avg_rating=float(meta["rating"].mean()),
                         avg_poi=float(meta["poi_distance_km"].mean()),
                         kitchen_share=share("Kitchen"), spa_share=share("SPA"),
                         pool_share=share("Pool"), seaview_share=share("SeaView"),
                         parking_share=share("Parking")))
    return pd.DataFrame(rows)
