"""
Генератор синтетического кросс-платформенного датасета (домашняя среда).

Возвращает pandas-таблицы доменных сущностей. Используется как для
наполнения БД (seed), так и для имитационной валидации движка.

Поведение пользователей детерминированно зависит от скрытого
бизнес-архетипа (ground truth), что движок НЕ видит — он восстанавливает
предпочтения из истории/анкеты.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

REF_DATE = np.datetime64("2026-05-01")

ARCHETYPES = {
    "Business_Express": dict(nights=(1, 2), cities=["Москва", "Санкт-Петербург"],
                             amenities=["WiFi", "Parking", "Gym"], meal="BB",
                             room=["Standard", "Suite"], price=(6000, 12000), stars=(3, 5), lead=(1, 4)),
    "Family_Vacation": dict(nights=(7, 14), cities=["Сочи", "Калининград", "Казань"],
                            amenities=["Kitchen", "KidsZone", "Pool", "BBQ"], meal="RO",
                            room=["Apartment"], price=(4000, 9000), stars=(3, 4), lead=(30, 120)),
    "Luxury_Retreat": dict(nights=(3, 6), cities=["Сочи", "Санкт-Петербург"],
                           amenities=["SPA", "Pool", "SeaView"], meal="HB",
                           room=["Suite"], price=(12000, 22000), stars=(5, 5), lead=(14, 60)),
    "Solo_Budget": dict(nights=(1, 3), cities=C.CITIES,
                        amenities=["WiFi"], meal="RO",
                        room=["Standard"], price=(2000, 5000), stars=(3, 4), lead=(1, 14)),
    "Couple_Romantic": dict(nights=(2, 4), cities=["Сочи", "Калининград"],
                            amenities=["SPA", "SeaView", "Pool"], meal="HB",
                            room=["Suite"], price=(8000, 15000), stars=(4, 5), lead=(7, 45)),
    "Group_Active": dict(nights=(2, 5), cities=["Казань", "Калининград"],
                         amenities=["Pool", "Gym", "BBQ", "Kitchen"], meal="BB",
                         room=["Apartment"], price=(5000, 10000), stars=(3, 4), lead=(7, 30)),
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())

PLATFORMS = [
    {"platform_id": "P1", "name": "Островок", "kind": "OTA"},
    {"platform_id": "P2", "name": "Яндекс Путешествия", "kind": "ecosystem"},
    {"platform_id": "P3", "name": "Твил", "kind": "south_niche"},
    {"platform_id": "P4", "name": "Броневик", "kind": "business"},
]


def _rng(seed=C.RANDOM_STATE):
    return np.random.default_rng(seed)


def _season_of(ts: np.datetime64) -> str:
    m = int(str(ts)[5:7])
    return "high" if m in (6, 7, 8, 12, 1) else "low"


def gen_platforms():
    return pd.DataFrame(PLATFORMS)


def gen_hotels(rng):
    chains = ["Azimut", "Cosmos", "Independent"]
    rows = []
    for i in range(C.DATA.n_hotels):
        city = C.CITIES[i % len(C.CITIES)]
        rows.append(dict(hotel_id=f"H{i+1:02d}", name=f"Hotel_{i+1:02d}", city=city,
                         stars=int(rng.integers(3, 6)), chain=chains[int(rng.integers(0, 3))],
                         rating=round(float(rng.uniform(7.5, 9.6)), 1),
                         poi_distance_km=round(float(rng.uniform(0.2, 8.0)), 1)))
    return pd.DataFrame(rows)


def gen_offers(rng, hotels):
    rows = []
    for i in range(C.DATA.n_offers):
        h = hotels.iloc[int(rng.integers(0, len(hotels)))]
        room = C.ROOM_TYPES[int(rng.integers(0, len(C.ROOM_TYPES)))]
        base = 1500 + h["stars"] * 1800 + (4000 if room == "Suite" else 2500 if room == "Apartment" else 0)
        price = max(2000, int(base + rng.normal(0, 1200)))
        amenities = list(rng.choice(C.AMENITIES, size=int(rng.integers(2, 6)), replace=False))
        rows.append(dict(offer_id=f"O{i+1:03d}", hotel_id=h["hotel_id"], city=h["city"],
                         stars=int(h["stars"]), rating=float(h["rating"]),
                         poi_distance_km=float(h["poi_distance_km"]), room_type=room,
                         price_per_night=price, meal_plan=C.MEAL_PLANS[int(rng.integers(0, 3))],
                         capacity=int(rng.integers(1, 7)), min_nights=int(rng.choice([1, 1, 2, 3])),
                         season=str(rng.choice(C.SEASONS)), amenities="|".join(amenities)))
    return pd.DataFrame(rows)


def _affinity(arch, offer):
    s = 0.0
    if offer["city"] in arch["cities"]: s += 1.5
    if offer["room_type"] in arch["room"]: s += 1.2
    if offer["meal_plan"] == arch["meal"]: s += 0.5
    s += 0.8 * len(set(offer["amenities"].split("|")) & set(arch["amenities"]))
    lo, hi = arch["price"]
    s += 1.0 if lo <= offer["price_per_night"] <= hi else (0.3 if offer["price_per_night"] < lo else 0.0)
    if arch["stars"][0] <= offer["stars"] <= arch["stars"][1]: s += 0.6
    s += (offer["rating"] - 7.5) * 0.25
    return s


def affinity_matrix(offers):
    mat = {}
    for name, arch in ARCHETYPES.items():
        raw = np.clip(np.array([_affinity(arch, offers.iloc[i]) for i in range(len(offers))]), 0.01, None)
        mat[name] = np.exp(raw) / np.exp(raw).sum()
    return mat


def gen_users(rng):
    regions = ["Москва", "Санкт-Петербург", "ПФО", "ЮФО", "СЗФО", "УФО"]
    rows = []
    for i in range(C.DATA.n_users):
        arch = ARCHETYPE_NAMES[int(rng.integers(0, len(ARCHETYPE_NAMES)))]
        revoked = rng.random() < C.DATA.consent_revoked_share
        rows.append(dict(user_id=f"U{i+1:04d}", archetype_truth=arch,
                         region=regions[int(rng.integers(0, len(regions)))],
                         loyalty=str(rng.choice(["new", "silver", "gold"], p=[0.5, 0.35, 0.15])),
                         consent_status="revoked" if revoked else "granted"))
    return pd.DataFrame(rows)


def gen_history_and_interactions(rng, users, offers):
    aff = affinity_matrix(offers)
    offer_ids = offers["offer_id"].to_numpy()
    history, interactions = [], []

    hist_len = 1 + (rng.pareto(1.5, size=len(users)) * 2).astype(int)
    hist_len = np.clip(hist_len, 0, 40)
    hist_len = (hist_len * (C.DATA.n_history / max(1, hist_len.sum()))).round().astype(int)

    hid = 0
    for ui, user in users.iterrows():
        arch = ARCHETYPES[user["archetype_truth"]]
        probs = aff[user["archetype_truth"]]
        for _ in range(int(hist_len[ui])):
            oi = int(rng.choice(len(offer_ids), p=probs))
            offer = offers.iloc[oi]
            nights = int(rng.integers(arch["nights"][0], arch["nights"][1] + 1))
            lead = int(rng.integers(arch["lead"][0], arch["lead"][1] + 1))
            days_ago = int(rng.integers(1, 365))
            ts = REF_DATE - np.timedelta64(days_ago, "D")
            check_in = ts + np.timedelta64(lead, "D")
            history.append(dict(history_id=f"B{hid:05d}", user_id=user["user_id"],
                                platform_id=PLATFORMS[int(rng.integers(0, 4))]["platform_id"],
                                offer_id=offer["offer_id"], hotel_id=offer["hotel_id"],
                                booked_at=str(ts), check_in=str(check_in), nights=nights,
                                lead_time_days=lead, season=_season_of(check_in),
                                amount=int(offer["price_per_night"] * nights),
                                guests=int(rng.integers(1, max(2, offer["capacity"] + 1)))))
            hid += 1

    iid = 0
    per_user = max(1, C.DATA.n_interactions // len(users))
    for ui, user in users.iterrows():
        probs = aff[user["archetype_truth"]]
        for _ in range(per_user):
            oi = int(rng.choice(len(offer_ids), p=probs))
            days_ago = int(rng.integers(1, 365))
            ts = REF_DATE - np.timedelta64(days_ago, "D")
            r = rng.random()
            event = "impression" if r < 0.6 else "click" if r < 0.9 else "book"
            interactions.append(dict(interaction_id=f"I{iid:06d}", user_id=user["user_id"],
                                     session_id=f"S{ui:05d}",
                                     platform_id=PLATFORMS[int(rng.integers(0, 4))]["platform_id"],
                                     offer_id=offer_ids[oi], event=event, timestamp=str(ts)))
            iid += 1
    return pd.DataFrame(history), pd.DataFrame(interactions)


def gen_questionnaires(rng, users):
    goal = {"Business_Express": "business", "Family_Vacation": "family", "Luxury_Retreat": "luxury",
            "Solo_Budget": "solo", "Couple_Romantic": "romantic", "Group_Active": "active"}
    rows = []
    for _, u in users.sample(frac=C.DATA.questionnaire_share, random_state=C.RANDOM_STATE).iterrows():
        a = ARCHETYPES[u["archetype_truth"]]
        budget = "low" if a["price"][1] <= 6000 else "high" if a["price"][0] >= 10000 else "mid"
        rows.append(dict(user_id=u["user_id"], budget=budget, trip_goal=goal[u["archetype_truth"]],
                         party=int(rng.integers(1, 5)),
                         needs_kitchen=int("Kitchen" in a["amenities"]),
                         needs_parking=int("Parking" in a["amenities"]),
                         with_kids=int("KidsZone" in a["amenities"])))
    return pd.DataFrame(rows)


def generate_all() -> dict:
    rng = _rng()
    platforms = gen_platforms()
    hotels = gen_hotels(rng)
    offers = gen_offers(rng, hotels)
    users = gen_users(rng)
    history, interactions = gen_history_and_interactions(rng, users, offers)
    questionnaires = gen_questionnaires(rng, users)
    return dict(platforms=platforms, hotels=hotels, offers=offers, users=users,
                history=history, interactions=interactions, questionnaires=questionnaires)
