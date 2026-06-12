"""
Генератор синтетического сэмпла в ТОЧНОМ формате Trivago RecSys 2019.

Нужен только для разработки и проверки конвейера офлайн-оценки без доступа
к реальному датасету. Создаёт train.csv и item_metadata.csv с теми же
колонками и семантикой (clickout item, impressions/prices, properties).
Поведение пользователей согласовано со свойствами отелей, поэтому
контентные и со-встречаемостные модели обоснованно превосходят порядок
показа.

Запуск:  python -m scripts.make_trivago_sample --out data/trivago_sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROPERTIES = [
    "1 Star", "2 Star", "3 Star", "4 Star", "5 Star", "WiFi", "Parking",
    "Swimming Pool", "Spa", "Pet Friendly", "Family Friendly", "Beach",
    "Air Conditioning", "Gym", "Restaurant", "Bar", "Sea View", "Kitchen",
    "Business Centre", "Airport Shuttle", "Non-Smoking Rooms", "Balcony",
]
ARCHETYPES = {
    "luxury":   ["5 Star", "Spa", "Swimming Pool", "Sea View", "Restaurant", "Bar"],
    "business": ["4 Star", "WiFi", "Business Centre", "Airport Shuttle", "Gym", "Non-Smoking Rooms"],
    "family":   ["Family Friendly", "Kitchen", "Swimming Pool", "Parking", "Pet Friendly", "Balcony"],
    "budget":   ["2 Star", "WiFi", "Non-Smoking Rooms", "Air Conditioning"],
    "beach":    ["Beach", "Sea View", "Swimming Pool", "Bar", "Balcony", "4 Star"],
}
ARCH_NAMES = list(ARCHETYPES.keys())
PLATFORMS = ["RU", "DE", "US", "UK"]
CITIES = ["Moscow, Russia", "Sochi, Russia", "Berlin, Germany", "Paris, France"]
DEVICES = ["desktop", "mobile", "tablet"]


def gen(n_items=300, n_sessions=2500, seed=42, out="data/trivago_sample"):
    rng = np.random.default_rng(seed)

    # --- item_metadata: свойства отелей по архетипам ---
    item_arch, items = {}, []
    for i in range(1, n_items + 1):
        iid = str(100000 + i)
        arch = ARCH_NAMES[i % len(ARCH_NAMES)]
        item_arch[iid] = arch
        props = set(ARCHETYPES[arch])
        # шумовые свойства
        for p in rng.choice(PROPERTIES, size=int(rng.integers(1, 4)), replace=False):
            props.add(p)
        items.append({"item_id": iid, "properties": "|".join(sorted(props))})
    meta = pd.DataFrame(items)
    pool = {a: [iid for iid, ar in item_arch.items() if ar == a] for a in ARCH_NAMES}
    price_of = {iid: int(rng.normal(120, 40) + 60 * ("5 Star" in item_arch_props(meta, iid)))
                for iid in item_arch}

    # --- train.csv: сессии с контекстом и clickout ---
    rows = []
    ts = 1541030400          # базовый timestamp (нояб. 2018, как в челлендже)
    for s in range(n_sessions):
        arch = ARCH_NAMES[int(rng.integers(0, len(ARCH_NAMES)))]
        uid = f"U{int(rng.integers(0, n_sessions // 2)):05d}"
        sid = f"{uid}-{s:05d}"
        ts += int(rng.integers(30, 600))
        step = 1
        my_pool = pool[arch]

        # предшествующие взаимодействия (контекст сессии)
        prior = list(rng.choice(my_pool, size=int(rng.integers(1, 4)), replace=False))
        for it in prior:
            atype = str(rng.choice(["interaction item image", "interaction item info",
                                    "search for item"]))
            rows.append(_row(uid, sid, ts, step, atype, it, "", ""))
            step += 1; ts += int(rng.integers(5, 60))

        # clickout: target из своего пула, дистракторы из чужих
        target = str(rng.choice(my_pool))
        distractors = [d for d in rng.choice(
            [x for x in item_arch if x != target], size=24, replace=False)]
        imps = distractors + [target]
        rng.shuffle(imps)                                   # порядок показа случайный
        prices = [str(max(20, price_of[i] + int(rng.normal(0, 15)))) for i in imps]
        rows.append(_row(uid, sid, ts, step, "clickout item", target,
                         "|".join(imps), "|".join(prices)))
        ts += int(rng.integers(60, 600))

    train = pd.DataFrame(rows)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    meta.to_csv(out / "item_metadata.csv", index=False)
    print(f"train.csv: {len(train)} строк, item_metadata.csv: {len(meta)} отелей -> {out}")


def item_arch_props(meta, iid):
    row = meta.loc[meta["item_id"] == iid, "properties"]
    return set(row.iloc[0].split("|")) if len(row) else set()


def _row(uid, sid, ts, step, atype, ref, imps, prices):
    return {
        "user_id": uid, "session_id": sid, "timestamp": ts, "step": step,
        "action_type": atype, "reference": ref,
        "platform": np.random.choice(PLATFORMS), "city": np.random.choice(CITIES),
        "device": np.random.choice(DEVICES), "current_filters": "",
        "impressions": imps, "prices": prices,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/trivago_sample")
    ap.add_argument("--sessions", type=int, default=2500)
    ap.add_argument("--items", type=int, default=300)
    a = ap.parse_args()
    gen(n_items=a.items, n_sessions=a.sessions, out=a.out)
