"""
Наполнение БД синтетическими данными.

Генерирует кросс-платформенный датасет и загружает его в PostgreSQL через
ORM: справочники, пользователей (unified_id), согласия, историю, события.

Запуск:  python -m scripts.seed_data
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from app.db.base import Base, SessionLocal, engine
from app.db.models import (BookingHistory, Consent, Hotel, InteractionEvent,
                           Offer, Platform, User)
from recsys.dataset import generate_all


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s).split("T")[0])


def seed():
    Base.metadata.create_all(bind=engine)
    t = generate_all()
    db = SessionLocal()
    try:
        for _, r in t["platforms"].iterrows():
            db.merge(Platform(platform_id=r["platform_id"], name=r["name"], kind=r["kind"]))
        for _, r in t["hotels"].iterrows():
            db.merge(Hotel(hotel_id=r["hotel_id"], name=r["name"], city=r["city"], stars=int(r["stars"]),
                           chain=r["chain"], rating=float(r["rating"]),
                           poi_distance_km=float(r["poi_distance_km"])))
        for _, r in t["offers"].iterrows():
            db.merge(Offer(offer_id=r["offer_id"], hotel_id=r["hotel_id"], city=r["city"],
                           stars=int(r["stars"]), rating=float(r["rating"]),
                           poi_distance_km=float(r["poi_distance_km"]), room_type=r["room_type"],
                           price_per_night=int(r["price_per_night"]), meal_plan=r["meal_plan"],
                           capacity=int(r["capacity"]), min_nights=int(r["min_nights"]),
                           season=r["season"], amenities=r["amenities"].split("|")))
        for _, r in t["users"].iterrows():
            db.merge(User(unified_id=r["user_id"], region=r["region"], loyalty=r["loyalty"]))
            db.merge(Consent(unified_id=r["user_id"], status=r["consent_status"]))
        db.flush()
        for _, r in t["history"].iterrows():
            db.add(BookingHistory(unified_id=r["user_id"], platform_id=r["platform_id"],
                                  offer_id=r["offer_id"], hotel_id=r["hotel_id"],
                                  booked_at=_dt(r["booked_at"]), check_in=_dt(r["check_in"]),
                                  nights=int(r["nights"]), lead_time_days=int(r["lead_time_days"]),
                                  season=r["season"], amount=int(r["amount"]), guests=int(r["guests"])))
        for _, r in t["interactions"].iterrows():
            db.add(InteractionEvent(unified_id=r["user_id"], session_id=r["session_id"],
                                    platform_id=r["platform_id"], offer_id=r["offer_id"],
                                    event=r["event"], timestamp=_dt(r["timestamp"])))
        db.commit()
        print(f"Загружено: {len(t['users'])} users, {len(t['offers'])} offers, "
              f"{len(t['history'])} bookings, {len(t['interactions'])} events")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
