"""Тесты слоя БД и репозитория (in-memory SQLite через SQLAlchemy)."""
from __future__ import annotations

from datetime import datetime


def _seed_minimal(repo):
    repo.upsert_platform(platform_id="P1", name="Островок", kind="OTA")
    repo.db.flush()
    from app.db.models import Hotel, Offer
    repo.db.add(Hotel(hotel_id="H01", name="H", city="Сочи", stars=5, chain="Cosmos",
                      rating=9.0, poi_distance_km=1.0))
    repo.db.add(Offer(offer_id="O001", hotel_id="H01", city="Сочи", stars=5, rating=9.0,
                      poi_distance_km=1.0, room_type="Suite", price_per_night=12000,
                      meal_plan="HB", capacity=2, min_nights=2, season="high", amenities=["SPA"]))
    repo.db.flush()


def test_unified_id_and_history(db_session):
    from app.db.repository import Repository
    repo = Repository(db_session)
    _seed_minimal(repo)
    repo.set_consent("U1", "granted")
    repo.add_booking(unified_id="U1", platform_id="P1", offer_id="O001", hotel_id="H01",
                     booked_at=datetime(2026, 3, 1), check_in=datetime(2026, 4, 1),
                     nights=3, lead_time_days=31, amount=36000, guests=2)
    db_session.commit()
    assert repo.history_count("U1") == 1
    assert repo.is_granted("U1")
    df = repo.history_df(granted_only=True)
    assert len(df) == 1 and df.iloc[0]["user_id"] == "U1"


def test_consent_revoke_anonymizes(db_session):
    from app.db.repository import Repository
    repo = Repository(db_session)
    _seed_minimal(repo)
    repo.set_consent("U2", "granted")
    repo.add_booking(unified_id="U2", platform_id="P1", offer_id="O001", hotel_id="H01",
                     booked_at=datetime(2026, 3, 1), check_in=datetime(2026, 4, 1),
                     nights=2, lead_time_days=10, amount=24000, guests=2)
    db_session.commit()
    assert repo.history_count("U2") == 1
    repo.set_consent("U2", "revoked")
    purged = repo.anonymize_user("U2")
    db_session.commit()
    assert purged == 1
    assert repo.history_count("U2") == 0
    assert not repo.is_granted("U2")


def test_cross_platform_aggregation(db_session):
    from app.db.repository import Repository
    repo = Repository(db_session)
    _seed_minimal(repo)
    repo.upsert_platform(platform_id="P2", name="Яндекс", kind="ecosystem")
    repo.set_consent("U3", "granted")
    for pid in ("P1", "P2"):
        repo.add_booking(unified_id="U3", platform_id=pid, offer_id="O001", hotel_id="H01",
                         booked_at=datetime(2026, 3, 1), check_in=datetime(2026, 4, 1),
                         nights=2, lead_time_days=10, amount=24000, guests=2)
    db_session.commit()
    df = repo.history_df()
    assert set(df[df["user_id"] == "U3"]["platform_id"]) == {"P1", "P2"}
