"""Приём событий: бронирования (booking_confirmed) и логи поведения."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.repository import Repository
from app.schemas import BookingIn, EventIn
from app.services.engine import ENGINE

router = APIRouter(prefix="/api/v1/events", tags=["events"])


def _parse(dt: str) -> datetime:
    return datetime.fromisoformat(dt.replace("Z", "+00:00")) if "T" in dt else datetime.fromisoformat(dt)


@router.post("/booking")
def booking_confirmed(ev: BookingIn, db: Session = Depends(get_db)):
    repo = Repository(db)
    if not repo.is_granted(ev.user_unified_id):
        raise HTTPException(status_code=403, detail="consent required")
    repo.add_booking(unified_id=ev.user_unified_id, platform_id=ev.platform_id,
                     offer_id=ev.offer_id, hotel_id=ev.hotel_id,
                     booked_at=datetime.utcnow(), check_in=_parse(ev.check_in),
                     nights=ev.nights, lead_time_days=ev.lead_time_days,
                     amount=ev.amount, guests=ev.guests)
    db.commit()
    # триггер booking_confirmed: внеплановый пересчёт профилей
    info = ENGINE.fit_from_repo(repo)
    db.commit()
    return {"event": "booking_confirmed", "recomputed": True, "duration_sec": info["duration_sec"]}


@router.post("/track")
def track_event(ev: EventIn, db: Session = Depends(get_db)):
    repo = Repository(db)
    if not repo.is_granted(ev.user_unified_id):
        raise HTTPException(status_code=403, detail="consent required")
    repo.add_event(unified_id=ev.user_unified_id, session_id=ev.session_id,
                   platform_id=ev.platform_id, offer_id=ev.offer_id,
                   event=ev.event, timestamp=datetime.utcnow())
    db.commit()
    return {"event": ev.event, "recorded": True}
