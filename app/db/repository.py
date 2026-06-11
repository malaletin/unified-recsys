"""
Слой доступа к данным (Repository).

Инкапсулирует все запросы к БД: наполнение справочников, регистрацию
unified_id, запись истории и событий, управление согласиями и выгрузку
данных в pandas для рекомендательного движка.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import (BookingHistory, Consent, InteractionEvent, Offer,
                           Platform, RecommendationLog, User, UserProfile)


class Repository:
    def __init__(self, db: Session):
        self.db = db

    # ----- справочники ----- #
    def upsert_platform(self, **kw):
        obj = self.db.get(Platform, kw["platform_id"]) or Platform(**kw)
        for k, v in kw.items():
            setattr(obj, k, v)
        self.db.merge(obj)

    def upsert_offer(self, **kw):
        self.db.merge(Offer(**kw))

    def list_offers_df(self) -> pd.DataFrame:
        rows = self.db.execute(select(Offer)).scalars().all()
        return pd.DataFrame([{c.name: getattr(o, c.name) for c in Offer.__table__.columns}
                             for o in rows])

    def offers_by_ids(self, offer_ids: list[str]) -> list[Offer]:
        return self.db.execute(select(Offer).where(Offer.offer_id.in_(offer_ids))).scalars().all()

    # ----- Unified ID ----- #
    def get_or_create_user(self, unified_id: str, region=None, loyalty="new") -> User:
        user = self.db.get(User, unified_id)
        if user is None:
            user = User(unified_id=unified_id, region=region, loyalty=loyalty)
            self.db.add(user)
            self.db.flush()
        return user

    # ----- история и события ----- #
    def add_booking(self, **kw) -> BookingHistory:
        self.get_or_create_user(kw["unified_id"])
        obj = BookingHistory(**kw)
        self.db.add(obj)
        return obj

    def add_event(self, **kw) -> InteractionEvent:
        self.get_or_create_user(kw["unified_id"])
        obj = InteractionEvent(**kw)
        self.db.add(obj)
        return obj

    def history_count(self, unified_id: str) -> int:
        return self.db.query(BookingHistory).filter_by(unified_id=unified_id).count()

    # ----- выгрузка для движка ----- #
    def history_df(self, granted_only: bool = True) -> pd.DataFrame:
        q = select(BookingHistory)
        rows = self.db.execute(q).scalars().all()
        df = pd.DataFrame([{
            "history_id": r.id, "user_id": r.unified_id, "platform_id": r.platform_id,
            "offer_id": r.offer_id, "hotel_id": r.hotel_id, "booked_at": str(r.booked_at),
            "check_in": str(r.check_in), "nights": r.nights, "lead_time_days": r.lead_time_days,
            "season": r.season, "amount": r.amount, "guests": r.guests} for r in rows])
        if granted_only and not df.empty:
            granted = self.granted_user_ids()
            df = df[df["user_id"].isin(granted)]
        return df

    def interactions_df(self, granted_only: bool = True) -> pd.DataFrame:
        rows = self.db.execute(select(InteractionEvent)).scalars().all()
        df = pd.DataFrame([{
            "interaction_id": r.id, "user_id": r.unified_id, "session_id": r.session_id,
            "platform_id": r.platform_id, "offer_id": r.offer_id, "event": r.event,
            "timestamp": str(r.timestamp)} for r in rows])
        if granted_only and not df.empty:
            df = df[df["user_id"].isin(self.granted_user_ids())]
        return df

    def users_df(self) -> pd.DataFrame:
        rows = self.db.execute(select(User)).scalars().all()
        out = []
        for u in rows:
            prof = self.db.get(UserProfile, u.unified_id)
            out.append({"user_id": u.unified_id, "region": u.region, "loyalty": u.loyalty,
                        "consent_status": self.consent_status(u.unified_id),
                        "archetype_truth": (prof.preferences or {}).get("archetype_truth") if prof else None})
        return pd.DataFrame(out)

    # ----- согласия (152-ФЗ) ----- #
    def set_consent(self, unified_id: str, status: str, source="onboarding_checkbox") -> Consent:
        self.get_or_create_user(unified_id)
        c = Consent(unified_id=unified_id, status=status, source=source,
                    updated_at=datetime.now(timezone.utc))
        self.db.add(c)
        return c

    def consent_status(self, unified_id: str) -> str:
        c = self.db.execute(
            select(Consent).where(Consent.unified_id == unified_id)
            .order_by(Consent.updated_at.desc(), Consent.id.desc()).limit(1)
        ).scalars().first()
        return c.status if c else "unknown"

    def is_granted(self, unified_id: str) -> bool:
        return self.consent_status(unified_id) == "granted"

    def granted_user_ids(self) -> set[str]:
        rows = self.db.execute(select(User.unified_id)).scalars().all()
        return {uid for uid in rows if self.is_granted(uid)}

    def anonymize_user(self, unified_id: str) -> int:
        """Право на забвение: удаление истории и событий пользователя."""
        n = self.db.query(BookingHistory).filter_by(unified_id=unified_id).count()
        self.db.execute(delete(BookingHistory).where(BookingHistory.unified_id == unified_id))
        self.db.execute(delete(InteractionEvent).where(InteractionEvent.unified_id == unified_id))
        prof = self.db.get(UserProfile, unified_id)
        if prof:
            prof.cluster_id = None
            prof.cluster_label = None
            prof.history_len = 0
            prof.preferences = {}
        return n

    # ----- профили ----- #
    def upsert_profile(self, unified_id: str, cluster_id=None, cluster_label=None,
                       history_len=0, preferences=None):
        prof = self.db.get(UserProfile, unified_id)
        if prof is None:
            prof = UserProfile(unified_id=unified_id)
            self.db.add(prof)
        prof.cluster_id = cluster_id
        prof.cluster_label = cluster_label
        prof.history_len = history_len
        if preferences is not None:
            prof.preferences = preferences

    # ----- лог выдач ----- #
    def log_recommendation(self, unified_id, platform_id, offers, cold_start=False, context=None):
        self.db.add(RecommendationLog(unified_id=unified_id, platform_id=platform_id,
                                      offers=offers, cold_start=cold_start, context=context or {}))
