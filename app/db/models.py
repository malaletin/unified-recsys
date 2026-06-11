"""
ORM-модели (SQLAlchemy 2.0).

Центральная сущность — User с полем unified_id: единый идентификатор
путешественника, под которым агрегируется кросс-платформенная история
бронирований (BookingHistory) и логи поведения (InteractionEvent).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String,
                        JSON, Index, func)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Platform(Base):
    __tablename__ = "platforms"
    platform_id: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))


class Hotel(Base):
    __tablename__ = "hotels"
    hotel_id: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    city: Mapped[str] = mapped_column(String(64), index=True)
    stars: Mapped[int] = mapped_column(Integer)
    chain: Mapped[str] = mapped_column(String(64))
    rating: Mapped[float] = mapped_column(Float)
    poi_distance_km: Mapped[float] = mapped_column(Float)
    offers: Mapped[list["Offer"]] = relationship(back_populates="hotel")


class Offer(Base):
    __tablename__ = "offers"
    offer_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    hotel_id: Mapped[str] = mapped_column(ForeignKey("hotels.hotel_id"), index=True)
    city: Mapped[str] = mapped_column(String(64), index=True)
    stars: Mapped[int] = mapped_column(Integer)
    rating: Mapped[float] = mapped_column(Float)
    poi_distance_km: Mapped[float] = mapped_column(Float)
    room_type: Mapped[str] = mapped_column(String(32))
    price_per_night: Mapped[int] = mapped_column(Integer, index=True)
    meal_plan: Mapped[str] = mapped_column(String(8))
    capacity: Mapped[int] = mapped_column(Integer)
    min_nights: Mapped[int] = mapped_column(Integer)
    season: Mapped[str] = mapped_column(String(8))
    amenities: Mapped[list] = mapped_column(JSON, default=list)   # JSONB в PostgreSQL
    hotel: Mapped["Hotel"] = relationship(back_populates="offers")


class User(Base):
    """Единый профиль путешественника. unified_id — сквозной идентификатор."""
    __tablename__ = "users"
    unified_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    region: Mapped[str] = mapped_column(String(64), nullable=True)
    loyalty: Mapped[str] = mapped_column(String(16), default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    consents: Mapped[list["Consent"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    history: Mapped[list["BookingHistory"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    events: Mapped[list["InteractionEvent"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    profile: Mapped["UserProfile"] = relationship(back_populates="user", uselist=False,
                                                  cascade="all, delete-orphan")


class Consent(Base):
    """Журнал согласий на обработку ПДн (152-ФЗ)."""
    __tablename__ = "consents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    unified_id: Mapped[str] = mapped_column(ForeignKey("users.unified_id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16))           # granted | revoked
    purpose: Mapped[str] = mapped_column(String(64), default="personalization")
    source: Mapped[str] = mapped_column(String(64), default="onboarding_checkbox")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped["User"] = relationship(back_populates="consents")


class BookingHistory(Base):
    """Кросс-платформенная история бронирований, агрегированная по unified_id."""
    __tablename__ = "booking_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    unified_id: Mapped[str] = mapped_column(ForeignKey("users.unified_id", ondelete="CASCADE"), index=True)
    platform_id: Mapped[str] = mapped_column(ForeignKey("platforms.platform_id"))
    offer_id: Mapped[str] = mapped_column(ForeignKey("offers.offer_id"))
    hotel_id: Mapped[str] = mapped_column(String(8))
    booked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    check_in: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    nights: Mapped[int] = mapped_column(Integer)
    lead_time_days: Mapped[int] = mapped_column(Integer, default=0)
    season: Mapped[str] = mapped_column(String(8), default="any")
    amount: Mapped[int] = mapped_column(Integer)
    guests: Mapped[int] = mapped_column(Integer, default=1)
    user: Mapped["User"] = relationship(back_populates="history")

    __table_args__ = (Index("ix_history_user_time", "unified_id", "booked_at"),)


class InteractionEvent(Base):
    """Логи поведения в виджете: impression / click / book."""
    __tablename__ = "interaction_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    unified_id: Mapped[str] = mapped_column(ForeignKey("users.unified_id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String(32))
    platform_id: Mapped[str] = mapped_column(ForeignKey("platforms.platform_id"))
    offer_id: Mapped[str] = mapped_column(ForeignKey("offers.offer_id"))
    event: Mapped[str] = mapped_column(String(16))           # impression | click | book
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    user: Mapped["User"] = relationship(back_populates="events")


class UserProfile(Base):
    """Производный профиль пользователя (обновляется batch-пайплайном)."""
    __tablename__ = "user_profiles"
    unified_id: Mapped[str] = mapped_column(ForeignKey("users.unified_id", ondelete="CASCADE"), primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=True)
    cluster_label: Mapped[str] = mapped_column(String(32), nullable=True)
    history_len: Mapped[int] = mapped_column(Integer, default=0)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())
    user: Mapped["User"] = relationship(back_populates="profile")


class RecommendationLog(Base):
    """Журнал выдач для аналитики и A/B-тестов."""
    __tablename__ = "recommendation_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    unified_id: Mapped[str] = mapped_column(String(32), index=True, nullable=True)
    platform_id: Mapped[str] = mapped_column(String(8))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    cold_start: Mapped[bool] = mapped_column(Boolean, default=False)
    offers: Mapped[list] = mapped_column(JSON, default=list)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
