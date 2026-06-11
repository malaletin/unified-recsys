"""Pydantic-схемы запросов и ответов Widget API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Questionnaire(BaseModel):
    budget: str = "mid"            # low | mid | high
    trip_goal: str = "solo"       # business | family | luxury | solo | romantic | active
    party: int = 2
    needs_kitchen: int = 0
    needs_parking: int = 0
    with_kids: int = 0


class RecommendRequest(BaseModel):
    user_unified_id: Optional[str] = Field(None, description="Единый идентификатор пользователя")
    platform_id: str
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    guests: int = 2
    available_offers: list[str]
    questionnaire: Optional[Questionnaire] = None
    top_k: int = 5


class RecommendationItem(BaseModel):
    offer_id: str
    rank: int
    score: float
    cluster_label: Optional[str] = None


class RecommendResponse(BaseModel):
    profile_ref: str
    consent: str
    cold_start: bool
    recommendations: list[RecommendationItem]


class BookingIn(BaseModel):
    user_unified_id: str
    platform_id: str
    offer_id: str
    hotel_id: str
    check_in: str
    nights: int = 1
    lead_time_days: int = 0
    amount: int = 0
    guests: int = 1


class EventIn(BaseModel):
    user_unified_id: str
    platform_id: str
    offer_id: str
    event: str                     # impression | click | book
    session_id: str = "external"


class ConsentIn(BaseModel):
    user_unified_id: str


class ConsentOut(BaseModel):
    user_unified_id: str
    status: str
    updated_at: str
