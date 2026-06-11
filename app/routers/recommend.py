"""Эндпоинт ранжированной выдачи для встраивания в виджет площадки."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.repository import Repository
from app.schemas import RecommendRequest, RecommendResponse
from app.services.engine import ENGINE

router = APIRouter(prefix="/api/v1/widget", tags=["widget"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, db: Session = Depends(get_db)):
    repo = Repository(db)
    quest = req.questionnaire.model_dump() if req.questionnaire else None
    result = ENGINE.recommend(
        repo,
        unified_id=req.user_unified_id,
        platform_id=req.platform_id,
        available_offers=req.available_offers,
        questionnaire=quest,
        check_in=req.check_in,
        guests=req.guests,
        top_k=req.top_k,
    )
    db.commit()
    return result
