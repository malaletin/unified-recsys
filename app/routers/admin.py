"""Административные операции: ручной batch-пересчёт профилей и сегментов."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.repository import Repository
from app.services.engine import ENGINE

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/recompute")
def recompute(db: Session = Depends(get_db)):
    repo = Repository(db)
    info = ENGINE.fit_from_repo(repo)
    db.commit()
    return {"recomputed": True, **info}
