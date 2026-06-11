"""Управление согласиями на обработку персональных данных (152-ФЗ)."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.repository import Repository
from app.schemas import ConsentIn, ConsentOut
from app.services.engine import ENGINE

router = APIRouter(prefix="/api/v1/consent", tags=["consent"])


@router.post("/grant", response_model=ConsentOut)
def grant(req: ConsentIn, db: Session = Depends(get_db)):
    repo = Repository(db)
    repo.set_consent(req.user_unified_id, "granted")
    db.commit()
    return ConsentOut(user_unified_id=req.user_unified_id, status="granted",
                      updated_at=datetime.now(timezone.utc).isoformat())


@router.post("/revoke", response_model=ConsentOut)
def revoke(req: ConsentIn, db: Session = Depends(get_db)):
    repo = Repository(db)
    repo.set_consent(req.user_unified_id, "revoked")
    removed = repo.anonymize_user(req.user_unified_id)   # право на забвение
    db.commit()
    return ConsentOut(user_unified_id=req.user_unified_id,
                      status=f"revoked (anonymized, {removed} records purged)",
                      updated_at=datetime.now(timezone.utc).isoformat())
