"""Health-check и статус движка."""
from fastapi import APIRouter

from app.services.engine import ENGINE

router = APIRouter(tags=["system"])


@router.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "last_fit": str(ENGINE.last_fit),
        "purity": ENGINE.purity,
        "segments": ENGINE.recommender.segmentation.cluster_labels if ENGINE.recommender else None,
    }
