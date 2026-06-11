"""
Точка входа FastAPI-приложения (Widget API).

Запуск:  uvicorn app.main:app --reload
Swagger: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.db.base import SessionLocal, engine, Base
from app.db.repository import Repository
from app.routers import admin, consent, events, health, recommend
from app.services.engine import ENGINE


@asynccontextmanager
async def lifespan(app: FastAPI):
    # На старте создаём таблицы (для dev; в проде — Alembic) и обучаем движок,
    # если в БД уже есть данные.
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        repo = Repository(db)
        if not repo.list_offers_df().empty:
            ENGINE.fit_from_repo(repo)
            db.commit()
    except Exception:  # pragma: no cover — пустая БД при первом старте
        db.rollback()
    finally:
        db.close()
    yield


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)
app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(events.router)
app.include_router(consent.router)
app.include_router(admin.router)


@app.get("/")
def root():
    return {"service": settings.api_title, "version": settings.api_version, "docs": "/docs"}
