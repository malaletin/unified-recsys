"""Фикстуры pytest. БД-тесты используют in-memory SQLite через SQLAlchemy."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def synthetic():
    from recsys.dataset import generate_all
    return generate_all()


@pytest.fixture()
def db_session():
    """In-memory SQLite сессия с создаными по моделям таблицами.

    Позволяет гонять БД-тесты без PostgreSQL (модели SQLAlchemy переносимы).
    """
    sa = pytest.importorskip("sqlalchemy")
    from sqlalchemy.orm import sessionmaker
    from app.db.base import Base
    from app.db import models  # noqa: F401

    engine = sa.create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    db = Session()
    try:
        yield db
    finally:
        db.close()
