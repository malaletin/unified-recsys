"""
Batch-планировщик пересчёта профилей.

В продакшене запускается как отдельный процесс (cron / Celery beat) и
каждые 12 часов (08:00 и 20:00 МСК) перестраивает сегменты и профили.
Здесь — самодостаточная реализация на стандартной библиотеке.

Запуск разового пересчёта:  python -m batch.scheduler --once
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from app.config import settings
from app.db.base import SessionLocal
from app.db.repository import Repository
from app.services.engine import ENGINE

RECOMPUTE_HOURS_MSK = (8, 20)


def recompute_once() -> dict:
    db = SessionLocal()
    try:
        info = ENGINE.fit_from_repo(Repository(db))
        db.commit()
        print(f"[{datetime.now(timezone.utc).isoformat()}] recompute: {info['duration_sec']}s, "
              f"users={info['users']}")
        return info
    finally:
        db.close()


def run_forever(poll_seconds: int = 1800):
    last_hour = None
    while True:
        now = datetime.now(timezone.utc)
        if now.hour in RECOMPUTE_HOURS_MSK and now.hour != last_hour:
            recompute_once()
            last_hour = now.hour
        time.sleep(poll_seconds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="разовый пересчёт и выход")
    args = ap.parse_args()
    if args.once:
        recompute_once()
    else:
        run_forever()
