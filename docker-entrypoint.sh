#!/usr/bin/env bash
set -e

# Ожидание доступности PostgreSQL
echo "Ожидание PostgreSQL..."
python - <<'PY'
import time, os
from sqlalchemy import create_engine, text
url = os.environ.get("DATABASE_URL", "postgresql+psycopg://recsys:recsys@db:5432/recsys")
for i in range(30):
    try:
        create_engine(url).connect().execute(text("SELECT 1"))
        print("PostgreSQL доступен."); break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("PostgreSQL недоступен")
PY

# Применение миграций
echo "Применение миграций Alembic..."
alembic upgrade head

# Первичное наполнение данными (только если БД пуста)
echo "Проверка наличия данных..."
python - <<'PY'
from app.db.base import SessionLocal
from app.db.repository import Repository
db = SessionLocal()
try:
    if Repository(db).list_offers_df().empty:
        print("БД пуста — выполняю seed...")
        from scripts.seed_data import seed
        seed()
    else:
        print("Данные уже загружены.")
finally:
    db.close()
PY

echo "Запуск API..."
exec "$@"
