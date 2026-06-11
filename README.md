# Unified Cross-Platform Hotel RecSys

Кросс-платформенная рекомендательная система для гостиничного бизнеса:
встраиваемый **FastAPI**-модуль, централизованная **PostgreSQL**-база с
единым профилем путешественника (**Unified ID**) и расширенный гибридный
рекомендательный движок (Content-Based + Collaborative SVD + кластеризация +
контекст + Time Decay).

[![CI](https://github.com/malaletin/unified-recsys/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Возможности

- **Widget API (FastAPI)** — встраивается в систему любого отеля или площадки через REST.
- **Unified ID + PostgreSQL** — единый идентификатор пользователя, под которым агрегируется кросс-платформенная история бронирований и события поведения.
- **Расширенный гибрид** — четыре контура ранжирования: контентный (CB), коллаборативный на латентных факторах (TruncatedSVD), кластерный буст (K-Means, k=10) и контекстный (сезон/даты); взвешивание сигналов с экспоненциальным затуханием по времени (Time Decay).
- **Batch-обновление** — пересчёт профилей по расписанию (12 ч) и по событию `booking_confirmed`.
- **152-ФЗ** — журнал согласий, отзыв согласия с анонимизацией (право на забвение), обезличенная выдача партнёрам.
- **Инфраструктура** — Alembic-миграции, Docker Compose (API + PostgreSQL + планировщик), GitHub Actions CI.

## Архитектура

```
app/                     Веб-приложение и доступ к данным
  main.py                FastAPI: сборка роутеров, lifespan
  config.py              Настройки (pydantic-settings, DATABASE_URL)
  db/
    base.py              Engine + сессии SQLAlchemy
    models.py            ORM-модели (User/Unified ID, History, Events, Consent, ...)
    repository.py        Слой доступа к данным
  routers/               recommend, events, consent, admin, health
  services/engine.py     Связка БД <-> рекомендательный движок
recsys/                  Чистое ядро рекомендаций (без зависимостей от БД/веба)
  mlcore.py              numpy KMeans / StandardScaler / TruncatedSVD / cosine
  content.py             Контентный контур (CB)
  collaborative.py       Коллаборативный контур (SVD + Time Decay)
  clustering.py          Сегментация (K-Means -> бизнес-сегменты)
  context.py             Контекстный контур (сезон/даты)
  hybrid.py              Комбайнер четырёх контуров + Cold Start
  dataset.py             Генератор синтетических данных
  simulation.py          Имитация сессий и валидация
batch/scheduler.py       Планировщик batch-пересчёта
migrations/              Alembic
scripts/                 seed_data, run_simulation, db_smoke, make_charts
tests/                   pytest (recsys + БД на in-memory SQLite)
```

Рекомендательное ядро (`recsys/`) намеренно не знает про БД и веб — это
упрощает тестирование и позволяет переиспользовать движок где угодно.

## Быстрый старт (Docker)

```bash
cp .env.example .env
docker compose up --build
```

Поднимутся PostgreSQL, API (с автоприменением миграций и сидом данных) и
планировщик. После старта:

- Swagger UI — http://localhost:8000/docs
- Health — http://localhost:8000/api/v1/health

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# вариант с PostgreSQL (поднимите вручную) или sqlite для отладки:
export DATABASE_URL=sqlite:///./dev.db

alembic upgrade head
python -m scripts.seed_data
uvicorn app.main:app --reload
```

Без поднятия БД можно проверить схему и сквозной поток одной командой:

```bash
python -m scripts.db_smoke          # ingest -> история по unified_id -> рекомендация
python -m scripts.run_simulation    # реальная валидация + графики в results/
```

## API

| Метод | Endpoint | Назначение |
|-------|----------|-----------|
| POST | `/api/v1/widget/recommend` | Ранжированная выдача топ-K для виджета |
| POST | `/api/v1/events/booking` | Триггер `booking_confirmed` (+ пересчёт профиля) |
| POST | `/api/v1/events/track` | Лог события (impression/click/book) |
| POST | `/api/v1/consent/grant` | Выдача согласия (онбординг) |
| POST | `/api/v1/consent/revoke` | Отзыв согласия + анонимизация (152-ФЗ) |
| POST | `/api/v1/admin/recompute` | Ручной batch-пересчёт сегментов |
| GET | `/api/v1/health` | Статус сервиса и сегменты |

Пример:

```bash
curl -X POST http://localhost:8000/api/v1/widget/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_unified_id":"U0078","platform_id":"P1","guests":2,
       "available_offers":["O001","O004","O007","O022","O030"]}'
```

Ответ содержит обезличенный `profile_ref`, статус согласия, флаг `cold_start`
и ранжированный список с меткой сегмента.

## Модель данных

Центральная таблица `users` хранит `unified_id`. К нему привязаны
`booking_history` (кросс-платформенные брони), `interaction_events`
(поведение), `consents` (152-ФЗ), `user_profiles` (производный сегмент и
вектор предпочтений). Полная схема — в `app/db/models.py` и миграции
`migrations/versions/0001_initial.py`.

## Результаты валидации (реальный прогон, 2000 сессий)

| Метрика | Baseline (популярность) | Расширенный гибрид | Дельта |
|---------|------------------------|--------------------|--------|
| CTR (топ-5) | 52.6% | 67.7% | **+28.6%** |
| Time-to-Book (медиана) | 11.42 ч | 7.26 ч | **−36.4%** |
| Conversion Rate | 39.8% | 52.3% | **+31.5%** |

Чистота кластеризации к скрытым сегментам — **0.69**. Оптимальный вес
коллаборативного контура `w_collab = 0.45`. По сценариям CTR растёт с
накоплением кросс-платформенной истории: S1 (новый) 58.3% → S2
(кросс-платформенный) 70.2% → S3 (возвратный) 76.5%, что подтверждает
ценность Unified ID.

## Тесты и CI

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

GitHub Actions поднимает сервис PostgreSQL, применяет миграции, прогоняет
линтер, тесты (БД-тесты — на in-memory SQLite) и smoke-проверку движка.

## Лицензия

MIT — см. [LICENSE](LICENSE).
