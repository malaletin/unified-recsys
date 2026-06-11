# Контрибьютинг

Спасибо за интерес к проекту! Ниже — короткий гайд для разработчиков.

## Среда разработки

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # при необходимости поправьте DATABASE_URL
```

Для локальной отладки без PostgreSQL можно указать в `.env`:
`DATABASE_URL=sqlite:///./dev.db`.

## Перед коммитом

```bash
ruff check .        # линтинг
pytest -q           # тесты (БД-тесты идут на in-memory SQLite)
```

## Стиль

- Длина строки — 110 символов (см. `pyproject.toml`).
- Бизнес-логика рекомендаций живёт в пакете `recsys/` и не зависит от БД/веба.
- Доступ к данным — только через `app/db/repository.py`.
- Новые таблицы добавляются миграцией Alembic: `alembic revision -m "..."`.

## Pull Request

1. Создайте ветку от `main`.
2. Добавьте тесты к изменениям.
3. Убедитесь, что CI зелёный (ruff + pytest + миграции на PostgreSQL).
