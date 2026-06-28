# Трек 4 — промышленная MLOps-инфраструктура

Доведение системы до production-grade уровня: масштабируемый ANN-поиск,
распределённый кэш, трекинг экспериментов и реестр моделей, мониторинг и
конвейер переобучения.

## Компоненты

| Компонент | Назначение | Файл / сервис |
|-----------|-----------|---------------|
| ANN-поиск (FAISS) | масштабируемый поиск похожих отелей вместо полного перебора; numpy-fallback | `mlops/ann.py` |
| Redis-кэш | распределённый кэш готовых выдач (TTL), общий для инстансов API; in-memory fallback | `app/services/cache.py` |
| MLflow | трекинг параметров/метрик экспериментов и Model Registry | `mlops/tracking.py`, сервис `mlflow` |
| Prometheus | сбор метрик сервиса (RPS, латентность, cache hit, cold start) | `app/observability.py`, `monitoring/prometheus.yml` |
| Grafana | дашборд мониторинга | `monitoring/grafana-dashboard.json` |
| Конвейер переобучения | плановое переобучение + регистрация лучшей модели | `mlops/retrain.py` |

Все интеграции выполнены через **fallback-паттерн**: если библиотека или сервис
недоступны, система продолжает работать (numpy вместо FAISS, in-memory вместо
Redis, no-op вместо MLflow/Prometheus). Это не ломает основной пайплайн и тесты.

## Установка

```powershell
pip install -r requirements-mlops.txt
```

## Запуск полного стека (Docker)

```powershell
docker compose -f docker-compose.yml -f docker-compose.mlops.yml up --build
```

Поднимутся: PostgreSQL, API, планировщик, **Redis**, **MLflow**, **Prometheus**,
**Grafana**. Адреса:

- API / Swagger — http://localhost:8000/docs
- Метрики Prometheus сервиса — http://localhost:8000/metrics
- MLflow UI — http://localhost:5000
- Prometheus — http://localhost:9090
- Grafana — http://localhost:3000 (anonymous / admin:admin), дашборд «RecSys Service»

API в составе стека автоматически получает `REDIS_URL` и `MLFLOW_TRACKING_URI`
через переменные окружения (см. `docker-compose.mlops.yml`).

## Трекинг экспериментов и переобучение

Включить логирование сравнения моделей в MLflow:

```powershell
$env:MLOPS_TRACKING = "1"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m scripts.compare_models --data data\trivago --stream --nrows 2000000 --epochs 10
```

Плановое переобучение с регистрацией лучшей модели:

```powershell
python -m mlops.retrain --data data\trivago --stream --nrows 2000000
```

## Мониторинг

FastAPI экспортирует метрики на `/metrics`:

- `recsys_requests_total{endpoint,status}` — число запросов;
- `recsys_request_latency_seconds` — гистограмма латентности;
- `recsys_recommendations_total`, `recsys_cold_start_total` — выдачи и доля cold start;
- `recsys_cache_hit_rate` — доля попаданий в кэш.

Prometheus собирает их, Grafana визуализирует (RPS, p95-латентность, hit rate,
доля cold start).

## ANN-поиск

`mlops/ann.py` строит индекс приближённого поиска (FAISS `IndexFlatIP` по
нормализованным векторам ≈ косинус). При большом каталоге это на порядки
быстрее полного перебора. Если FAISS не установлен — используется корректный
numpy-перебор с тем же интерфейсом (`search`, `rank_subset`).
