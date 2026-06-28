"""
Наблюдаемость сервиса: метрики Prometheus.

Экспортирует счётчики и гистограммы (число запросов, латентность, доля cold
start) и эндпоинт /metrics для сбора Prometheus. Если prometheus_client не
установлен, инструментирование тихо отключается.
"""
from __future__ import annotations

import time

try:
    from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge,
                                   Histogram, generate_latest)
    PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    PROM_AVAILABLE = False

if PROM_AVAILABLE:
    REQUESTS = Counter("recsys_requests_total", "Число запросов", ["endpoint", "status"])
    LATENCY = Histogram("recsys_request_latency_seconds", "Латентность запроса", ["endpoint"])
    RECS = Counter("recsys_recommendations_total", "Число выданных рекомендаций")
    COLD = Counter("recsys_cold_start_total", "Число выдач в режиме cold start")
    CACHE_HIT = Gauge("recsys_cache_hit_rate", "Доля попаданий в кэш")


def observe_recommend(cold_start: bool, n: int):
    if PROM_AVAILABLE:
        RECS.inc(n)
        if cold_start:
            COLD.inc()


def set_cache_hit_rate(rate: float):
    if PROM_AVAILABLE:
        CACHE_HIT.set(rate)


def setup_observability(app):
    """Добавляет middleware латентности и эндпоинт /metrics к FastAPI-приложению."""
    if not PROM_AVAILABLE:
        return app
    from fastapi import Response

    @app.middleware("http")
    async def _metrics_mw(request, call_next):
        t0 = time.time()
        endpoint = request.url.path
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            LATENCY.labels(endpoint=endpoint).observe(time.time() - t0)
            REQUESTS.labels(endpoint=endpoint, status=locals().get("status", "err")).inc()

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
