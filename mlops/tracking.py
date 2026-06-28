"""
Трекинг экспериментов через MLflow.

Логирует параметры, метрики и артефакты прогонов сравнения моделей и подбора
гиперпараметров, а также регистрирует лучшую модель в Model Registry. Если
MLflow не установлен или сервер недоступен, обёртка работает в no-op режиме,
не ломая основной пайплайн.

Адрес сервера берётся из переменной окружения MLFLOW_TRACKING_URI
(по умолчанию http://localhost:5000).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    MLFLOW_AVAILABLE = False

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "trivago-ranking")


def _enabled() -> bool:
    return MLFLOW_AVAILABLE and os.environ.get("MLOPS_TRACKING", "0") == "1"


@contextmanager
def run(name: str, params: dict | None = None):
    """Контекст одного запуска. В no-op режиме просто исполняет тело."""
    if not _enabled():
        yield None
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as r:
        if params:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
        yield r


def _safe_metric_name(k: str) -> str:
    # MLflow допускает в именах метрик только [A-Za-z0-9_./ -]; '@' заменяем
    return str(k).replace("@", "_at_")


def log_metrics(metrics: dict):
    if _enabled():
        mlflow.log_metrics({_safe_metric_name(k): float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})


def log_artifact(path: str):
    if _enabled() and path and os.path.exists(path):
        mlflow.log_artifact(path)


def log_comparison(rows: list[dict], artifacts: list[str] | None = None):
    """Логирует сравнение моделей: по run на модель + общий артефакт-таблицу."""
    if not _enabled():
        return
    for r in rows:
        with run(name=str(r.get("model", "model")), params={"model": r.get("model")}):
            log_metrics({k: v for k, v in r.items() if k != "model"})
    for a in artifacts or []:
        with run(name="artifacts"):
            log_artifact(a)


def register_best(model_path: str, name: str = "trivago_ranker"):
    """Регистрирует лучшую модель в Model Registry (если включено)."""
    if not _enabled():
        return None
    with run(name=f"register-{name}"):
        log_artifact(model_path)
        try:
            return mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{os.path.basename(model_path)}", name)
        except Exception:
            return None
