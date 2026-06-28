"""
Конвейер переобучения (retraining) с трекингом в MLflow.

Запускается по расписанию (cron / Celery beat / GitHub Actions) и заново
прогоняет сравнение моделей на свежих данных, логирует параметры и метрики в
MLflow и регистрирует лучшую модель в Model Registry. При выключенном трекинге
(MLOPS_TRACKING != 1) просто выполняет сравнение.

Запуск:
  $env:MLOPS_TRACKING=1
  python -m mlops.retrain --data data/trivago --stream --nrows 2000000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trivago")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--nrows", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()

    os.environ.setdefault("MLOPS_TRACKING", "1")

    from recsys.data.trivago import load, load_stream
    from recsys.evaluation.baselines import all_baselines
    from recsys.evaluation.protocol import evaluate_all, temporal_split
    from mlops.tracking import log_comparison, register_best

    data = (load_stream(a.data, max_rows=a.nrows) if a.stream else load(a.data, nrows=a.nrows))
    train, test = temporal_split(data.clickouts, test_frac=0.2)

    models = list(all_baselines())
    from recsys.models.ltr_lightgbm import LIGHTGBM_AVAILABLE, LightGBMRanker
    if LIGHTGBM_AVAILABLE:
        params_path = RESULTS / "ltr_best_params.json"
        params = json.loads(params_path.read_text()).get("params") if params_path.exists() else None
        ltr = LightGBMRanker(drop=["pop_log", "pop_impr_log"])   # без популярностного смещения
        if params:
            ltr.params.update(params)
        models.append(ltr)

    rows = evaluate_all(models, train, test, metadata=data.metadata)
    best = rows[0]
    print(f"Лучшая модель: {best['model']} (MRR={best['MRR']:.4f})")

    out = RESULTS / "track3_comparison.json"
    out.write_text(json.dumps({"best_model": best["model"], "best_mrr": best["MRR"], "results": rows},
                              ensure_ascii=False, indent=2))
    log_comparison(rows, artifacts=[str(out)])
    register_best(str(out), name="trivago_ranker")
    print("Переобучение завершено, метрики и модель залогированы в MLflow.")


if __name__ == "__main__":
    main()
