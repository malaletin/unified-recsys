"""
Подбор гиперпараметров LightGBM-LTR через Optuna (метрика — MRR на валидации).

Внутренний temporal split train -> (sub-train, validation); для каждой пробы
обучается LightGBM и измеряется MRR на валидации; возвращается лучшая
конфигурация.

Запуск:  python -m scripts.tune_optuna --data data/trivago --trials 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys.data.trivago import load, load_stream
from recsys.evaluation.metrics import MetricAccumulator
from recsys.evaluation.protocol import temporal_split

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _mrr(model, test):
    acc = MetricAccumulator(ks=(1, 5))
    for inst in test:
        if inst.has_target:
            acc.add(model.rank(inst), inst.target)
    return acc.result().get("MRR", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trivago_sample")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    import optuna
    from recsys.models.ltr_lightgbm import LightGBMRanker

    data = (load_stream(args.data, max_rows=args.nrows) if args.stream
            else load(args.data, nrows=args.nrows))
    train_all, _ = temporal_split(data.clickouts, test_frac=0.2)
    sub_train, valid = temporal_split(train_all, test_frac=0.2)

    def objective(trial):
        params = dict(
            objective="lambdarank", metric="ndcg",
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42, n_jobs=-1,
        )
        model = LightGBMRanker(params=params).fit(sub_train, data.metadata)
        return _mrr(model, valid)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    print(f"\nЛучший MRR (валидация): {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")
    (RESULTS / "ltr_best_params.json").write_text(
        json.dumps({"best_mrr": study.best_value, "params": study.best_params},
                   ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
