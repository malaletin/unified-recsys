"""
Сквозное сравнение всех моделей Трека 3 на едином протоколе Trivago.

Прогоняет baseline-линейку, LightGBM-LTR, BPR-MF/ALS и последовательные
модели (GRU4Rec, SASRec) на одинаковом temporal split, печатает общую
таблицу метрик (MRR, HitRate@K, NDCG@K, MAP@K) и выбирает лучшую модель.
Модели, чьи зависимости не установлены, аккуратно пропускаются.

Запуск:
    python -m scripts.compare_models --data data/trivago
    python -m scripts.compare_models --data data/trivago --stream --epochs 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recsys.data.trivago import load, load_stream
from recsys.evaluation.baselines import all_baselines
from recsys.evaluation.protocol import evaluate_all, format_table, temporal_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def build_models(epochs: int, only: set | None = None, ltr_params: dict | None = None,
                 ltr_drop: list | None = None):
    models = list(all_baselines())

    from recsys.models.ltr_lightgbm import LIGHTGBM_AVAILABLE, LightGBMRanker
    if LIGHTGBM_AVAILABLE:
        ltr = LightGBMRanker(drop=ltr_drop)
        if ltr_params:
            ltr.params.update(ltr_params)          # подобранные Optuna гиперпараметры
            print(f"[ltr] применены настроенные гиперпараметры: {ltr_params}")
        if ltr_drop:
            print(f"[ltr] исключены признаки: {ltr_drop}")
        models.append(ltr)
    else:
        print("[skip] LightGBM не установлен (pip install lightgbm)")

    from recsys.models.mf_implicit import IMPLICIT_AVAILABLE, ALSModel, BPRModel
    if IMPLICIT_AVAILABLE:
        models += [ALSModel(), BPRModel()]
    else:
        print("[skip] implicit не установлен (pip install implicit)")

    from recsys.models.sequential.ranker import TORCH_AVAILABLE, SequentialRanker
    if TORCH_AVAILABLE:
        models += [SequentialRanker(kind="gru", epochs=epochs),
                   SequentialRanker(kind="sasrec", epochs=epochs)]
    else:
        print("[skip] torch не установлен (см. requirements-gpu.txt)")

    if only:
        models = [m for m in models if getattr(m, "name", type(m).__name__) in only]
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trivago_sample")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--models", nargs="*", default=None,
                    help="ограничить набор моделей по именам (напр. ContentSim LightGBM-LTR SASRec)")
    ap.add_argument("--ltr-params", default=None,
                    help="путь к results/ltr_best_params.json — применить подобранные гиперпараметры LightGBM")
    ap.add_argument("--ltr-drop", nargs="*", default=None,
                    help="исключить признаки из LightGBM (напр. pop_log pop_impr_log)")
    args = ap.parse_args()

    ltr_params = None
    if args.ltr_params and Path(args.ltr_params).exists():
        ltr_params = json.loads(Path(args.ltr_params).read_text()).get("params")

    t0 = time.time()
    print(f"Загрузка {args.data} (stream={args.stream}) ...")
    data = (load_stream(args.data, chunksize=args.chunksize, max_rows=args.nrows)
            if args.stream else load(args.data, nrows=args.nrows))
    train, test = temporal_split(data.clickouts, test_frac=args.test_frac)
    print(f"clickouts={len(data.clickouts)}  train={len(train)}  test={len(test)}  "
          f"items={len(data.metadata.item_ids)}")

    models = build_models(args.epochs, only=set(args.models) if args.models else None,
                          ltr_params=ltr_params, ltr_drop=args.ltr_drop)
    print(f"\nМоделей к сравнению: {[getattr(m, 'name', type(m).__name__) for m in models]}\n")
    rows = evaluate_all(models, train, test, metadata=data.metadata)

    print("\n=== Итоговое сравнение (сортировка по MRR) ===")
    print(format_table(rows, cols=("model", "MRR", "HitRate@5", "NDCG@5", "MAP@5", "HitRate@25", "n")))
    best = rows[0]
    print(f"\nЛучшая модель: {best['model']}  (MRR={best['MRR']:.4f})")

    payload = {"data": str(args.data), "train": len(train), "test": len(test),
               "best_model": best["model"], "best_mrr": best["MRR"],
               "runtime_sec": round(time.time() - t0, 1), "results": rows}
    (RESULTS / "track3_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {RESULTS / 'track3_comparison.json'}  ({payload['runtime_sec']} c)")


if __name__ == "__main__":
    main()
