"""
Офлайн-оценка моделей на датасете Trivago RecSys 2019 (или его сэмпле).

Загружает train.csv + item_metadata.csv, делает temporal split, оценивает
линейку baseline по ранжирующим метрикам (MRR, HitRate@K, NDCG@K, MAP@K,
Novelty) и печатает сравнительную таблицу. Результаты сохраняются в
results/trivago_metrics.json.

Запуск:
    python -m scripts.eval_trivago --data data/trivago_sample
    python -m scripts.eval_trivago --data data/trivago        # реальные данные
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trivago_sample", help="каталог с train.csv и item_metadata.csv")
    ap.add_argument("--nrows", type=int, default=None, help="ограничение строк train (для больших данных)")
    ap.add_argument("--stream", action="store_true", help="потоковое чтение чанками (для многогигабайтных train.csv)")
    ap.add_argument("--chunksize", type=int, default=500_000, help="размер чанка при --stream")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Загрузка данных из {args.data} ... (stream={args.stream})")
    if args.stream:
        data = load_stream(args.data, chunksize=args.chunksize, max_rows=args.nrows, require_target=True)
    else:
        data = load(args.data, nrows=args.nrows, require_target=True)
    print(f"  clickout-инстансов с целью: {len(data.clickouts)}, отелей в метаданных: {len(data.metadata.item_ids)}")

    train, test = temporal_split(data.clickouts, test_frac=args.test_frac)
    print(f"  temporal split: train={len(train)}, test={len(test)}")

    print("\nОценка моделей ...")
    rows = evaluate_all(all_baselines(), train, test, metadata=data.metadata)
    print("\n" + format_table(rows))

    payload = {"data": str(args.data), "n_clickouts": len(data.clickouts),
               "train": len(train), "test": len(test),
               "runtime_sec": round(time.time() - t0, 2), "results": rows}
    (RESULTS / "trivago_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {RESULTS / 'trivago_metrics.json'}  ({payload['runtime_sec']} c)")


if __name__ == "__main__":
    main()
