"""
Стадия 2 батчевой офлайн-оценки: загрузка извлечённых clickout-инстансов из
pickle, построение метаданных нужных отелей и оценка линейки моделей.

Запуск:  python -m scripts.trivago_eval_cached --clickouts /tmp/clickouts.pkl --meta data/trivago/item_metadata.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from recsys.data.trivago import ItemMetadata, _needed_items, load_metadata
from recsys.evaluation.baselines import all_baselines
from recsys.evaluation.protocol import evaluate_all, format_table, temporal_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clickouts", nargs="+", default=["/tmp/clickouts.pkl"],
                    help="один или несколько pickle-файлов (батчей)")
    ap.add_argument("--meta", required=True, help="путь к item_metadata.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", default=str(RESULTS / "trivago_metrics.json"))
    a = ap.parse_args()

    t0 = time.time()
    clickouts = []
    for path in a.clickouts:
        with open(path, "rb") as f:
            clickouts.extend(pickle.load(f))
    print(f"загружено clickout-инстансов: {len(clickouts)} из {len(a.clickouts)} батчей")

    metadata = load_metadata(Path(a.meta), needed=_needed_items(clickouts))
    print(f"отелей в метаданных: {len(metadata.item_ids)}")

    train, test = temporal_split(clickouts, test_frac=a.test_frac)
    print(f"temporal split: train={len(train)}, test={len(test)}")

    rows = evaluate_all(all_baselines(), train, test, metadata=metadata)
    print("\n" + format_table(rows))

    payload = {"n_clickouts": len(clickouts), "train": len(train), "test": len(test),
               "runtime_sec": round(time.time() - t0, 2), "results": rows}
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {a.out}  ({payload['runtime_sec']} c)")


if __name__ == "__main__":
    main()
