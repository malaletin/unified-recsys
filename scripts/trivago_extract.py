"""
Стадия 1 батчевой офлайн-оценки: потоковое извлечение clickout-инстансов
из train.csv и сохранение их в компактный pickle.

Разделение на стадии нужно для сред с жёстким лимитом времени/памяти на один
запуск: тяжёлое чтение многогигабайтного CSV выполняется один раз и
кэшируется, после чего оценку можно прогонять многократно дёшево.

Запуск:  python -m scripts.trivago_extract --data data/trivago --nrows 4000000 --out /tmp/clickouts.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from recsys.data.trivago import extract_slice, stream_clickouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="каталог с train.csv")
    ap.add_argument("--train-file", default="train.csv")
    ap.add_argument("--nrows", type=int, default=None, help="число строк среза (None = весь файл)")
    ap.add_argument("--skiprows", type=int, default=0, help="смещение начала среза (для батчей)")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--out", default="/tmp/clickouts.pkl")
    a = ap.parse_args()

    t0 = time.time()
    train_path = Path(a.data) / a.train_file
    if a.skiprows:
        clickouts = extract_slice(train_path, skiprows=a.skiprows, nrows=a.nrows, require_target=True)
    else:
        clickouts = stream_clickouts(train_path, chunksize=a.chunksize,
                                     max_rows=a.nrows, require_target=True)
    with open(a.out, "wb") as f:
        pickle.dump(clickouts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"извлечено clickout-инстансов: {len(clickouts)} -> {a.out}  ({time.time()-t0:.1f} c)")


if __name__ == "__main__":
    main()
