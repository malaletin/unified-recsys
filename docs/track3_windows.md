# Трек 3 — продвинутые модели (Windows 11 + RTX 4070 Super)

Сравнение моделей ранжирования на датасете Trivago RecSys 2019: baseline,
**LightGBM Learning-to-Rank**, **BPR-MF / ALS** и последовательные нейросети
**GRU4Rec / SASRec** — на едином офлайн-протоколе (temporal split, метрики
MRR / HitRate@K / NDCG@K / MAP@K), с выбором лучшей модели, ablation и Optuna.

Распределение нагрузки: SASRec/GRU4Rec — на GPU (CUDA), LightGBM и
implicit (BPR/ALS) — на CPU (Ryzen 7700, многопоточно).

## Установка (PowerShell)

```powershell
cd "$HOME\Claude\Projects\Diploma work\unified-recsys"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch с поддержкой CUDA 12.4 (под RTX 4070 Super)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Остальные зависимости Трека 3
pip install -r requirements-gpu.txt
```

Проверка, что GPU виден PyTorch:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидаемый вывод: `CUDA: True NVIDIA GeForce RTX 4070 SUPER`.

## Данные

Положи `train.csv` и `item_metadata.csv` в `data\trivago\` (или укажи путь
в `--data`). См. `docs/trivago_dataset.md`.

## Запуск

Сначала быстрая проверка на синтетическом сэмпле (без данных):

```powershell
python -m scripts.make_trivago_sample --out data\trivago_sample
python -m scripts.compare_models --data data\trivago_sample --epochs 3
```

Полное сравнение всех моделей на реальных данных (32 ГБ RAM хватает на полный
прогон без батчей):

```powershell
python -m scripts.compare_models --data data\trivago --epochs 10
```

Подбор гиперпараметров LightGBM (Optuna) и ablation признаков:

```powershell
python -m scripts.tune_optuna  --data data\trivago --trials 30
python -m scripts.ablation_ltr --data data\trivago
```

Результаты сохраняются в `results\track3_comparison.json`,
`results\ltr_best_params.json`, `results\ltr_ablation.json`.

## Состав моделей

| Модель | Тип | Устройство | Файл |
|--------|-----|-----------|------|
| ImpressionOrder / Popularity / PriceAsc / ItemKNN / ContentSim | baseline | CPU | `recsys/evaluation/baselines.py` |
| LightGBM-LTR (LambdaMART) | gradient boosting | CPU | `recsys/models/ltr_lightgbm.py` |
| BPR-MF, ALS | matrix factorization | CPU | `recsys/models/mf_implicit.py` |
| GRU4Rec, SASRec | sequential (neural) | GPU | `recsys/models/sequential/` |

Все модели реализуют единый интерфейс `fit / rank` (`recsys/models/base.py`)
и сравниваются одним протоколом `recsys/evaluation/protocol.py`.

## Поэтапный план запуска (рекомендуемый порядок)

0. Подготовка окружения (один раз) — см. раздел «Установка» выше.
1. **Дымовая проверка на сэмпле** (1–2 мин, проверяет, что всё ставится и
   запускается, GPU виден):
   ```powershell
   python -m scripts.make_trivago_sample --out data\trivago_sample
   python -m scripts.compare_models --data data\trivago_sample --epochs 3
   ```
2. **Пробный прогон на подвыборке реальных данных** (быстрее полного, чтобы
   убедиться, что данные читаются и модели обучаются):
   ```powershell
   python -m scripts.compare_models --data data\trivago --stream --nrows 3000000 --epochs 5
   ```
3. **Полный прогон сравнения всех моделей** (32 ГБ RAM хватает без батчей):
   ```powershell
   python -m scripts.compare_models --data data\trivago --epochs 10
   python -m scripts.plot_track3
   ```
4. **Ablation и подбор гиперпараметров**:
   ```powershell
   python -m scripts.ablation_ltr --data data\trivago
   python -m scripts.tune_optuna --data data\trivago --trials 30
   ```
5. **Один командой** (шаги 3–4 сразу): активируйте `.venv` и запустите
   оркестратор:
   ```powershell
   .\run_track3.ps1 -Data data\trivago -Epochs 10 -Trials 30
   ```

Запуск отдельной модели или подмножества (если хотите гонять по частям):
```powershell
python -m scripts.compare_models --data data\trivago --models ContentSim LightGBM-LTR SASRec --epochs 10
```
Имена моделей: `ImpressionOrder Popularity PriceAsc ItemKNN ContentSim
LightGBM-LTR ALS BPR-MF GRU4Rec SASRec`.

## Возможные проблемы и решения

- **`torch.cuda.is_available()` = False.** Установлена CPU-сборка torch.
  Переустановите: `pip uninstall torch -y; pip install torch --index-url
  https://download.pytorch.org/whl/cu124`. Проверьте, что драйвер NVIDIA
  свежий (`nvidia-smi`).
- **Ошибка установки `implicit`.** Нужен свежий pip (`python -m pip install
  --upgrade pip`); ставьте `pip install implicit` (есть готовые Windows-колёса).
  Возможно предупреждение про OpenMP — на работу не влияет.
- **Долгий инференс последовательных моделей** на полном тесте (~317k
  инстансов, по одному forward на инстанс). Это нормально; для ускорения
  можно сначала прогнать на `--nrows`, а полный — оставить на ночь.
- **Нехватка памяти при полном `load()`.** Используйте `--stream` (чтение
  чанками) и при необходимости `--nrows`.
- **PowerShell не запускает `.ps1`** (политика выполнения). Один раз:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Артефакты для отчёта/диплома

После полного прогона в `results\` появятся: `track3_comparison.json`
(таблица всех моделей + лучшая), `track3_mrr.png` (график сравнения),
`ltr_ablation.json` (вклад групп признаков), `ltr_best_params.json`
(лучшие гиперпараметры LightGBM). Пришлите таблицу из
`track3_comparison.json` — соберём по ней главу ВКР со сравнением.

## Примечания

- Если какая-то библиотека не установлена, `compare_models` пропустит
  соответствующие модели и сравнит остальные — пайплайн не упадёт.
- Инференс последовательных моделей скорит только показанные отели
  (impressions), поэтому быстр независимо от размера каталога (~900k отелей).
- Для воспроизводимости зафиксирован `random_state=42` / `torch.manual_seed`.
