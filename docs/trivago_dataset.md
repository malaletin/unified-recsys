# Trivago RecSys Challenge 2019 — данные и офлайн-оценка

Трек 1 диплома: строгая офлайн-оценка на реальном датасете отельных сессий
вместо синтетики и имитации.

## Задача и метрика

Датасет описывает действия пользователей в поисковых сессиях. Целевое
действие — `clickout item`: пользователь кликает по одному из показанных
отелей (`impressions`). Требуется переранжировать список `impressions` так,
чтобы реально кликнутый отель (`reference`) был выше. Официальная метрика —
**Mean Reciprocal Rank (MRR)**; дополнительно считаются HitRate@K, NDCG@K,
MAP@K и Novelty.

## Получение данных

Нужны два файла: `train.csv` и `item_metadata.csv` (файл `test.csv` для
офлайн-оценки не используется — в нём метки кликов скрыты, они были для
сабмита; честная оценка делается temporal-split'ом по `train.csv`).

```bash
pip install kaggle
kaggle datasets download -d pranavmahajan725/trivagorecsyschallengedata2019
unzip trivagorecsyschallengedata2019.zip -d data/trivago/
# в каталоге data/trivago/ должны лежать train.csv и item_metadata.csv
```

Официальный источник (с регистрацией): https://recsys.trivago.cloud/challenge/dataset/

## Формат

`train.csv` — лог действий:
`user_id, session_id, timestamp, step, action_type, reference, platform,
city, device, current_filters, impressions, prices`. Поля `impressions` и
`prices` (разделитель `|`) заполнены только в строках `clickout item`.

`item_metadata.csv` — свойства отелей: `item_id, properties` (свойства
через `|`, например `5 Star|WiFi|Spa|Sea View`).

## Запуск оценки

```bash
# синтетический сэмпл в формате Trivago (для проверки конвейера без данных)
python -m scripts.make_trivago_sample --out data/trivago_sample
python -m scripts.eval_trivago --data data/trivago_sample

# реальные данные (при ограниченной памяти используйте --nrows)
python -m scripts.eval_trivago --data data/trivago --nrows 1000000
# на машине с достаточной памятью — полный прогон без --nrows
python -m scripts.eval_trivago --data data/trivago
```

Загрузчик читает только нужные колонки и строит метаданные лишь для
встречающихся отелей, что позволяет работать с многогигабайтным `train.csv`
при ограниченной оперативной памяти.

## Результаты (реальные данные, подвыборка 1 млн строк, temporal split)

Подвыборка: 99 821 clickout-инстанс, train = 79 856, test = 19 965.

| Модель | MRR | HitRate@5 | NDCG@5 | MAP@5 |
|--------|-----|-----------|--------|-------|
| ContentSim (свойства отелей) | 0.512 | 0.624 | 0.515 | 0.479 |
| ImpressionOrder (порядок показа) | 0.459 | 0.612 | 0.470 | 0.423 |
| ItemKNN (со-встречаемость в сессии) | 0.404 | 0.571 | 0.415 | 0.364 |
| Popularity | 0.291 | 0.448 | 0.290 | 0.239 |
| PriceAsc | 0.251 | 0.363 | 0.238 | 0.197 |

Порядок показа площадки — сильный baseline (платформа уже хорошо ранжирует),
и контентная модель его умеренно превосходит. Это согласуется с известными
результатами челленджа и задаёт честную точку отсчёта для продвинутых
моделей Трека 3 (BPR, ALS, нейросетевые и последовательные подходы).

## Структура кода

```
recsys/data/trivago.py            загрузчик и адаптер формата Trivago
recsys/evaluation/metrics.py      MRR, HitRate@K, NDCG@K, MAP@K, Novelty, Diversity
recsys/evaluation/protocol.py     temporal split и цикл оценки
recsys/evaluation/baselines.py    ImpressionOrder, Popularity, PriceAsc, ItemKNN, ContentSim
scripts/make_trivago_sample.py    генератор сэмпла в формате Trivago
scripts/eval_trivago.py           запуск офлайн-оценки
```
