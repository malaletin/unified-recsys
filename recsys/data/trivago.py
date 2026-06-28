"""
Загрузчик и адаптер датасета Trivago RecSys Challenge 2019.

Формат исходных данных:
  train.csv / test.csv — действия пользователей в сессиях:
    user_id, session_id, timestamp, step, action_type, reference,
    platform, city, device, current_filters, impressions, prices
  item_metadata.csv — свойства отелей:
    item_id, properties  (свойства разделены символом '|')

Задача челленджа — для каждого действия `clickout item` переранжировать
список показанных отелей (`impressions`) так, чтобы реально кликнутый отель
(`reference`) оказался выше. Метрика — Mean Reciprocal Rank (MRR).

Загрузчик извлекает из сессий «инстансы clickout» с контекстом
(предшествующие взаимодействия в сессии) и строит признаковое описание
отелей по полю properties.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CLICKOUT = "clickout item"
# типы взаимодействий с конкретным отелем (reference = item_id)
ITEM_INTERACTIONS = {
    "clickout item", "interaction item rating", "interaction item info",
    "interaction item image", "interaction item deals", "search for item",
}


@dataclass
class ClickoutInstance:
    """Один инстанс clickout: показанные отели, цены, контекст и цель."""
    user_id: str
    session_id: str
    timestamp: int
    step: int
    impressions: list[str]          # показанные item_id
    prices: list[float]             # цены в том же порядке
    target: str | None              # реально кликнутый item_id (reference)
    prior_items: list[str] = field(default_factory=list)   # предыдущие взаимодействия в сессии
    platform: str = ""
    city: str = ""
    device: str = ""
    filters: list[str] = field(default_factory=list)

    @property
    def has_target(self) -> bool:
        return self.target is not None and self.target in self.impressions


def _split_pipe(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return []
    return [v for v in str(value).split("|") if v != ""]


# --------------------------------------------------------------------------- #
#  Свойства отелей (item metadata)                                            #
# --------------------------------------------------------------------------- #
class ItemMetadata:
    """Бинарное признаковое пространство отелей по свойствам (properties)."""

    def __init__(self):
        self.item_ids: list[str] = []
        self.properties: list[str] = []
        self._prop_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self.matrix: np.ndarray | None = None
        self.item_props: dict[str, set[str]] = {}

    def fit(self, df: pd.DataFrame, min_freq: int = 1) -> "ItemMetadata":
        # локальный словарь свойств — не храним в атрибуте ради экономии памяти
        item_props = {str(r.item_id): _split_pipe(r.properties) for r in df.itertuples()}

        from collections import Counter
        freq = Counter(p for props in item_props.values() for p in props)
        self.properties = sorted(p for p, c in freq.items() if c >= min_freq)
        self._prop_index = {p: i for i, p in enumerate(self.properties)}

        self.item_ids = list(item_props.keys())
        self._item_index = {it: i for i, it in enumerate(self.item_ids)}
        # бинарная матрица int8 (в ~4 раза легче float32; апкаст происходит при счёте сходства)
        mat = np.zeros((len(self.item_ids), len(self.properties)), dtype=np.int8)
        for it, props in item_props.items():
            row = self._item_index[it]
            for p in props:
                j = self._prop_index.get(p)
                if j is not None:
                    mat[row, j] = 1
        self.matrix = mat
        self.item_props = {}            # не удерживаем тяжёлый словарь
        return self

    def vector(self, item_id: str) -> np.ndarray:
        idx = self._item_index.get(item_id)
        if idx is None:
            return np.zeros(len(self.properties), dtype=np.float32)
        return self.matrix[idx]

    def has(self, item_id: str) -> bool:
        return item_id in self._item_index


# --------------------------------------------------------------------------- #
#  Извлечение clickout-инстансов из лога действий                             #
# --------------------------------------------------------------------------- #
def extract_clickouts(actions: pd.DataFrame, require_target: bool = True) -> list[ClickoutInstance]:
    """Проходит сессии и собирает инстансы clickout с контекстом."""
    actions = actions.sort_values(["session_id", "timestamp", "step"], kind="stable")
    instances: list[ClickoutInstance] = []

    for sid, sess in actions.groupby("session_id", sort=False):
        prior: list[str] = []
        for row in sess.itertuples():
            atype = row.action_type
            ref = None if (isinstance(row.reference, float) and np.isnan(row.reference)) else str(row.reference)
            if atype == CLICKOUT:
                imps = [sys.intern(x) for x in _split_pipe(row.impressions)]
                prices = [float(x) for x in _split_pipe(row.prices)]
                if imps:
                    target = sys.intern(ref) if (ref in imps) else None
                    if (not require_target) or (target is not None):
                        instances.append(ClickoutInstance(
                            user_id=str(row.user_id), session_id=str(sid),
                            timestamp=int(row.timestamp), step=int(row.step),
                            impressions=imps, prices=prices, target=target,
                            prior_items=list(prior),
                            platform=str(getattr(row, "platform", "")),
                            city=str(getattr(row, "city", "")),
                            device=str(getattr(row, "device", "")),
                            filters=_split_pipe(getattr(row, "current_filters", "")),
                        ))
            # накапливаем контекст сессии (взаимодействия с конкретными отелями)
            if atype in ITEM_INTERACTIONS and ref is not None and not ref.isspace():
                if ref.isdigit():
                    prior.append(sys.intern(ref))
    return instances


def stream_clickouts(train_path: Path, chunksize: int = 500_000,
                     max_rows: int | None = None, require_target: bool = True) -> list[ClickoutInstance]:
    """
    Потоковое извлечение clickout-инстансов из больших train.csv.

    Читает файл чанками, накапливая только компактные инстансы. Сессии в
    Trivago идут непрерывными блоками, поэтому на границе чанка достаточно
    «придержать» строки последней (возможно, незавершённой) сессии и
    дочитать их вместе со следующим чанком.
    """
    instances: list[ClickoutInstance] = []
    carry: pd.DataFrame | None = None
    rows_read = 0
    for chunk in pd.read_csv(train_path, usecols=USECOLS, chunksize=chunksize):
        rows_read += len(chunk)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        last_sid = chunk["session_id"].iloc[-1]
        is_last = chunk["session_id"].to_numpy() == last_sid
        carry = chunk[is_last]                               # хвостовая сессия -> в следующий чанк
        complete = chunk[~is_last]
        if len(complete):
            instances.extend(extract_clickouts(complete, require_target=require_target))
        if max_rows is not None and rows_read >= max_rows:
            break
    if carry is not None and len(carry):
        instances.extend(extract_clickouts(carry, require_target=require_target))
    return instances


# --------------------------------------------------------------------------- #
#  Точка входа загрузки                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class TrivagoData:
    metadata: ItemMetadata
    clickouts: list[ClickoutInstance]


# колонки, реально нужные конвейеру (экономит память на больших данных);
# platform нужен для федеративного партиционирования (Трек 2)
USECOLS = ["user_id", "session_id", "timestamp", "step", "action_type",
           "reference", "platform", "impressions", "prices"]
# индексы тех же колонок в исходном 12-колоночном train.csv (для чтения без заголовка)
USECOLS_IDX = [0, 1, 2, 3, 4, 5, 6, 10, 11]


def extract_slice(train_path: Path, skiprows: int = 0, nrows: int | None = None,
                  require_target: bool = True) -> list[ClickoutInstance]:
    """
    Извлекает clickout-инстансы из среза строк [skiprows, skiprows+nrows).

    Чтение по смещению (skiprows) выполняется на уровне C и быстрое, что
    позволяет обрабатывать большой файл порциями в нескольких запусках.
    Срез независим: целостность теряется лишь для пограничной сессии (1 шт.).
    """
    if skiprows and skiprows > 0:
        df = pd.read_csv(train_path, skiprows=skiprows + 1, nrows=nrows,
                         header=None, usecols=USECOLS_IDX, names=USECOLS)
    else:
        df = pd.read_csv(train_path, nrows=nrows, usecols=USECOLS)
    return extract_clickouts(df, require_target=require_target)


def _needed_items(clickouts) -> set[str]:
    need: set[str] = set()
    for c in clickouts:
        need.update(c.impressions)
        need.update(p for p in c.prior_items if p.isdigit())
        if c.target:
            need.add(c.target)
    return need


def load_metadata(meta_path: Path, needed: set[str] | None = None,
                  chunksize: int = 200_000) -> ItemMetadata:
    """Чтение item_metadata.csv (при необходимости — только нужные отели, чанками)."""
    if needed is None:
        return ItemMetadata().fit(pd.read_csv(meta_path))
    keep = []
    for chunk in pd.read_csv(meta_path, chunksize=chunksize):
        chunk["item_id"] = chunk["item_id"].astype(str)
        keep.append(chunk[chunk["item_id"].isin(needed)])
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=["item_id", "properties"])
    return ItemMetadata().fit(df)


def load(data_dir: str | Path, train_file: str = "train.csv",
         meta_file: str = "item_metadata.csv", nrows: int | None = None,
         require_target: bool = True, restrict_metadata: bool = True) -> TrivagoData:
    """
    Загружает train + item_metadata и извлекает clickout-инстансы.

    Память: читаются только нужные колонки; при restrict_metadata=True
    метаданные строятся лишь для отелей, встречающихся в выборке.
    """
    data_dir = Path(data_dir)
    actions = pd.read_csv(data_dir / train_file, nrows=nrows, usecols=USECOLS)
    clickouts = extract_clickouts(actions, require_target=require_target)
    del actions                                       # освобождаем память до чтения метаданных
    needed = _needed_items(clickouts) if restrict_metadata else None
    metadata = load_metadata(data_dir / meta_file, needed=needed)
    return TrivagoData(metadata=metadata, clickouts=clickouts)


def load_stream(data_dir: str | Path, train_file: str = "train.csv",
                meta_file: str = "item_metadata.csv", chunksize: int = 500_000,
                max_rows: int | None = None, require_target: bool = True) -> TrivagoData:
    """Потоковая загрузка больших данных: чанковое чтение train + метаданные нужных отелей."""
    data_dir = Path(data_dir)
    clickouts = stream_clickouts(data_dir / train_file, chunksize=chunksize,
                                 max_rows=max_rows, require_target=require_target)
    needed = _needed_items(clickouts)
    metadata = load_metadata(data_dir / meta_file, needed=needed)
    return TrivagoData(metadata=metadata, clickouts=clickouts)
