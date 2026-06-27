"""
Ablation-исследование признаков LightGBM-LTR.

Поочерёдно исключает группы признаков и измеряет падение MRR на тесте —
показывает вклад каждой группы (цена, популярность, контент, сигналы сессии,
позиция). Также печатает важности признаков полной модели.

Запуск:  python -m scripts.ablation_ltr --data data/trivago
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recsys.data.trivago import load, load_stream
from recsys.evaluation.metrics import MetricAccumulator
from recsys.evaluation.protocol import temporal_split
from recsys.models.features import FEATURE_NAMES

RESULTS = Path(__file__).resolve().parent.parent / "results"

GROUPS = {
    "position": ["position"],
    "price": ["price", "price_rank", "price_rel_mean", "price_is_min"],
    "popularity": ["pop_log", "pop_impr_log"],
    "session": ["sess_interactions", "sess_seen"],
    "content": ["content_sim", "stars", "n_props", "rating_prop"],
}


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
    args = ap.parse_args()

    from recsys.models.ltr_lightgbm import LightGBMRanker

    data = (load_stream(args.data, max_rows=args.nrows) if args.stream
            else load(args.data, nrows=args.nrows))
    train, test = temporal_split(data.clickouts, test_frac=0.2)

    full = LightGBMRanker().fit(train, data.metadata)
    base_mrr = _mrr(full, test)
    print(f"Полная модель: MRR={base_mrr:.4f}")
    print("Важности признаков:", json.dumps(full.feature_importance(), ensure_ascii=False))

    rows = [{"variant": "full", "MRR": base_mrr, "delta": 0.0}]
    for gname, gfeats in GROUPS.items():
        keep = [f for f in FEATURE_NAMES if f not in gfeats]
        keep_idx = [FEATURE_NAMES.index(f) for f in keep]

        class _Masked(LightGBMRanker):
            def fit(self, clickouts, metadata):
                from recsys.models.features import FeatureBuilder
                self.fb = FeatureBuilder(metadata).fit(clickouts)
                X, y, groups = self.fb.build_training(clickouts)
                from lightgbm import LGBMRanker
                self.model = LGBMRanker(**self.params)
                self.model.fit(X[:, keep_idx], y, group=groups)
                self._keep_idx = keep_idx
                return self

            def rank(self, inst):
                from recsys.models.base import stable_order
                feats = self.fb.transform_instance(inst)[:, keep_idx]
                return stable_order(inst.impressions, self.model.predict(feats))

        m = _Masked().fit(train, data.metadata)
        mrr = _mrr(m, test)
        rows.append({"variant": f"-{gname}", "MRR": mrr, "delta": round(mrr - base_mrr, 4)})
        print(f"  без группы '{gname}': MRR={mrr:.4f}  (Δ={mrr - base_mrr:+.4f})")

    (RESULTS / "ltr_ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {RESULTS / 'ltr_ablation.json'}")


if __name__ == "__main__":
    main()
