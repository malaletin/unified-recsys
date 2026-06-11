"""
Прогон имитационной валидации расширенного движка.

Генерирует данные, обучает гибрид (CB + CF/SVD + cluster + context + Time Decay),
подбирает вес коллаборативного контура, прогоняет 2000 сессий и сравнивает с
baseline. Сохраняет results/metrics.json и графики.

Запуск:  python -m scripts.run_simulation
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from recsys import config as C
from recsys.dataset import generate_all
from recsys.hybrid import HybridRecommender, PopularityBaseline
from recsys.metrics import compute_metrics
from recsys.simulation import SessionSimulator, run_validation

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def _safe(o):
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items() if k != "_sessions"}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o


def main():
    t0 = time.time()
    t = generate_all()
    granted = set(t["users"].loc[t["users"]["consent_status"] == "granted", "user_id"])
    hist = t["history"][t["history"]["user_id"].isin(granted)]
    inter = t["interactions"][t["interactions"]["user_id"].isin(granted)]

    sim = SessionSimulator(t["offers"], t["users"], t["questionnaires"], hist)

    # ---- подбор веса коллаборативного контура ----
    print("=== Подбор веса коллаборативного контура (w_collab) ===")
    grid = []
    for wc in C.MODEL.w_collab_grid:
        rest = 1.0 - wc
        weights = dict(content=rest * 0.6, collab=wc, cluster=rest * 0.25, context=rest * 0.15)
        rec = HybridRecommender(t["offers"], weights=weights).fit(hist, inter, t["users"])
        m = compute_metrics(sim.run(rec, f"wc={wc}"))
        grid.append(dict(w_collab=wc, ctr=m["ctr"], cr=m["conversion_rate"], ttb=m["time_to_book"]))
        print(f"  w_collab={wc:.2f} -> CTR={m['ctr']:.4f} CR={m['conversion_rate']:.4f} TTB={m['time_to_book']:.2f}")
    best = max(grid, key=lambda r: r["ctr"])["w_collab"]
    print(f"  Лучший w_collab = {best}")

    # ---- финальная валидация ----
    rest = 1.0 - best
    weights = dict(content=rest * 0.6, collab=best, cluster=rest * 0.25, context=rest * 0.15)
    rec = HybridRecommender(t["offers"], weights=weights).fit(hist, inter, t["users"])
    cmp = run_validation(rec, PopularityBaseline(hist), sim)
    p, b = cmp["prototype"], cmp["baseline"]
    print("\n=== Валидация: расширенный гибрид vs baseline ===")
    print(f"  CTR  {b['ctr']:.4f} -> {p['ctr']:.4f} (+{cmp['ctr_uplift_pct']:.1f}%)")
    print(f"  TTB  {b['time_to_book']:.2f} -> {p['time_to_book']:.2f} ({cmp['ttb_reduction_pct']:.1f}%)")
    print(f"  CR   {b['conversion_rate']:.4f} -> {p['conversion_rate']:.4f} (+{cmp['cr_uplift_pct']:.1f}%)")

    payload = dict(best_w_collab=best, weight_grid=grid,
                   purity=round(float(rec.segmentation.purity(t["users"])), 4),
                   comparison=_safe({k: v for k, v in cmp.items() if k != "_sessions"}),
                   weights=weights, runtime_sec=round(time.time() - t0, 2), n_sessions=C.SIM.n_sessions)
    (RESULTS / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nМетрики сохранены: {RESULTS / 'metrics.json'}")

    try:
        from scripts.make_charts import make_all
        make_all(cmp, grid, rec.segmentation)
        print("Графики сохранены в results/")
    except Exception as e:  # pragma: no cover
        print(f"[viz] {e}")
    print(f"Готово за {payload['runtime_sec']} c")


if __name__ == "__main__":
    main()
