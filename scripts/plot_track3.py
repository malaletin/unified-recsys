"""
График сравнения моделей Трека 3 по MRR (из results/track3_comparison.json).

Запуск:  python -m scripts.plot_track3
Артефакт: results/track3_mrr.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

# модели, считающиеся baseline (серый цвет); остальные — синий (продвинутые)
BASELINES = {"ImpressionOrder", "Popularity", "PriceAsc", "ItemKNN", "ContentSim"}


def main():
    data = json.loads((RESULTS / "track3_comparison.json").read_text())
    rows = sorted(data["results"], key=lambda r: r["MRR"])
    names = [r["model"] for r in rows]
    mrr = [r["MRR"] for r in rows]
    colors = ["#94a3b8" if n in BASELINES else "#2563eb" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, mrr, color=colors)
    for i, v in enumerate(mrr):
        ax.annotate(f"{v:.4f}", (v, i), va="center", ha="left", fontsize=10)
    ax.set_xlabel("MRR")
    ax.set_xlim(0, max(mrr) * 1.15)
    ax.set_title("Trivago RecSys 2019 — сравнение моделей (Трек 3, MRR)")
    fig.tight_layout()
    out = RESULTS / "track3_mrr.png"
    fig.savefig(out, dpi=130)
    print(f"Лучшая модель: {data.get('best_model')}  (MRR={data.get('best_mrr'):.4f})")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
