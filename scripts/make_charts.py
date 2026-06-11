"""Визуализация результатов валидации (matplotlib)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
BLUE, GREEN, GRAY = "#2563eb", "#16a34a", "#94a3b8"


def _save(fig, name):
    fig.tight_layout(); fig.savefig(RESULTS / name, dpi=130, bbox_inches="tight"); plt.close(fig)


def make_all(cmp, grid, segmentation):
    _comparison(cmp); _weight_grid(grid); _segments(segmentation); _by_scenario(cmp)


def _comparison(cmp):
    p, b = cmp["prototype"], cmp["baseline"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(["Baseline", "Гибрид"], [b["ctr"] * 100, p["ctr"] * 100], color=[GRAY, BLUE])
    ax[0].set_title(f"CTR, %  (+{cmp['ctr_uplift_pct']:.1f}%)")
    ax[1].bar(["Baseline", "Гибрид"], [b["time_to_book"], p["time_to_book"]], color=[GRAY, GREEN])
    ax[1].set_title(f"Time-to-Book, ч  ({cmp['ttb_reduction_pct']:.1f}%)")
    ax[2].bar(["Baseline", "Гибрид"], [b["conversion_rate"] * 100, p["conversion_rate"] * 100], color=[GRAY, BLUE])
    ax[2].set_title(f"Conversion Rate, %  (+{cmp['cr_uplift_pct']:.1f}%)")
    for a in ax:
        for patch in a.patches:
            a.annotate(f"{patch.get_height():.2f}", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                       ha="center", va="bottom", fontsize=9)
    fig.suptitle("Расширенный гибрид vs baseline (популярность)", fontweight="bold")
    _save(fig, "metrics_comparison.png")


def _weight_grid(grid):
    xs = [g["w_collab"] for g in grid]; ys = [g["ctr"] * 100 for g in grid]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, "o-", color=BLUE, lw=2)
    best = max(grid, key=lambda r: r["ctr"])
    ax.scatter([best["w_collab"]], [best["ctr"] * 100], color=GREEN, s=120, zorder=5,
               label=f"w* = {best['w_collab']}")
    ax.set_xlabel("Вес коллаборативного контура (w_collab)"); ax.set_ylabel("CTR, %")
    ax.set_title("Подбор веса коллаборативного контура (SVD)"); ax.grid(alpha=0.3); ax.legend()
    _save(fig, "weight_tuning.png")


def _segments(seg):
    counts = Counter(seg.cluster_labels[c] for c in seg.user_to_cluster.values())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(list(counts), list(counts.values()), color=BLUE)
    ax.set_xlabel("Число пользователей"); ax.set_title("Бизнес-сегменты (K-Means, k=10)")
    for i, v in enumerate(counts.values()):
        ax.annotate(str(v), (v, i), va="center", ha="left", fontsize=9)
    _save(fig, "segments.png")


def _by_scenario(cmp):
    by = cmp.get("by_scenario", {})
    if not by: return
    nm = {"S1_new": "S1 New", "S2_cross": "S2 Cross", "S3_repeat": "S3 Repeat"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([nm.get(k, k) for k in by], [by[k]["ctr"] * 100 for k in by], color=[BLUE, GREEN, GRAY])
    ax.set_ylabel("CTR, %"); ax.set_title("CTR по сценариям (прототип)")
    for i, k in enumerate(by):
        ax.annotate(f"{by[k]['ctr']*100:.1f}", (i, by[k]["ctr"] * 100), ha="center", va="bottom", fontsize=9)
    _save(fig, "ctr_by_scenario.png")
