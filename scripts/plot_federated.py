"""График trade-off «приватность ↔ качество» и сравнение режимов обучения."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"


def plot_tradeoff(rows=None):
    if rows is None:
        rows = json.loads((RESULTS / "track2_federated.json").read_text())["results"]

    # --- сравнение режимов по MRR ---
    labels, mrrs, colors = [], [], []
    for r in rows:
        if r["mode"] == "Centralized":
            labels.append("Centralized"); colors.append("#16a34a")
        elif r["mode"] == "Federated":
            labels.append("Federated"); colors.append("#2563eb")
        else:
            labels.append(f"Fed+DP σ={r['sigma']}"); colors.append("#94a3b8")
        mrrs.append(r["MRR"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, mrrs, color=colors)
    for i, v in enumerate(mrrs):
        ax.annotate(f"{v:.4f}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MRR"); ax.set_title("Режимы обучения: централизованное / федеративное / + DP")
    plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(RESULTS / "track2_modes.png", dpi=130); plt.close(fig)

    # --- кривая trade-off ε ↔ MRR (только DP-точки с числовым ε) ---
    dp = [(r["epsilon"], r["MRR"]) for r in rows
          if r["mode"].startswith("Federated+DP") and isinstance(r["epsilon"], (int, float))]
    if dp:
        dp.sort()
        xs, ys = zip(*dp)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, "o-", color="#2563eb", lw=2)
        fed = next((r["MRR"] for r in rows if r["mode"] == "Federated"), None)
        if fed is not None:
            ax.axhline(fed, ls="--", color="#94a3b8", label="Federated (без DP)")
            ax.legend()
        ax.set_xlabel("Бюджет приватности ε (меньше — приватнее)")
        ax.set_ylabel("MRR")
        ax.set_title("Trade-off «приватность ↔ качество» (DP-SGD)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(RESULTS / "track2_tradeoff.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    plot_tradeoff()
    print("Готово: results/track2_modes.png, results/track2_tradeoff.png")
