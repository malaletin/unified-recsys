"""Тесты рекомендательного движка (не требуют БД)."""
from __future__ import annotations

import numpy as np

from recsys.dataset import generate_all
from recsys.hybrid import HybridRecommender, PopularityBaseline
from recsys.mlcore import KMeans, StandardScaler, TruncatedSVD, cosine_similarity


def test_mlcore_primitives():
    X = np.random.default_rng(0).normal(size=(40, 5))
    assert abs(StandardScaler().fit_transform(X).mean()) < 1e-6
    km = KMeans(n_clusters=3, random_state=42).fit(X)
    assert km.cluster_centers_.shape == (3, 5)
    svd = TruncatedSVD(n_components=3, random_state=42).fit(np.abs(X))
    assert svd.item_factors().shape[1] == 3
    assert cosine_similarity(X[:2], X[:4]).shape == (2, 4)


def _fit():
    t = generate_all()
    g = set(t["users"].loc[t["users"]["consent_status"] == "granted", "user_id"])
    hist = t["history"][t["history"]["user_id"].isin(g)]
    inter = t["interactions"][t["interactions"]["user_id"].isin(g)]
    rec = HybridRecommender(t["offers"]).fit(hist, inter, t["users"])
    return t, hist, rec


def test_recommend_shape_and_ranking():
    t, hist, rec = _fit()
    pool = t["offers"]["offer_id"].tolist()[:30]
    out = rec.recommend("U0078", pool, top_k=5)
    assert len(out) == 5
    assert out[0]["rank"] == 1
    assert all(out[i]["score"] >= out[i + 1]["score"] for i in range(len(out) - 1))


def test_cold_start_switch():
    t, hist, rec = _fit()
    pool = t["offers"]["offer_id"].tolist()[:20]
    out = rec.recommend(None, pool, questionnaire={"budget": "high", "trip_goal": "luxury"}, top_k=5)
    assert all(r["cold_start"] for r in out)
    assert all(r["score_collab"] == 0.0 for r in out)  # коллаборатив отключён


def test_baseline_runs():
    t, hist, rec = _fit()
    base = PopularityBaseline(hist)
    out = base.recommend("U0078", t["offers"]["offer_id"].tolist()[:20], top_k=5)
    assert len(out) == 5


def test_cluster_purity_reasonable():
    t, hist, rec = _fit()
    assert rec.segmentation.purity(t["users"]) > 0.4
