"""Сегментация пользователей (K-Means, k=10) с маппингом центроидов на бизнес-сегменты."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import BEHAVIORAL_COLS
from .mlcore import KMeans, StandardScaler


class UserSegmentation:
    def __init__(self, n_clusters=C.MODEL.n_clusters, random_state=C.RANDOM_STATE):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_labels: dict[int, str] = {}
        self.user_to_cluster: dict[str, int] = {}
        self.centroids: pd.DataFrame | None = None

    def fit(self, behavioral: pd.DataFrame) -> "UserSegmentation":
        active = behavioral[behavioral["history_len"] > 0].copy()
        Xs = self.scaler.fit_transform(active[BEHAVIORAL_COLS].to_numpy(dtype=float))
        active["cluster_id"] = self.kmeans.fit_predict(Xs)
        self.user_to_cluster = dict(zip(active["user_id"], active["cluster_id"]))
        self.centroids = active.groupby("cluster_id")[BEHAVIORAL_COLS].mean().reindex(range(self.n_clusters))
        self.cluster_labels = self._map(self.centroids)
        return self

    def predict(self, row: dict) -> int:
        X = np.array([[row.get(c, 0.0) for c in BEHAVIORAL_COLS]], dtype=float)
        return int(self.kmeans.predict(self.scaler.transform(X))[0])

    def segment_of(self, user_id: str):
        cid = self.user_to_cluster.get(user_id)
        return self.cluster_labels.get(cid) if cid is not None else None

    @staticmethod
    def _map(centroids):
        labels = {}
        for cid, c in centroids.iterrows():
            if c.isna().all(): labels[cid] = "Undefined"; continue
            if c["business_share"] >= 0.6 and c["avg_nights"] <= 3: seg = "Business_Express"
            elif c["avg_nights"] >= 6 and c["kitchen_share"] >= 0.3: seg = "Family_Vacation"
            elif c["avg_price"] >= 12000 and c["spa_share"] >= 0.25: seg = "Luxury_Retreat"
            elif c["seaview_share"] >= 0.3 or (c["spa_share"] >= 0.3 and c["avg_nights"] <= 4): seg = "Couple_Romantic"
            elif c["pool_share"] >= 0.3 and c["avg_guests"] >= 3: seg = "Group_Active"
            elif c["avg_price"] <= 5000: seg = "Solo_Budget"
            else: seg = "General_Leisure"
            labels[cid] = seg
        return labels

    def cluster_offer_affinity(self, history: pd.DataFrame) -> dict:
        h = history.copy()
        h["cluster_id"] = h["user_id"].map(self.user_to_cluster)
        h = h.dropna(subset=["cluster_id"])
        aff = {}
        for cid, g in h.groupby("cluster_id"):
            counts = g["offer_id"].value_counts()
            aff[int(cid)] = (counts / counts.max()).to_dict() if counts.max() > 0 else {}
        return aff

    def purity(self, users: pd.DataFrame) -> float:
        truth = users.set_index("user_id")["archetype_truth"].to_dict()
        df = pd.DataFrame({"cluster_id": list(self.user_to_cluster.values()),
                           "truth": [truth.get(u) for u in self.user_to_cluster]})
        if len(df) == 0: return 0.0
        return sum(g["truth"].value_counts().iloc[0] for _, g in df.groupby("cluster_id")) / len(df)
