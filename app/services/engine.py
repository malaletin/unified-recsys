"""
Сервис рекомендаций: связывает слой БД и рекомендательный движок.

Хранит обученный HybridRecommender, перестраивает его по данным из БД
(batch-пересчёт) и обслуживает запросы виджета с проверкой согласия,
анонимизацией выдачи и логированием.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from app.config import settings
from app.db.repository import Repository
from recsys.hybrid import HybridRecommender


def anonymize(unified_id: str | None) -> str:
    if not unified_id:
        return "guest"
    return "anon_" + hashlib.sha256(unified_id.encode()).hexdigest()[:12]


class EngineState:
    """Singleton-обёртка над обученным рекомендателем (потокобезопасное обновление)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.recommender: HybridRecommender | None = None
        self.offers_df: pd.DataFrame | None = None
        self.last_fit: datetime | None = None
        self.purity: float | None = None
        from app.services.cache import RecommendationCache
        self.cache = RecommendationCache(ttl_hours=settings.cache_ttl_hours)

    # ----- batch-пересчёт ----- #
    def fit_from_repo(self, repo: Repository) -> dict:
        t0 = time.time()
        offers = repo.list_offers_df()
        history = repo.history_df(granted_only=True)
        interactions = repo.interactions_df(granted_only=True)
        users = repo.users_df()
        rec = HybridRecommender(offers)
        rec.fit(history, interactions, users)

        # сохраняем производные профили в БД
        for uid, cid in rec.segmentation.user_to_cluster.items():
            repo.upsert_profile(uid, cluster_id=int(cid),
                                cluster_label=rec.segmentation.cluster_labels.get(cid),
                                history_len=int(history[history["user_id"] == uid].shape[0]))
        with self._lock:
            self.recommender = rec
            self.offers_df = offers
            self.last_fit = datetime.now(timezone.utc)
            self.purity = float(rec.segmentation.purity(users)) if "archetype_truth" in users else None
            self.cache.clear()
        return dict(duration_sec=round(time.time() - t0, 3),
                    users=len(users), offers=len(offers),
                    segments=rec.segmentation.cluster_labels)

    # ----- выдача ----- #
    def recommend(self, repo: Repository, *, unified_id, platform_id, available_offers,
                  questionnaire=None, check_in=None, guests=2, top_k=None) -> dict:
        if self.recommender is None:
            self.fit_from_repo(repo)
        top_k = top_k or settings.top_k

        from app.observability import observe_recommend, set_cache_hit_rate

        # middleware: проверка согласия -> при отзыве переходим в Cold Start
        consent = repo.consent_status(unified_id) if unified_id else "guest"
        effective_uid = unified_id if (unified_id and consent == "granted") else None

        # ключ кэша зависит от профиля, пула офферов и контекста запроса
        key = f"{effective_uid or 'guest'}:{hash(tuple(sorted(available_offers)))}:{check_in}:{guests}:{top_k}"
        cached = self.cache.get(key)
        if cached is not None:
            set_cache_hit_rate(self.cache.stats()["hit_rate"])
            return cached

        pool = self._prefilter(repo, available_offers, guests)
        recs = self.recommender.recommend(effective_uid, pool, questionnaire=questionnaire,
                                          check_in=check_in, top_k=top_k)
        cold = bool(recs[0]["cold_start"]) if recs else True
        repo.log_recommendation(unified_id, platform_id,
                                offers=[r["offer_id"] for r in recs], cold_start=cold,
                                context={"check_in": check_in, "guests": guests})
        payload = dict(profile_ref=anonymize(unified_id), consent=consent, cold_start=cold,
                       recommendations=[{"offer_id": r["offer_id"], "rank": r["rank"],
                                         "score": r["score"], "cluster_label": r["cluster_label"]}
                                        for r in recs])
        self.cache.set(key, payload)
        observe_recommend(cold, len(recs))
        set_cache_hit_rate(self.cache.stats()["hit_rate"])
        return payload

    def _prefilter(self, repo: Repository, offers: list[str], guests: int) -> list[str]:
        meta = {o.offer_id: o.capacity for o in repo.offers_by_ids(offers)}
        keep = [o for o in offers if meta.get(o, 99) >= guests]
        return keep or offers


ENGINE = EngineState()
