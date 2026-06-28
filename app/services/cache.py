"""
Кэш рекомендаций с TTL: бэкенд Redis или in-memory fallback.

В продакшене готовые выдачи кэшируются в Redis (общий для всех инстансов
API). Если Redis недоступен или библиотека не установлена, автоматически
используется локальный in-memory словарь — система остаётся работоспособной.
Интерфейс совпадает с прежним кэшем (get/set/invalidate_user/clear) плюс
учёт hit/miss для мониторинга.
"""
from __future__ import annotations

import json
import os
import time

try:
    import redis
    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    REDIS_AVAILABLE = False

REDIS_URL = os.environ.get("REDIS_URL", "")


class RecommendationCache:
    def __init__(self, ttl_hours: int = 12, prefix: str = "rec:"):
        self.ttl = ttl_hours * 3600
        self.prefix = prefix
        self.hits = 0
        self.misses = 0
        self._redis = None
        self._mem: dict[str, tuple[float, list]] = {}
        if REDIS_AVAILABLE and REDIS_URL:
            try:
                self._redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def get(self, key: str):
        k = self.prefix + key
        if self._redis is not None:
            v = self._redis.get(k)
            if v is not None:
                self.hits += 1
                return json.loads(v)
            self.misses += 1
            return None
        entry = self._mem.get(k)
        if entry and entry[0] > time.time():
            self.hits += 1
            return entry[1]
        self.misses += 1
        return None

    def set(self, key: str, payload):
        k = self.prefix + key
        if self._redis is not None:
            self._redis.setex(k, self.ttl, json.dumps(payload, ensure_ascii=False))
        else:
            self._mem[k] = (time.time() + self.ttl, payload)

    def invalidate_user(self, user_id: str):
        if self._redis is not None:
            for k in self._redis.scan_iter(match=f"{self.prefix}{user_id}:*"):
                self._redis.delete(k)
        else:
            for k in [k for k in self._mem if k.startswith(f"{self.prefix}{user_id}:")]:
                self._mem.pop(k, None)

    def clear(self):
        if self._redis is not None:
            for k in self._redis.scan_iter(match=f"{self.prefix}*"):
                self._redis.delete(k)
        else:
            self._mem.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"backend": self.backend, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}
