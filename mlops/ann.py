"""
Масштабируемый поиск ближайших соседей (ANN) для контентного контура.

При большом каталоге (сотни тысяч отелей) полный перебор косинусных сходств
дорог. Здесь строится индекс приближённого поиска: при наличии FAISS
используется он (IndexFlatIP по нормализованным векторам ≈ косинус), иначе —
корректный numpy-fallback полным перебором. Интерфейс одинаков, поэтому код
системы не зависит от наличия FAISS.
"""
from __future__ import annotations

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    FAISS_AVAILABLE = False


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1e-12
    return x / n


class ANNIndex:
    """Индекс приближённого поиска по косинусной близости."""

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = list(ids)
        self._id_pos = {i: p for p, i in enumerate(self.ids)}
        self.matrix = _normalize(vectors)
        self.dim = self.matrix.shape[1]
        self._faiss = None
        if FAISS_AVAILABLE and len(self.ids):
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self.matrix)

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, query: np.ndarray, k: int = 10):
        """Топ-k ближайших: возвращает (списки id, списки сходств)."""
        q = _normalize(np.atleast_2d(query))
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, min(k, len(self.ids)))
            ids = [[self.ids[j] for j in row if j >= 0] for row in idx]
            return ids, sims
        sims_all = q @ self.matrix.T
        idx = np.argsort(-sims_all, axis=1)[:, :k]
        ids = [[self.ids[j] for j in row] for row in idx]
        sims = np.take_along_axis(sims_all, idx, axis=1)
        return ids, sims

    def rank_subset(self, query: np.ndarray, candidate_ids: list[str]) -> np.ndarray:
        """Скоринг косинусной близости запроса к заданному подмножеству id."""
        q = _normalize(np.atleast_2d(query)).ravel()
        pos = [self._id_pos[c] for c in candidate_ids if c in self._id_pos]
        out = np.full(len(candidate_ids), -1.0, dtype=np.float32)
        if not pos:
            return out
        sims = self.matrix[pos] @ q
        j = 0
        for k_, c in enumerate(candidate_ids):
            if c in self._id_pos:
                out[k_] = sims[j]; j += 1
        return out
