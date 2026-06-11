"""
Лёгкое ML-ядро на numpy: StandardScaler, KMeans (k-means++),
TruncatedSVD, косинусное сходство.

При наличии scikit-learn автоматически используются его реализации
(референсный стек). Иначе работают numpy-эквиваленты — это снимает
тяжёлые зависимости и позволяет запускать движок в любом окружении.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler as _SKScaler
    from sklearn.cluster import KMeans as _SKKMeans
    from sklearn.decomposition import TruncatedSVD as _SKSVD
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if SKLEARN_AVAILABLE:
        return _sk_cosine(a, b)
    an = np.linalg.norm(a, axis=1, keepdims=True); an[an == 0] = 1e-12
    bn = np.linalg.norm(b, axis=1, keepdims=True); bn[bn == 0] = 1e-12
    return (a / an) @ (b / bn).T


class StandardScaler:
    def __init__(self):
        self._impl = _SKScaler() if SKLEARN_AVAILABLE else None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self._impl is not None:
            self._impl.fit(X); self.mean_, self.scale_ = self._impl.mean_, self._impl.scale_
            return self
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0); std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self._impl.transform(X) if self._impl is not None else (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KMeans:
    def __init__(self, n_clusters=10, random_state=42, max_iter=300, n_init=10, tol=1e-4):
        self.n_clusters, self.random_state = n_clusters, random_state
        self.max_iter, self.n_init, self.tol = max_iter, n_init, tol
        self.cluster_centers_ = self.labels_ = self.inertia_ = None
        self._impl = _SKKMeans(n_clusters=n_clusters, random_state=random_state,
                               max_iter=max_iter, n_init=n_init, tol=tol) if SKLEARN_AVAILABLE else None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self._impl is not None:
            self._impl.fit(X)
            self.cluster_centers_, self.labels_, self.inertia_ = (
                self._impl.cluster_centers_, self._impl.labels_, self._impl.inertia_)
            return self
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            centers = self._pp_init(X, rng)
            labels = None
            for _ in range(self.max_iter):
                labels = self._assign(X, centers)
                new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                                for j in range(self.n_clusters)])
                if np.linalg.norm(new - centers) <= self.tol:
                    centers = new; break
                centers = new
            inertia = float(sum(np.sum((X[labels == j] - centers[j]) ** 2) for j in range(self.n_clusters)))
            if best is None or inertia < best[0]:
                best = (inertia, centers, labels)
        self.inertia_, self.cluster_centers_, self.labels_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._impl.predict(X) if self._impl is not None else self._assign(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X); return self.labels_

    def _pp_init(self, X, rng):
        n = X.shape[0]; centers = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            p = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
            centers.append(X[rng.choice(n, p=p)])
        return np.array(centers)

    @staticmethod
    def _assign(X, centers):
        return np.argmin(np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2), axis=1)


class TruncatedSVD:
    """Усечённое сингулярное разложение для коллаборативного контура.

    Раскладывает разреженную матрицу взаимодействий R ≈ U Σ Vᵀ и проецирует
    пользователей/объекты в латентное пространство меньшей размерности.
    """

    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self._impl = _SKSVD(n_components=n_components, random_state=random_state) if SKLEARN_AVAILABLE else None
        self.components_ = None         # Vᵀ (n_components x n_items)
        self.singular_values_ = None
        self._item_factors = None       # (n_items x n_components)

    def fit(self, R: np.ndarray) -> "TruncatedSVD":
        R = np.asarray(R, dtype=float)
        k = min(self.n_components, min(R.shape) - 1) if min(R.shape) > 1 else 1
        if self._impl is not None:
            self._impl.n_components = k
            self._impl.fit(R)
            self.components_ = self._impl.components_
            self.singular_values_ = self._impl.singular_values_
        else:
            U, S, Vt = np.linalg.svd(R, full_matrices=False)
            self.components_ = Vt[:k]
            self.singular_values_ = S[:k]
        self._item_factors = self.components_.T          # (n_items x k)
        return self

    def user_factors(self, R: np.ndarray) -> np.ndarray:
        """Проекция строк матрицы взаимодействий в латентное пространство."""
        return np.asarray(R, dtype=float) @ self._item_factors

    def item_factors(self) -> np.ndarray:
        return self._item_factors
