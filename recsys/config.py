"""Гиперпараметры рекомендательного движка (не зависят от инфраструктуры)."""
from __future__ import annotations

from dataclasses import dataclass, field

RANDOM_STATE = 42

CITIES = ["Москва", "Санкт-Петербург", "Сочи", "Казань", "Калининград"]
MEAL_PLANS = ["RO", "BB", "HB"]
ROOM_TYPES = ["Standard", "Suite", "Apartment"]
AMENITIES = ["WiFi", "Parking", "Kitchen", "SPA", "Pool", "Pets", "Gym", "BBQ", "KidsZone", "SeaView"]
SEASONS = ["high", "low", "any"]

# веса поведенческих сигналов (шкала намерений)
EVENT_WEIGHTS = {"impression": 1.0, "click": 3.0, "book": 10.0}


@dataclass
class ModelConfig:
    # кластеризация
    n_clusters: int = 10
    # размерность латентного пространства коллаборативного контура (TruncatedSVD)
    svd_components: int = 20
    # веса контуров гибрида: контент / коллаборатив / кластер / контекст
    w_content: float = 0.45
    w_collab: float = 0.30
    w_cluster: float = 0.15
    w_context: float = 0.10
    # порог истории для активации коллаборативного и кластерного контуров
    cold_start_threshold: int = 3
    # период полураспада Time Decay (в днях)
    time_decay_halflife_days: float = 120.0
    top_k: int = 5
    top_n: int = 10
    # сетка подбора веса коллаборативного контура
    w_collab_grid: tuple = (0.0, 0.15, 0.30, 0.45)


@dataclass
class DataConfig:
    n_platforms: int = 4
    n_hotels: int = 15
    n_offers: int = 150
    n_users: int = 1000
    n_history: int = 12_000
    n_interactions: int = 40_000
    consent_revoked_share: float = 0.07
    questionnaire_share: float = 0.40


@dataclass
class SimConfig:
    n_sessions: int = 2000
    scenario_mix: dict = field(default_factory=lambda: {"S1_new": 0.35, "S2_cross": 0.40, "S3_repeat": 0.25})
    pool_size: int = 30
    click_base: float = 0.05
    click_slope: float = 0.85
    book_base: float = 0.10
    book_slope: float = 0.70
    ttb_min_hours: float = 2.0
    ttb_max_hours: float = 26.0
    bootstrap_iters: int = 1000
    confidence: float = 0.95


MODEL = ModelConfig()
DATA = DataConfig()
SIM = SimConfig()
