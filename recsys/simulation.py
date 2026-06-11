"""Имитационное моделирование пользовательских сессий (Simulation-Based Validation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .dataset import ARCHETYPES, _affinity
from .metrics import compute_metrics, compare


class SessionSimulator:
    def __init__(self, offers, users, questionnaires, history, seed=C.RANDOM_STATE):
        self.offers_idx = offers.set_index("offer_id")
        self.users = users.set_index("user_id")
        self.quest = questionnaires.set_index("user_id").to_dict("index")
        self.hist_counts = history.groupby("user_id").size().to_dict()
        self.offer_ids = offers["offer_id"].tolist()
        self.seed = seed

    def _true_rel(self, archetype, offer_ids):
        arch = ARCHETYPES[archetype]
        raw = np.array([_affinity(arch, self.offers_idx.loc[o]) for o in offer_ids])
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

    def _pick_user(self, rng, scenario):
        uids = self.users.index.to_numpy()
        if scenario == "S1_new":
            cand = [u for u in uids if self.hist_counts.get(u, 0) < C.MODEL.cold_start_threshold]
        elif scenario == "S3_repeat":
            cand = [u for u in uids if self.hist_counts.get(u, 0) >= 8]
        else:
            cand = [u for u in uids if self.hist_counts.get(u, 0) >= C.MODEL.cold_start_threshold]
        cand = cand or list(uids)
        return cand[int(rng.integers(0, len(cand)))]

    def run(self, recommender, label="prototype") -> list[dict]:
        rng = np.random.default_rng(self.seed)
        mix = C.SIM.scenario_mix
        scenarios = list(rng.choice(list(mix), size=C.SIM.n_sessions, p=list(mix.values())))
        sessions = []
        for si, scenario in enumerate(scenarios):
            uid = self._pick_user(rng, scenario)
            arch = self.users.loc[uid, "archetype_truth"]
            consent = self.users.loc[uid, "consent_status"]
            pool = list(rng.choice(self.offer_ids, size=min(C.SIM.pool_size, len(self.offer_ids)), replace=False))
            quest = self.quest.get(uid)
            use_uid = None if (scenario == "S1_new" or consent == "revoked") else uid
            recs = recommender.recommend(use_uid, pool, questionnaire=quest, top_k=C.MODEL.top_k)
            rec_ids = [r["offer_id"] for r in recs]

            rel = dict(zip(pool, self._true_rel(arch, pool)))
            top_rel = np.array([rel[o] for o in rec_ids]) if rec_ids else np.array([0.0])
            pos_w = np.array([1.0 / np.log2(i + 2) for i in range(len(top_rel))])
            wrel = float((top_rel * pos_w).sum() / pos_w.sum())

            p_click = min(0.98, C.SIM.click_base + C.SIM.click_slope * wrel)
            clicked = rng.random() < p_click
            booked, ttb = False, float("nan")
            if clicked:
                best = float(top_rel.max())
                booked = rng.random() < min(0.95, C.SIM.book_base + C.SIM.book_slope * best)
                if booked:
                    ttb = C.SIM.ttb_max_hours - (C.SIM.ttb_max_hours - C.SIM.ttb_min_hours) * wrel
                    ttb = max(C.SIM.ttb_min_hours, float(ttb + rng.normal(0, 1.5)))
            sessions.append(dict(session_id=f"SIM{si:05d}", scenario=scenario, user_id=uid,
                                 ranker=label, clicked=clicked, booked=booked,
                                 time_to_book=ttb, weighted_relevance=wrel))
        return sessions


def run_validation(recommender, baseline, simulator) -> dict:
    proto = simulator.run(recommender, "prototype")
    base = simulator.run(baseline, "baseline")
    cmp = compare(proto, base)
    cmp["by_scenario"] = {s: compute_metrics([x for x in proto if x["scenario"] == s])
                          for s in C.SIM.scenario_mix if any(x["scenario"] == s for x in proto)}
    cmp["_sessions"] = proto
    return cmp
