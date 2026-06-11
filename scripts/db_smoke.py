"""
Smoke-проверка схемы БД и сквозного потока на stdlib sqlite3.

Не требует SQLAlchemy/PostgreSQL — создаёт эквивалентную схему в SQLite,
загружает синтетические данные, агрегирует кросс-платформенную историю по
unified_id, строит рекомендации движком и проверяет сценарий отзыва согласия
(анонимизация -> Cold Start). Используется для верификации модели данных там,
где production-стек недоступен.

Запуск:  python -m scripts.db_smoke
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from recsys.dataset import generate_all
from recsys.hybrid import HybridRecommender

DDL = """
CREATE TABLE platforms (platform_id TEXT PRIMARY KEY, name TEXT, kind TEXT);
CREATE TABLE hotels (hotel_id TEXT PRIMARY KEY, name TEXT, city TEXT, stars INT,
                     chain TEXT, rating REAL, poi_distance_km REAL);
CREATE TABLE offers (offer_id TEXT PRIMARY KEY, hotel_id TEXT, city TEXT, stars INT,
                     rating REAL, poi_distance_km REAL, room_type TEXT, price_per_night INT,
                     meal_plan TEXT, capacity INT, min_nights INT, season TEXT, amenities TEXT);
CREATE TABLE users (unified_id TEXT PRIMARY KEY, region TEXT, loyalty TEXT);
CREATE TABLE consents (id INTEGER PRIMARY KEY AUTOINCREMENT, unified_id TEXT, status TEXT, updated_at TEXT);
CREATE TABLE booking_history (id INTEGER PRIMARY KEY AUTOINCREMENT, unified_id TEXT, platform_id TEXT,
                     offer_id TEXT, hotel_id TEXT, booked_at TEXT, check_in TEXT, nights INT,
                     lead_time_days INT, season TEXT, amount INT, guests INT);
CREATE TABLE interaction_events (id INTEGER PRIMARY KEY AUTOINCREMENT, unified_id TEXT, session_id TEXT,
                     platform_id TEXT, offer_id TEXT, event TEXT, timestamp TEXT);
CREATE INDEX ix_history_user_time ON booking_history(unified_id, booked_at);
CREATE INDEX ix_events_user ON interaction_events(unified_id);
"""


def main():
    con = sqlite3.connect(":memory:")
    con.executescript(DDL)
    t = generate_all()

    # ---- загрузка данных ----
    t["platforms"].to_sql("platforms", con, if_exists="append", index=False)
    t["hotels"].to_sql("hotels", con, if_exists="append", index=False)
    t["offers"].to_sql("offers", con, if_exists="append", index=False)
    t["users"].rename(columns={"user_id": "unified_id"})[["unified_id", "region", "loyalty"]] \
        .to_sql("users", con, if_exists="append", index=False)
    t["users"].rename(columns={"user_id": "unified_id", "consent_status": "status"})[["unified_id", "status"]] \
        .assign(updated_at="2026-05-01").to_sql("consents", con, if_exists="append", index=False)
    t["history"].rename(columns={"user_id": "unified_id"}).drop(columns=["history_id"]) \
        .to_sql("booking_history", con, if_exists="append", index=False)
    t["interactions"].rename(columns={"user_id": "unified_id"}).drop(columns=["interaction_id"]) \
        .to_sql("interaction_events", con, if_exists="append", index=False)

    print("=== Загрузка в БД (sqlite) ===")
    for tbl in ["platforms", "hotels", "offers", "users", "consents", "booking_history", "interaction_events"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:20s}: {n}")

    # ---- проверка агрегации кросс-платформенной истории по unified_id ----
    sample = con.execute("""
        SELECT unified_id, COUNT(*) c, COUNT(DISTINCT platform_id) platforms
        FROM booking_history GROUP BY unified_id ORDER BY c DESC LIMIT 1""").fetchone()
    uid = sample[0]
    print(f"\n=== Кросс-платформенная история по unified_id={uid} ===")
    print(f"  бронирований: {sample[1]}, площадок: {sample[2]}")
    rows = con.execute("""SELECT platform_id, COUNT(*) FROM booking_history
                          WHERE unified_id=? GROUP BY platform_id""", (uid,)).fetchall()
    for pid, c in rows:
        print(f"    {pid}: {c}")

    # ---- сквозной поток: БД -> движок -> рекомендация ----
    granted = pd.read_sql("SELECT unified_id FROM consents WHERE status='granted'", con)["unified_id"].tolist()
    history = pd.read_sql("SELECT * FROM booking_history", con).rename(columns={"unified_id": "user_id"})
    inter = pd.read_sql("SELECT * FROM interaction_events", con).rename(columns={"unified_id": "user_id"})
    offers = pd.read_sql("SELECT * FROM offers", con)
    users = pd.read_sql("SELECT unified_id as user_id, region, loyalty FROM users", con)
    users["consent_status"] = users["user_id"].apply(lambda u: "granted" if u in set(granted) else "revoked")
    users["archetype_truth"] = t["users"].set_index("user_id")["archetype_truth"].reindex(users["user_id"]).values

    hist_g = history[history["user_id"].isin(granted)]
    inter_g = inter[inter["user_id"].isin(granted)]
    rec = HybridRecommender(offers).fit(hist_g, inter_g, users)
    pool = offers["offer_id"].tolist()[:30]
    out = rec.recommend(uid, pool, top_k=5)
    print(f"\n=== Рекомендация для {uid} (сегмент {out[0]['cluster_label']}) ===")
    for r in out:
        print(f"  #{r['rank']} {r['offer_id']} score={r['score']} "
              f"(cb={r['score_content']} cf={r['score_collab']} clu={r['score_cluster']})")

    # ---- сценарий 152-ФЗ: отзыв согласия -> анонимизация -> Cold Start ----
    print(f"\n=== Отзыв согласия unified_id={uid} (право на забвение) ===")
    before = con.execute("SELECT COUNT(*) FROM booking_history WHERE unified_id=?", (uid,)).fetchone()[0]
    con.execute("DELETE FROM booking_history WHERE unified_id=?", (uid,))
    con.execute("DELETE FROM interaction_events WHERE unified_id=?", (uid,))
    con.execute("UPDATE consents SET status='revoked' WHERE unified_id=?", (uid,))
    after = con.execute("SELECT COUNT(*) FROM booking_history WHERE unified_id=?", (uid,)).fetchone()[0]
    print(f"  история до/после: {before} -> {after} (удалено {before - after})")
    cold = rec.recommend(None, pool, questionnaire={"budget": "high", "trip_goal": "luxury"}, top_k=3)
    print(f"  Cold Start выдача: {[r['offer_id'] for r in cold]}, cold_start={cold[0]['cold_start']}")

    con.close()
    print("\nOK: схема БД и сквозной поток ingest -> история по unified_id -> рекомендация работают.")


if __name__ == "__main__":
    main()
