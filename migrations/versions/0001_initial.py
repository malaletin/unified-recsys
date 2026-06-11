"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-11
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platforms",
        sa.Column("platform_id", sa.String(8), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
    )
    op.create_table(
        "hotels",
        sa.Column("hotel_id", sa.String(8), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("city", sa.String(64), nullable=False),
        sa.Column("stars", sa.Integer, nullable=False),
        sa.Column("chain", sa.String(64), nullable=False),
        sa.Column("rating", sa.Float, nullable=False),
        sa.Column("poi_distance_km", sa.Float, nullable=False),
    )
    op.create_index("ix_hotels_city", "hotels", ["city"])

    op.create_table(
        "offers",
        sa.Column("offer_id", sa.String(12), primary_key=True),
        sa.Column("hotel_id", sa.String(8), sa.ForeignKey("hotels.hotel_id"), nullable=False),
        sa.Column("city", sa.String(64), nullable=False),
        sa.Column("stars", sa.Integer, nullable=False),
        sa.Column("rating", sa.Float, nullable=False),
        sa.Column("poi_distance_km", sa.Float, nullable=False),
        sa.Column("room_type", sa.String(32), nullable=False),
        sa.Column("price_per_night", sa.Integer, nullable=False),
        sa.Column("meal_plan", sa.String(8), nullable=False),
        sa.Column("capacity", sa.Integer, nullable=False),
        sa.Column("min_nights", sa.Integer, nullable=False),
        sa.Column("season", sa.String(8), nullable=False),
        sa.Column("amenities", sa.JSON, nullable=True),
    )
    op.create_index("ix_offers_hotel_id", "offers", ["hotel_id"])
    op.create_index("ix_offers_city", "offers", ["city"])
    op.create_index("ix_offers_price_per_night", "offers", ["price_per_night"])

    op.create_table(
        "users",
        sa.Column("unified_id", sa.String(32), primary_key=True),
        sa.Column("region", sa.String(64), nullable=True),
        sa.Column("loyalty", sa.String(16), nullable=False, server_default="new"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "consents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("unified_id", sa.String(32),
                  sa.ForeignKey("users.unified_id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("purpose", sa.String(64), server_default="personalization"),
        sa.Column("source", sa.String(64), server_default="onboarding_checkbox"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_consents_unified_id", "consents", ["unified_id"])

    op.create_table(
        "booking_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("unified_id", sa.String(32),
                  sa.ForeignKey("users.unified_id", ondelete="CASCADE"), nullable=False),
        sa.Column("platform_id", sa.String(8), sa.ForeignKey("platforms.platform_id")),
        sa.Column("offer_id", sa.String(12), sa.ForeignKey("offers.offer_id")),
        sa.Column("hotel_id", sa.String(8)),
        sa.Column("booked_at", sa.DateTime(timezone=True)),
        sa.Column("check_in", sa.DateTime(timezone=True)),
        sa.Column("nights", sa.Integer),
        sa.Column("lead_time_days", sa.Integer, server_default="0"),
        sa.Column("season", sa.String(8), server_default="any"),
        sa.Column("amount", sa.Integer),
        sa.Column("guests", sa.Integer, server_default="1"),
    )
    op.create_index("ix_booking_history_unified_id", "booking_history", ["unified_id"])
    op.create_index("ix_booking_history_booked_at", "booking_history", ["booked_at"])
    op.create_index("ix_history_user_time", "booking_history", ["unified_id", "booked_at"])

    op.create_table(
        "interaction_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("unified_id", sa.String(32),
                  sa.ForeignKey("users.unified_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(32)),
        sa.Column("platform_id", sa.String(8), sa.ForeignKey("platforms.platform_id")),
        sa.Column("offer_id", sa.String(12), sa.ForeignKey("offers.offer_id")),
        sa.Column("event", sa.String(16)),
        sa.Column("timestamp", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_interaction_events_unified_id", "interaction_events", ["unified_id"])
    op.create_index("ix_interaction_events_timestamp", "interaction_events", ["timestamp"])

    op.create_table(
        "user_profiles",
        sa.Column("unified_id", sa.String(32),
                  sa.ForeignKey("users.unified_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("cluster_id", sa.Integer, nullable=True),
        sa.Column("cluster_label", sa.String(32), nullable=True),
        sa.Column("history_len", sa.Integer, server_default="0"),
        sa.Column("preferences", sa.JSON, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "recommendation_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("unified_id", sa.String(32), nullable=True),
        sa.Column("platform_id", sa.String(8)),
        sa.Column("requested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("cold_start", sa.Boolean, server_default=sa.false()),
        sa.Column("offers", sa.JSON, nullable=True),
        sa.Column("context", sa.JSON, nullable=True),
    )
    op.create_index("ix_recommendation_logs_unified_id", "recommendation_logs", ["unified_id"])


def downgrade() -> None:
    for t in ["recommendation_logs", "user_profiles", "interaction_events",
              "booking_history", "consents", "users", "offers", "hotels", "platforms"]:
        op.drop_table(t)
