"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2026-03-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("oauth_provider", sa.String(20), nullable=False),
        sa.Column("oauth_subject", sa.String(255), nullable=False),
        sa.Column(
            "subscription_tier", sa.String(20), nullable=False, server_default="free"
        ),
        sa.Column("has_contributed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("oauth_provider", "oauth_subject", name="uq_users_oauth"),
    )
    op.create_index("idx_users_oauth", "users", ["oauth_provider", "oauth_subject"])

    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("refresh_token_hash", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_sessions_user", "sessions", ["user_id"])
    op.create_index("idx_sessions_refresh_hash", "sessions", ["refresh_token_hash"])

    op.create_table(
        "anonymous_transitions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_a_fingerprint", sa.String(255), nullable=False),
        sa.Column("track_b_fingerprint", sa.String(255), nullable=False),
        sa.Column("contributed_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "community_scores",
        sa.Column("track_a_fingerprint", sa.String(255), primary_key=True),
        sa.Column("track_b_fingerprint", sa.String(255), primary_key=True),
        sa.Column("frequency", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("community_scores")
    op.drop_table("anonymous_transitions")
    op.drop_table("sessions")
    op.drop_table("users")
