"""Add stripe_customer_id to users table.

Revision ID: 002
Revises: 001
Create Date: 2026-03-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("stripe_customer_id", sa.String(255), nullable=True, unique=True),
    )
    op.create_index(
        "idx_users_stripe_customer", "users", ["stripe_customer_id"], unique=True
    )


def downgrade() -> None:
    op.drop_index("idx_users_stripe_customer", table_name="users")
    op.drop_column("users", "stripe_customer_id")
