"""Add embedding_model_type to eval_runs

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-15

"""
from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_runs",
        sa.Column("embedding_model_type", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("eval_runs", "embedding_model_type")
