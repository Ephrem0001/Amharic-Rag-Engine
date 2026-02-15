"""Add embedding_model and vector_id to chunks if missing

Revision ID: 0002_add_chunks_embedding_columns
Revises: 0001_initial_tables
Create Date: 2026-02-15

"""
from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001_initial_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Add columns only if they don't exist (e.g. DB was created from an older migration)
    r = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'vector_id'"
        )
    )
    if r.scalar() is None:
        op.add_column("chunks", sa.Column("vector_id", sa.Integer(), nullable=True))
    r = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding_model'"
        )
    )
    if r.scalar() is None:
        op.add_column("chunks", sa.Column("embedding_model", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "vector_id")
