"""Initial tables

Revision ID: 0001_initial_tables
Revises: 
Create Date: 2026-02-13 16:00:37

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001_initial_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )

    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_pages_document_id", "pages", ["document_id"])
    op.create_index("ix_pages_document_id_page_number", "pages", ["document_id", "page_number"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("vector_id", sa.Integer(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_document_id_page_number", "chunks", ["document_id", "page_number"])

    op.create_table(
        "retrieval_pairs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("positive_chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chunks.id"), nullable=False),
        sa.Column("hard_negative_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index("ix_retrieval_pairs_document_id", "retrieval_pairs", ["document_id"])

    op.create_table(
        "eval_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_pages", sa.JSON(), nullable=True),
        sa.Column("expected_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index("ix_eval_questions_document_id", "eval_questions", ["document_id"])

    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("run_name", sa.String(), nullable=False),
        sa.Column("embedding_model_name", sa.String(), nullable=True),
        sa.Column("embedding_model_path", sa.String(), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=True),
        sa.Column("recall_at_k", sa.Float(), nullable=True),
        sa.Column("mrr_at_k", sa.Float(), nullable=True),
        sa.Column("ndcg_at_k", sa.Float(), nullable=True),
        sa.Column("hit_rate_at_k", sa.Float(), nullable=True),
        sa.Column("mean_latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_index("ix_eval_questions_document_id", table_name="eval_questions")
    op.drop_table("eval_questions")
    op.drop_index("ix_retrieval_pairs_document_id", table_name="retrieval_pairs")
    op.drop_table("retrieval_pairs")
    op.drop_index("ix_chunks_document_id_page_number", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_pages_document_id_page_number", table_name="pages")
    op.drop_index("ix_pages_document_id", table_name="pages")
    op.drop_table("pages")
    op.drop_table("documents")
