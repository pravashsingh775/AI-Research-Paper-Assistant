"""Add document storage fields and query performance indexes.

Revision ID: 20260903_storage_and_indexes
Revises: 20260902_evidence_state
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa

revision = "20260903_storage_and_indexes"
down_revision = "20260902_evidence_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Document storage columns
    op.add_column("documents", sa.Column("object_key", sa.String(500), nullable=True))
    op.add_column("documents", sa.Column("size_bytes", sa.BigInteger(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "mime_type", sa.String(100), nullable=True, server_default="application/pdf"
        ),
    )
    op.create_index("ix_documents_object_key", "documents", ["object_key"])

    # 2. Query performance indexes
    op.create_index("ix_papers_owner_id", "papers", ["owner_id"])
    op.create_index("ix_papers_created_at", "papers", ["created_at"])
    op.create_index("ix_chat_sessions_paper_id", "chat_sessions", ["paper_id"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])
    op.create_index(
        "ix_background_jobs_type_status", "background_jobs", ["type", "status"]
    )


def downgrade() -> None:
    op.drop_index("ix_background_jobs_type_status", table_name="background_jobs")
    op.drop_index("ix_chat_messages_created_at", table_name="chat_messages")
    op.drop_index("ix_chat_sessions_paper_id", table_name="chat_sessions")
    op.drop_index("ix_papers_created_at", table_name="papers")
    op.drop_index("ix_papers_owner_id", table_name="papers")

    op.drop_index("ix_documents_object_key", table_name="documents")
    op.drop_column("documents", "mime_type")
    op.drop_column("documents", "size_bytes")
    op.drop_column("documents", "object_key")
