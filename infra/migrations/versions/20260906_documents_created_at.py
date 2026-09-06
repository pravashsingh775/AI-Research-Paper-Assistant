"""Add created_at timestamp to documents table.

Revision ID: 20260906_documents_created_at
Revises: 20260903_storage_and_indexes
Create Date: 2026-09-06
"""

from alembic import op
import sqlalchemy as sa

revision = "20260906_documents_created_at"
down_revision = "20260903_storage_and_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_created_at", "documents", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_documents_created_at", table_name="documents")
    op.drop_column("documents", "created_at")
