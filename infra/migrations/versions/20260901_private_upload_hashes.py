"""Scope duplicate uploaded-paper checks to the paper owner."""

from alembic import op


revision = "20260901_private_upload_hashes"
down_revision = "20260901_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("papers_external_id_key", "papers", type_="unique")
    op.create_index("ix_papers_external_id", "papers", ["external_id"])
    op.drop_constraint("documents_content_hash_key", "documents", type_="unique")
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])


def downgrade() -> None:
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.create_unique_constraint("documents_content_hash_key", "documents", ["content_hash"])
    op.drop_index("ix_papers_external_id", table_name="papers")
    op.create_unique_constraint("papers_external_id_key", "papers", ["external_id"])
