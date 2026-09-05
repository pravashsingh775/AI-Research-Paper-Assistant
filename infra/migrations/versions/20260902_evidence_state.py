"""Make paper evidence availability explicit."""
from alembic import op
import sqlalchemy as sa

revision = "20260902_evidence_state"
down_revision = "20260901_private_upload_hashes"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column("papers", sa.Column("evidence_state", sa.String(32), nullable=False, server_default="metadata-only"))
    op.create_index("ix_papers_evidence_state", "papers", ["evidence_state"])

def downgrade() -> None:
    op.drop_index("ix_papers_evidence_state", table_name="papers")
    op.drop_column("papers", "evidence_state")
