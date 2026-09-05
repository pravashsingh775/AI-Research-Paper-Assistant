"""Initial persistent research platform schema."""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260901_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table("users", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("email", sa.String(320), nullable=False, unique=True), sa.Column("password_hash", sa.String(255), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_table("papers", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("external_id", sa.String(255), unique=True), sa.Column("title", sa.Text(), nullable=False), sa.Column("abstract", sa.Text(), nullable=False), sa.Column("authors", sa.Text(), nullable=False), sa.Column("venue", sa.String(500)), sa.Column("year", sa.Integer()), sa.Column("doi", sa.String(500)), sa.Column("source", sa.String(80), nullable=False), sa.Column("citation_count", sa.Integer(), nullable=False, server_default="0"), sa.Column("url", sa.Text()), sa.Column("pdf_url", sa.Text()), sa.Column("owner_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE")), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_papers_doi", "papers", ["doi"])
    op.create_table("documents", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("paper_id", sa.UUID(), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False), sa.Column("filename", sa.String(255), nullable=False), sa.Column("content", sa.Text(), nullable=False, server_default=""), sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"), sa.Column("content_hash", sa.String(64), nullable=False, unique=True))
    op.create_index("ix_documents_paper_id", "documents", ["paper_id"])
    op.create_table("document_chunks", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False), sa.Column("text", sa.Text(), nullable=False), sa.Column("section", sa.String(255)), sa.Column("page", sa.Integer()), sa.Column("chunk_index", sa.Integer(), nullable=False), sa.Column("embedding", Vector(384)), sa.UniqueConstraint("document_id", "chunk_index"))
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_table("paper_analyses", sa.Column("paper_id", sa.UUID(), sa.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True), sa.Column("payload", sa.JSON(), nullable=False), sa.Column("model", sa.String(255), nullable=False), sa.Column("confidence", sa.Float()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_table("research_collections", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(255), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_research_collections_user_id", "research_collections", ["user_id"])
    op.create_table("collection_papers", sa.Column("collection_id", sa.UUID(), sa.ForeignKey("research_collections.id", ondelete="CASCADE"), primary_key=True), sa.Column("paper_id", sa.UUID(), sa.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True))
    op.create_table("chat_sessions", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("paper_id", sa.UUID(), sa.ForeignKey("papers.id", ondelete="CASCADE")), sa.Column("title", sa.String(255), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_table("chat_messages", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("session_id", sa.UUID(), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False), sa.Column("role", sa.String(20), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("citations", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_table("background_jobs", sa.Column("id", sa.UUID(), primary_key=True), sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE")), sa.Column("type", sa.String(80), nullable=False), sa.Column("status", sa.String(20), nullable=False), sa.Column("progress", sa.Integer(), nullable=False), sa.Column("error", sa.Text()), sa.Column("result", sa.JSON()), sa.Column("started_at", sa.DateTime(timezone=True)), sa.Column("completed_at", sa.DateTime(timezone=True)), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_background_jobs_user_id", "background_jobs", ["user_id"])
    op.create_index("ix_background_jobs_status", "background_jobs", ["status"])

def downgrade() -> None:
    for table in ("background_jobs", "chat_messages", "chat_sessions", "collection_papers", "research_collections", "paper_analyses", "document_chunks", "documents", "papers", "users"):
        op.drop_table(table)
    op.execute("DROP EXTENSION IF EXISTS vector")
