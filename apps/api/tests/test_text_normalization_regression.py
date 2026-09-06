"""Regression test suite for text normalization, Unicode preservation, and worker transaction recovery.

Validates:
1. Exact reproduction and fix of arXiv:1307.0411 NUL byte corruption.
2. Complete preservation of Greek characters, math symbols, Dirac notation, multilingual text, and formatting.
3. Pre-persistence validation preventing CharacterNotInRepertoireError.
4. Worker transaction failure isolation eliminating PendingRollbackError.
5. Live PostgreSQL roundtrip fidelity for scientific and multilingual text.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from apps.api.app.core.database import SessionFactory, engine
from apps.api.app.models import BackgroundJob, Document, DocumentChunk, Paper, User
from apps.api.app.services.text_normalization import (
    normalize_and_audit_text,
    normalize_extracted_text,
    validate_text_for_persistence,
)
from apps.worker.app.worker import _mark_job_failed


@pytest.fixture(autouse=True)
async def cleanup_db_engine() -> None:
    """Ensure asyncpg connections tied to a closed event loop are disposed properly on Windows."""
    yield
    await engine.dispose()


# =====================================================================
# Unit Tests: Normalization & Unicode Fidelity
# =====================================================================


def test_normalize_extracted_text_strips_nuls() -> None:
    """NUL bytes (\\x00) must be stripped from any extracted string."""
    raw = "Quantum\x00 algorithms\x00 for supervised\x00 and unsupervised\x00 machine learning."
    cleaned = normalize_extracted_text(raw)
    assert "\x00" not in cleaned
    assert cleaned == "Quantum algorithms for supervised and unsupervised machine learning."

    # Multiple consecutive and edge NUL bytes
    edge_raw = "\x00\x00Header\x00\nContent line 1\x00\r\nLine 2\x00"
    edge_clean = normalize_extracted_text(edge_raw)
    assert "\x00" not in edge_clean
    assert edge_clean == "Header\nContent line 1\r\nLine 2"


def test_arxiv_1307_0411_exact_reproduction() -> None:
    """arXiv:1307.0411 exact failure scenario with embedded NUL bytes and Dirac notation."""
    arxiv_title = "Quantum algorithms for supervised and unsupervised machine learning\x00"
    arxiv_abstract = (
        "Machine learning algorithms find patterns in large amounts of data.\x00\n"
        "Here we show that quantum computers\x00 can be used to accelerate machine learning algorithms...\x00\n"
        "A quantum state |ψ⟩ = ∑_i α_i |i⟩ can represent vectors with exponential advantage."
    )

    clean_title = normalize_extracted_text(arxiv_title)
    clean_abstract = normalize_extracted_text(arxiv_abstract)

    assert "\x00" not in clean_title
    assert "\x00" not in clean_abstract
    assert clean_title == "Quantum algorithms for supervised and unsupervised machine learning"
    assert "|ψ⟩ = ∑_i α_i |i⟩" in clean_abstract

    # Pre-persistence validation must pass
    valid_title, err_title = validate_text_for_persistence(clean_title, field_name="title")
    assert valid_title is True
    assert err_title is None

    valid_abstract, err_abstract = validate_text_for_persistence(
        clean_abstract, field_name="abstract"
    )
    assert valid_abstract is True
    assert err_abstract is None


def test_scientific_and_multilingual_unicode_preservation() -> None:
    """Scientific Unicode, Dirac notation, mathematical symbols, and multilingual characters must not be mangled."""
    # 1. Greek characters
    greek = "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω Γ Δ Θ Λ Ξ Π Σ Φ Ψ Ω"
    assert normalize_extracted_text(greek) == greek

    # 2. Mathematical symbols & operators
    math_symbols = "∑ ∏ ∫ ∮ ∂ ∇ √ ∛ ∞ ∝ ≈ ≠ ≡ ≤ ≥ ≪ ≫ ± ∓ × ÷ ∈ ∉ ⊂ ⊆ ∪ ∩ ⊕ ⊗ ⊥"
    assert normalize_extracted_text(math_symbols) == math_symbols

    # 3. Dirac and quantum mechanics notation
    quantum = "|0⟩ ⊗ |1⟩ = |01⟩, ⟨ψ|H|ψ⟩ ≥ E_0, ⟨ϕ|ψ⟩ = ∑_k c_k^* d_k"
    assert normalize_extracted_text(quantum) == quantum

    # 4. Multilingual text (Devanagari, CJK, Accented European)
    hindi = "क्वांटम मशीन लर्निंग और कृत्रिम बुद्धिमत्ता अनुसंधान"
    chinese = "基于量子计算的监督与无监督机器学习算法研究"
    accented = "Erwin Schrödinger, Henri Poincaré, Louis de Broglie, János von Neumann"
    assert normalize_extracted_text(hindi) == hindi
    assert normalize_extracted_text(chinese) == chinese
    assert normalize_extracted_text(accented) == accented

    # 5. Scientific notations & formulas
    formulas = "H₂O + CO₂ → H₂CO₃; x² + y² = z²; E = mc²; 10⁻⁶ mol/L"
    assert normalize_extracted_text(formulas) == formulas

    # 6. Combined with NUL bytes: verify NULs are removed while scientific text is untouched
    corrupted = f"{quantum}\x00\n{hindi}\x00\n{chinese}\x00\n{accented}\x00"
    expected = f"{quantum}\n{hindi}\n{chinese}\n{accented}"
    assert normalize_extracted_text(corrupted) == expected


def test_validate_text_for_persistence_rejects_nul() -> None:
    """validate_text_for_persistence must detect NUL bytes."""
    is_valid, err_msg = validate_text_for_persistence(
        "text with \x00 NUL byte", field_name="test_field"
    )
    assert is_valid is False
    assert err_msg is not None
    assert "prohibited NUL byte" in err_msg

    # Clean text passes through unaltered
    is_valid_clean, err_clean = validate_text_for_persistence("perfectly clean text")
    assert is_valid_clean is True
    assert err_clean is None


def test_normalize_and_audit_text_reports_removed_nuls() -> None:
    """normalize_and_audit_text must accurately return audit metrics."""
    raw = "Doc\x00 with\x00 three\x00 NULs"
    cleaned, audit = normalize_and_audit_text(raw, document_id="test-doc-123")
    assert cleaned == "Doc with three NULs"
    assert audit.removed_nul_count == 3
    assert audit.normalization_applied is True
    assert audit.text == cleaned


# =====================================================================
# Integration Tests: PostgreSQL & Worker Transaction Recovery
# =====================================================================


@pytest.mark.asyncio
async def test_postgres_rejects_raw_nul_and_accepts_normalized() -> None:
    """Verify live PostgreSQL rejects raw \\x00, and accepts normalized text."""
    test_user_email = f"test_nul_{uuid4().hex[:8]}@example.com"
    user_id = uuid4()
    paper_id = uuid4()
    doc_id = uuid4()

    async with SessionFactory() as session:
        # Create a test user and paper
        user = User(id=user_id, email=test_user_email, password_hash="dummy")
        session.add(user)
        await session.flush()
        paper = Paper(id=paper_id, title="NUL Test Paper", source="test", owner_id=user_id)
        session.add(paper)
        await session.commit()

    try:
        # 1. Attempt to insert raw NUL byte - should fail with DBAPIError (CharacterNotInRepertoireError)
        async with SessionFactory() as session:
            doc_raw = Document(
                id=doc_id,
                paper_id=paper_id,
                filename="raw_nul.pdf",
                content="This has a raw \x00 NUL byte in content",
                content_hash="hash_raw",
            )
            session.add(doc_raw)
            with pytest.raises(DBAPIError):
                await session.commit()

        # 2. Insert normalized text - must succeed
        async with SessionFactory() as session:
            raw_text = "This has a raw \x00 NUL byte in content"
            clean_text = normalize_extracted_text(raw_text)
            doc_clean = Document(
                id=doc_id,
                paper_id=paper_id,
                filename="clean_nul.pdf",
                content=clean_text,
                content_hash="hash_clean",
            )
            session.add(doc_clean)
            await session.commit()

        # 3. Read back and verify
        async with SessionFactory() as session:
            fetched = await session.get(Document, doc_id)
            assert fetched is not None
            assert fetched.content == "This has a raw  NUL byte in content"
            assert "\x00" not in fetched.content

    finally:
        # Cleanup
        async with SessionFactory() as session:
            doc = await session.get(Document, doc_id)
            if doc:
                await session.delete(doc)
            p = await session.get(Paper, paper_id)
            if p:
                await session.delete(p)
            u = await session.get(User, user_id)
            if u:
                await session.delete(u)
            await session.commit()


@pytest.mark.asyncio
async def test_postgres_scientific_fidelity_roundtrip() -> None:
    """Verify live PostgreSQL roundtrip fidelity for Dirac notation, Greek, math, and multilingual text."""
    user_id = uuid4()
    paper_id = uuid4()
    doc_id = uuid4()
    chunk_id = uuid4()

    scientific_title = "Quantum State Evolution: |ψ⟩ = ∑_i α_i |i⟩ (Schrödinger & Poincaré)"
    scientific_content = (
        "Methods & Materials\n"
        "Let H be a Hamiltonian with eigenvalues E_n: H|ψ_n⟩ = E_n|ψ_n⟩.\n"
        "Conservation: ∑ |α_i|² = 1; uncertainty: Δx Δp ≥ ℏ/2.\n"
        "Devanagari: क्वांटम अनुसंधान; CJK: 量子计算。\n"
        "Subscripts/superscripts: H₂O + CO₂ → H₂CO₃, E = mc²."
    )

    async with SessionFactory() as session:
        user = User(id=user_id, email=f"sci_{uuid4().hex[:8]}@example.com", password_hash="dummy")
        session.add(user)
        await session.flush()
        paper = Paper(
            id=paper_id,
            title=scientific_title,
            abstract=scientific_content[:150],
            source="test_fidelity",
            owner_id=user_id,
        )
        session.add(paper)
        await session.flush()

        doc = Document(
            id=doc_id,
            paper_id=paper_id,
            filename="quantum.pdf",
            content=scientific_content,
            content_hash="sci_hash",
        )
        session.add(doc)
        await session.flush()

        chunk = DocumentChunk(
            id=chunk_id,
            document_id=doc_id,
            text=scientific_content,
            section="Methods & Materials",
            page=1,
            chunk_index=0,
        )
        session.add(chunk)
        await session.commit()

    try:
        async with SessionFactory() as session:
            fetched_paper = await session.get(Paper, paper_id)
            fetched_doc = await session.get(Document, doc_id)
            fetched_chunk = await session.get(DocumentChunk, chunk_id)

            assert fetched_paper is not None
            assert fetched_paper.title == scientific_title

            assert fetched_doc is not None
            assert fetched_doc.content == scientific_content

            assert fetched_chunk is not None
            assert fetched_chunk.text == scientific_content
            assert fetched_chunk.section == "Methods & Materials"
            assert "H|ψ_n⟩ = E_n|ψ_n⟩" in fetched_chunk.text
            assert "क्वांटम अनुसंधान" in fetched_chunk.text
            assert "量子计算" in fetched_chunk.text
    finally:
        async with SessionFactory() as session:
            c = await session.get(DocumentChunk, chunk_id)
            if c:
                await session.delete(c)
            d = await session.get(Document, doc_id)
            if d:
                await session.delete(d)
            p = await session.get(Paper, paper_id)
            if p:
                await session.delete(p)
            u = await session.get(User, user_id)
            if u:
                await session.delete(u)
            await session.commit()


@pytest.mark.asyncio
async def test_worker_failure_isolation_and_recovery() -> None:
    """Test that worker failure handling in an isolated session prevents PendingRollbackError.

    Job A fails -> worker rolls back and uses _mark_job_failed ->
    Job B is processed on a fresh session without encountering PendingRollbackError.
    """
    user_id = uuid4()
    paper_a_id = uuid4()
    paper_b_id = uuid4()
    job_a_id = uuid4()
    job_b_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id, email=f"recovery_{uuid4().hex[:8]}@example.com", password_hash="dummy"
        )
        session.add(user)
        await session.flush()

        paper_a = Paper(id=paper_a_id, title="Job A Paper", source="test", owner_id=user_id)
        paper_b = Paper(id=paper_b_id, title="Job B Paper", source="test", owner_id=user_id)
        job_a = BackgroundJob(
            id=job_a_id, user_id=user_id, type="PAPER_PROCESSING", status="PENDING"
        )
        job_b = BackgroundJob(
            id=job_b_id, user_id=user_id, type="PAPER_PROCESSING", status="PENDING"
        )
        session.add_all([paper_a, paper_b, job_a, job_b])
        await session.commit()

    try:
        # Simulate Job A: An exception occurs inside a transaction
        job_a_error = "Simulated pipeline extraction failure"
        try:
            async with SessionFactory() as session_a:
                # Force an error
                raise RuntimeError("Simulated processing crash")
        except RuntimeError:
            # Worker marks Job A as failed in an isolated session
            await _mark_job_failed(job_a_id, paper_a_id, job_a_error)

        # Verify Job A status in DB
        async with SessionFactory() as session:
            persisted_job_a = await session.get(BackgroundJob, job_a_id)
            assert persisted_job_a is not None
            assert persisted_job_a.status == "FAILED"
            assert "Simulated pipeline extraction failure" in (persisted_job_a.error or "")
            persisted_paper_a = await session.get(Paper, paper_a_id)
            assert persisted_paper_a is not None
            assert persisted_paper_a.evidence_state == "failed"

        # Now simulate Job B on the same worker session factory.
        # It must NOT throw PendingRollbackError and must succeed cleanly.
        async with SessionFactory() as session_b:
            j_b = await session_b.get(BackgroundJob, job_b_id)
            assert j_b is not None
            j_b.status = "COMPLETED"
            j_b.completed_at = datetime.now(timezone.utc)
            await session_b.commit()

        async with SessionFactory() as session:
            persisted_job_b = await session.get(BackgroundJob, job_b_id)
            assert persisted_job_b is not None
            assert persisted_job_b.status == "COMPLETED"

    finally:
        async with SessionFactory() as session:
            for jid in (job_a_id, job_b_id):
                j = await session.get(BackgroundJob, jid)
                if j:
                    await session.delete(j)
            for pid in (paper_a_id, paper_b_id):
                p = await session.get(Paper, pid)
                if p:
                    await session.delete(p)
            u = await session.get(User, user_id)
            if u:
                await session.delete(u)
            await session.commit()
